from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from enum import Enum
import copy
import imageio
from largesteps.parameterize import to_differential, from_differential
from largesteps.optimize import AdamUniform
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt
from pathlib import Path
import math
from umbra import MeshViewer
import torch
from tqdm import tqdm
from typing import Optional

from diffshadow.common import vflip, to_display_image
from diffshadow.simple_renderer import SimpleRenderer, Mesh, Camera, azimuth_elevation_to_direction
from diffshadow.transformation import create_lookat_matrix, create_perspective_projection_matrix, create_rotation_matrix

from drpg import *
from drpg.rendering import *

def create_camera(azimuth: float, elevation: float, distance: float, device: torch.device, focus=None):
    if focus is None:
        focus=[0, -0.5, 0]
    return Camera(
        view_matrix=create_lookat_matrix(eye=distance*azimuth_elevation_to_direction(azimuth, elevation), focus=focus, up=[0, 1, 0]).to(device),
        projection_matrix = create_perspective_projection_matrix(fovy=45, device=device)
    )

def create_cameras(elevation: float = math.pi/4, num_cameras: int = 9, distance: float = 3, focus=None, device: torch.device = None):
    return [
        create_camera(2*math.pi*i/num_cameras, elevation, distance, focus=focus, device=device) for i in range(0, num_cameras)
    ]

def load_config(path: Path) -> dict:
    config = load_json(path)

    if "include" in config:
        print(config)
        include_path: Path = Path(f'{config["include"]}.json')
        print(f"Including {include_path.name} for config {path.name}")
        config_base = load_config(path.parent / include_path)
        config_ = config_base.copy()
        config_.update(config)
        
        config = config_
        config.pop("include")

    return config

class InitialSurface(Enum):
    Cube = 0

class TargetVariables(Enum):
    ControlPoints = 0
    Vertices      = 1

def run_multiview_objects(output_dir, input_path: Path, transform: Optional[List[float]], initial_surface: InitialSurface, initial_scale: float, control_grid_resolution: int, uv_resolution: int, adaptive_tess_threshold: Optional[float], subdivision_schedule: List[int], elevation_schedule: List[int], large_steps_lambda: Optional[float], image_loss_weight: Optional[float], g1_loss_weight: Optional[float], lr: float, lr_decay: Optional[float], num_iterations:int, output_checkpoints: List[int], output_interval: int, output_image_indices: int, no_output: bool, viewer: MeshViewer, seed=0):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Basic setup
    device = torch.device('cuda:0')
    context = dr.RasterizeGLContext(device=device)

    # Setup the general scene
    resolution = (256, 256)

    cameras = \
    create_cameras(elevation=math.radians(35), focus=[0, 0, 0], num_cameras=9, device=device) + \
    create_cameras(elevation=math.radians(5), focus=[0, 0, 0], num_cameras=5, device=device) + \
    create_cameras(elevation=-math.pi/5, focus=[0, 0, 0], num_cameras=5, device=device)

    envmap = EnvironmentMap(load_environment_map("data/kloppenheim_06_2k.hdr", device=device))
    sh     = SphericalHarmonics(envmap.envmap)

    # Setup the rendering function
    renderer = SimpleRenderer(context)
    def render_image(meshes: List[Mesh], i: int):
        gbuffer = renderer._gbuffer_pass(mesh=meshes[0], camera=cameras[i], resolution=resolution)
        rays    = get_view_rays(cameras[i].view_matrix, cameras[i].projection_matrix, resolution)
        background = sh.eval(rays)/math.pi
        normal = gbuffer['normal']
        img = torch.where(gbuffer['mask'][:, :, None] > 0, sh.eval(normal) / math.pi, vflip(background))
        img = dr.antialias(img[None].contiguous(), gbuffer['rast_out'], gbuffer['v_clipspace'], gbuffer['f'])[0]

        mask  = dr.antialias(gbuffer['mask'][None, :, :, None].to(dtype=torch.float32), gbuffer['rast_out'], gbuffer['v_clipspace'], gbuffer['f'])[0]

        return img, mask

    # Load the reference object
    v, f, n = load_mesh(input_path, device=device)
    if "abc" in input_path.stem or "ABC" in input_path.stem:
        # Rotate ABC objects y-up
        R = create_rotation_matrix(math.radians(90), 1, 2, device=device)
        v = apply_transformation(v, R, mode="euclidean")
        v = v*0.8
    elif transform is not None:
        T = torch.tensor(transform, dtype=torch.float32, device=device).reshape(4, 4)
        v = apply_transformation(v, T, mode="affine")

    # Render the reference images
    v_ref = v
    f_ref = f
    mesh_ref = Mesh(v_ref, f_ref)
    imgs_ref = [render_image([mesh_ref], i) for i in range(len(cameras))]

    save_mesh(output_dir / "reference.obj", mesh_ref.vertices, mesh_ref.faces, vertex_normals=mesh_ref.normals.cpu().numpy())

    if viewer:
        viewer.set_mesh(v_ref.detach().cpu().numpy(), f_ref.cpu().numpy(), c=[0, 1, 0], object_name="surface_mesh_opt")

    if not no_output:
        for i, (img_ref, mask_ref) in enumerate(imgs_ref):
            imageio.imwrite(output_dir / f"reference_color_{i}.png", to_display_image(img_ref, grayscale_to_rgb=True, to_uint8=True))
            imageio.imwrite(output_dir / f"reference_mask_{i}.png",  to_display_image(mask_ref, grayscale_to_rgb=True, to_uint8=True))

    # Define the initial surface 
    if initial_surface == InitialSurface.Cube:
        print(f"n={control_grid_resolution}")
        patches = bezier_cube(n=control_grid_resolution, device=device)
        patches.V *= initial_scale
    else:
        raise RuntimeError(f"Initial Surface '{str(initial_surface)}' not implemented")

    v_init, f_init, tu_init, tv_init = SurfaceTessellator((16, 16), SurfaceType.Bezier).tessellate(patches)
    n_init                           = get_normals_from_tangents(tu_init, tv_init)
    mesh_init = Mesh(v_init, f_init, n_init)
    if viewer:
        viewer.set_mesh(v_init.detach().cpu().numpy(), f_init.cpu().numpy(), c=[0, 1, 0], object_name="surface_mesh_opt")

    # Set up the tessellator
    tessellator = SurfaceTessellator((uv_resolution, uv_resolution), SurfaceType.Bezier)

    use_large_steps = large_steps_lambda is not None
    num_setups      = 1
    lr_decay        = lr_decay if lr_decay is not None else 1.0
    def setup_optimizer(patches: ParametricPatches, num_setups):
        patches_opt = copy.deepcopy(patches)

        if use_large_steps:
            # Compute the system matrix
            V, F = convert_patches_to_mesh(patches_opt, MeshType.Quad)
            M    = compute_lsig_matrix(V, F, lambda_=large_steps_lambda)

            # Parameterize
            u_opt = to_differential(M, patches_opt.V)
            u_opt.requires_grad = True
            print(large_steps_lambda, lr, lr_decay, num_setups)
            optimizer = AdamUniform([u_opt], lr=lr*lr_decay**num_setups)
        else:
            patches_opt.V.requires_grad = True
            optimizer = torch.optim.Adam([patches_opt.V], lr=lr*lr_decay**num_setups)

        def compute_mesh():
            if use_large_steps:
                patches_opt.V = from_differential(M, u_opt, 'Cholesky')

            v_opt, f_opt, tu_opt, tv_opt = tessellator.tessellate(patches_opt, return_per_patch=True, adaptive_threshold=adaptive_tess_threshold)
            n_opt                        = get_normals_from_tangents(tu_opt, tv_opt)
            return v_opt, f_opt, n_opt
        
        return optimizer, patches_opt, compute_mesh, num_setups + 1

    optimizer, patches_opt, compute_mesh, num_setups = setup_optimizer(patches, num_setups)

    tangent_map  = build_tangent_map(patches_opt)
    patch_colors = torch.rand((patches_opt.F.shape[0], 3)) + 0.4
    
    profiler = Profiler()
    profiler.start()

    imgs_initial = None
    progress_bar = tqdm(range(num_iterations))
    frame_index  = 0 
    for iteration in progress_bar:
        if iteration in subdivision_schedule:
            with torch.no_grad():
                patches_subdiv = ParametricPatches(n=patches_opt.n, device=device)
                patches_subdiv.add(subdivide_bezier_surface(patches_opt.V[patches_opt.F]))
                patches_subdiv.merge_duplicate_vertices()
                optimizer, patches_opt, compute_mesh, num_setups = setup_optimizer(patches_subdiv, num_setups)
                tangent_map                                      = build_tangent_map(patches_opt)
                patch_colors                                     = torch.rand((patches_opt.F.shape[0], 3)) + 0.4
        
        if iteration in elevation_schedule:
            with torch.no_grad():
                patches_elevated = ParametricPatches(n=patches_opt.n+1, device=device)
                patches_elevated.add(elevate_bezier_surface_degree(patches_opt.V[patches_opt.F]))
                patches_elevated.merge_duplicate_vertices()
                optimizer, patches_opt, compute_mesh, num_setups = setup_optimizer(patches_elevated, num_setups)
                tangent_map                                      = build_tangent_map(patches_opt)
                patch_colors                                     = torch.rand((patches_opt.F.shape[0], 3)) + 0.4

        # Get the current mesh and render it
        v_opt, f_opt, n_opt = compute_mesh()
        mesh_opt = Mesh(v_opt.reshape(-1, 3), f_opt.reshape(-1, 3).to(dtype=torch.int32), n_opt.reshape(-1, 3))
        imgs_opt = [render_image([mesh_opt], i) for i in range(len(cameras))]
        
        if iteration == 0:
            imgs_initial = [img.detach().clone() for (img, mask) in imgs_opt]

        loss = 0.0
        for (img_ref, mask_ref), (img_opt, mask_opt) in zip(imgs_ref, imgs_opt):
            loss += torch.mean((mask_ref - mask_opt).abs())
            loss += image_loss_weight*torch.mean((img_ref - img_opt).abs())

        if g1_loss_weight is not None and g1_loss_weight != 0.0: 
            loss += g1_loss_weight*g1_loss(patches_opt, tangent_map=tangent_map)

        progress_bar.set_postfix({'loss': f"{float(loss):0.5f}"})

        mesh_union_opt = merge_meshes([mesh_opt])
        if viewer:
            viewer.set_mesh(mesh_union_opt.vertices.detach().cpu().numpy(), mesh_union_opt.faces.cpu().numpy(), n=n.detach().cpu().numpy(), c=[0, 1, 0], object_name="surface_mesh_opt")

        if (not no_output) and (output_interval > 0) and (iteration % output_interval == 0):
            for image_idx in output_image_indices:
                image_output_dir = output_dir / "images" / str(image_idx)
                image_output_dir.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(image_output_dir / f"{frame_index}.png", to_display_image(imgs_opt[image_idx][0], grayscale_to_rgb=True, to_uint8=True))

            frame_output_dir = output_dir / "patches" / str(frame_index)
            frame_output_dir.mkdir(parents=True, exist_ok=True)
            frame_index += 1
            with torch.no_grad():
                for i in range(len(v_opt)):
                    save_mesh(frame_output_dir / f"surface_{i}.obj", v_opt[i].reshape(-1, 3).cpu().numpy(), f_opt[0].cpu().numpy(), vertex_normals=n_opt[i].reshape(-1, 3).cpu().numpy())
                
                for i in range(len(patches_opt.F)):
                    P = patches_opt.V[patches_opt.F[i]]
                    F = generate_grid_faces(P.shape[0], P.shape[1], quads=True)
                    write_quad_mesh(frame_output_dir / f"teaser_controlmesh_{i}.obj", P.reshape(-1, 3), F)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    profiler.stop()

    if no_output:
        return

    profiler.export(output_dir / "runtime.json")

    save_mesh(output_dir / "initial.obj", mesh_init.vertices, mesh_init.faces)
    save_mesh(output_dir / "optimized.obj", mesh_opt.vertices, mesh_opt.faces)

    for i, (img_initial, (img_opt, mask_opt)) in enumerate(zip(imgs_initial, imgs_opt)):
        imageio.imwrite(output_dir / f"initial_{i}.png",   to_display_image(img_initial, grayscale_to_rgb=True, to_uint8=True))
        imageio.imwrite(output_dir / f"optimized_{i}.png", to_display_image(img_opt, grayscale_to_rgb=True, to_uint8=True))

    # Save the control point patches
    patches_opt.save(output_dir / "control_mesh.npz")

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", type=Path, help="Path to the configuration file")
    parser.add_argument("--viewer", default=False, action='store_true', help="Indicator if the viewer should be used")
    parser.add_argument("--output_interval", type=int, default=0, help="Interval for saving optimization outputs (default: 0/off)")
    parser.add_argument("--output_image_indices", type=int, default=[0], nargs='+', help="Indices of the image to output during optimization")
    parser.add_argument("--no_output", default=False, action='store_true', help="Indicator if output is produced")
    args = parser.parse_args()

    viewer = MeshViewer() if args.viewer else None

    # Load the config
    print(f"Loading config '{args.config}'")
    config = load_config(args.config)

    # Build the output dir from the config
    if "output_dir" not in config:
        raise RuntimeError("Expected output directory")
    output_dir = Path(config["output_dir"])

    initial_map = {
        'cube': InitialSurface.Cube
    }

    config["initial_surface"]         = initial_map[config.get("initial_surface", "cube")]
    config["initial_scale"]           = config.get("initial_scale", 1.0)
    config["control_grid_resolution"] = config.get("control_grid_resolution", 4)
    config["uv_resolution"]           = config.get("uv_resolution", 8)
    config["adaptive_tess_threshold"] = config.get("adaptive_tess_threshold", None)
    config["subdivision_schedule"]    = config.get("subdivision_schedule", [300, 600])
    config["elevation_schedule"]      = config.get("elevation_schedule", [])
    config["large_steps_lambda"]      = config.get("large_steps_lambda", 3)
    config["image_loss_weight"]       = config.get("image_loss_weight", 1.0)
    config["g1_loss_weight"]          = config.get("g1_loss_weight", 5.0)
    config["lr"]                      = config.get("lr", 0.1)
    config["lr_decay"]                = config.get("lr_decay", 0.8)
    config["num_iterations"]          = config.get("num_iterations", 900)
    config["output_checkpoints"]      = config.get("output_checkpoints", [])
    config["transform"]               = config.get("transform", None)

    print("config:\n", config)

    run_multiview_objects(**config, output_interval=args.output_interval, output_image_indices=args.output_image_indices, no_output=args.no_output, viewer=viewer)