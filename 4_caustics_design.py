from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import copy
import drjit
import imageio
import math
import matplotlib.pyplot as plt
import mitsuba as mi
from mitsuba.scalar_rgb import Transform4f as T
import numpy as np
from pathlib import Path
from umbra import MeshViewer
import torch
from tqdm import tqdm
from typing import List

from NURBSDiff.surf_eval import SurfEval
from drpg import generate_grid_faces, save_mesh, compute_lsig_matrix, create_planar_grid, write_quad_mesh, ParametricPatches, SurfaceTessellator, SurfaceType

# This code is heavily inspired by the Mitsuba 3 caustics optimization tutorial:
# (https://github.com/mitsuba-renderer/mitsuba-tutorials/blob/master/inverse_rendering/caustics_optimization.ipynb)

def load_ref_image(path, resolution, output_dir):
    b = mi.Bitmap(str(path))
    b = b.convert(mi.Bitmap.PixelFormat.RGB, mi.Bitmap.Float32, False)
    if b.size() != resolution:
        b = b.resample(resolution)

    mi.util.write_bitmap(str(output_dir / 'out_ref.exr'), b)
    
    print('[i] Loaded reference image from:', path)
    return mi.TensorXf(b).torch()

def create_emitter(emitter_type):
    if emitter_type == 'gray':
        return {
            'type':'directionalarea',
            'radiance': {
                'type': 'spectrum', 
                'value': 0.8
            },
        }
    elif emitter_type == 'bayer':
        bayer = drjit.zeros(mi.TensorXf, (32, 32, 3))
        bayer[ ::2,  ::2, 2] = 2.2
        bayer[ ::2, 1::2, 1] = 2.2
        bayer[1::2, 1::2, 0] = 2.2

        return {
            'type':'directionalarea',
            'radiance': {
                'type': 'bitmap',
                'bitmap': mi.Bitmap(bayer),
                'raw': True,
                'filter_type': 'nearest'
            },
        }

def create_integrator():
    return {
        'type': 'ptracer',
        'samples_per_pass': 256,
        'max_depth': 4,
        'hide_emitters': False,
    }

def create_sensor(resolution):
    # Looking at the receiving plane, not looking through the lens
    sensor_to_world = mi.ScalarTransform4f.look_at(
        target=[0, -20, 0],
        origin=[0, -4.65, 0],
        up=[0, 0, 1]
    )

    return {
        'type': 'perspective',
        'near_clip': 1,
        'far_clip': 1000,
        'fov': 45,
        'to_world': sensor_to_world,

        'sampler': {
            'type': 'independent',
            'sample_count': 512  # Not really used
        },
        'film': {
            'type': 'hdrfilm',
            'width': resolution[0],
            'height': resolution[1],
            'pixel_format': 'rgb',
            'rfilter': {
                # Important: smooth reconstruction filter with a footprint larger than 1 pixel.
                'type': 'gaussian'
            }
        },
    }

def create_scene(integrator, sensor, emitter, lens_filename):
    return {
        'type': 'scene',
        'sensor': sensor,
        'integrator': integrator,
        # Glass BSDF
        'simple-glass': {
            'type': 'dielectric',
            'id': 'simple-glass-bsdf',
            'ext_ior': 'air',
            'int_ior': 1.5,
            'specular_reflectance': { 'type': 'spectrum', 'value': 0 },
        },
        'white-bsdf': {
            'type': 'diffuse',
            'id': 'white-bsdf',
            'reflectance': { 'type': 'rgb', 'value': (1, 1, 1) },
        },
        'black-bsdf': {
            'type': 'diffuse',
            'id': 'black-bsdf',
            'reflectance': { 'type': 'spectrum', 'value': 0 },
        },
        # Receiving plane
        'receiving-plane': {
            'type': 'obj',
            'id': 'receiving-plane',
            'filename': 'rectangle.obj',
            'to_world': \
                mi.ScalarTransform4f.look_at(
                    target=[0, 1, 0],
                    origin=[0, -7, 0],
                    up=[0, 0, 1]
                ).scale((5, 5, 5)),
            'bsdf': {'type': 'ref', 'id': 'white-bsdf'},
        },
        # Glass slab, excluding the 'exit' face (added separately below)
        'slab': {
            'type': 'obj',
            'id': 'slab',
            'filename': 'slab.obj',
            'to_world': mi.ScalarTransform4f.rotate(axis=(1, 0, 0), angle=90),
            'bsdf': {'type': 'ref', 'id': 'simple-glass'},
        },

        # Glass rectangle, to be optimized
        'lens': {
            'type': 'obj' if 'obj' in lens_filename else 'ply',
            'filename': lens_filename,
            'id': 'lens',
            'bsdf': {'type': 'ref', 'id': 'simple-glass'},
        },

        # Directional area emitter placed behind the glass slab
        'focused-emitter-shape': {
            'type': 'obj',
            'filename': 'rectangle.obj',
            'to_world': mi.ScalarTransform4f.look_at(
                target=[0, 0, 0],
                origin=[0, 5, 0],
                up=[0, 0, 1]
            ),
            'bsdf': {'type': 'ref', 'id': 'black-bsdf'},
            'focused-emitter': emitter,
        },
    }

def scale_independent_loss(img_opt, img_ref):
    """Brightness-independent L2 loss function."""
    img_opt = img_opt / torch.mean(img_opt.detach())
    img_ref = img_ref / torch.mean(img_ref)
    return torch.mean((img_opt - img_ref)**2)

def run_caustics_design(image_path: Path, output_dir: Path, num_iterations=500, degree=4, n=20, m=20, render_resolution=(128, 128), uv_resolution=(128, 128), emitter_type='gray', refinement_iterations: List[int] = [0, 100, 400, 2000], large_steps_lambdas: List[float] = [5., 5., 0., 0.], step_sizes: List[float] = [0.1, 0.01, 0.001, 0.0005],  output_interval: int = 50, save_intermediate_lenses: bool = False, display_plot: bool = False, viewer: MeshViewer=None, data_dir: Path = Path("./data")):
    
    device = torch.device('cuda:0')

    # Create a NURBS control mesh for the side of the lens that is optimized
    patches = ParametricPatches(n=n, m=m, device=device)
    patches.add(create_planar_grid(n=n, m=m, position=[0, 0, 0], normal=[0, -1, 0], tangent=[1, 0, 0], scale=1, device=device))

    # Create the initial tessellation
    tessellator = SurfaceTessellator(uv_resolution, SurfaceType.NURBS)
    with torch.no_grad():
        v, f, _, _ = tessellator.tessellate(patches, degree=degree)
    f = f.to(device, dtype=torch.int32)

    if viewer:
       viewer.set_points(patches.V.reshape(-1, 3).cpu().numpy(), c=[0, 0, 1], object_name="control_points")
       viewer.set_mesh(v.cpu().numpy(), f=f.cpu().numpy(), c=[0, 1, 0], object_name="surface_points")

    # Make the data and output directory visible to the Mitsuba file resolver
    mi.set_variant('cuda_ad_rgb')
    mi.Thread.thread().file_resolver().append(str(data_dir.absolute()))
    mi.Thread.thread().file_resolver().append(str(output_dir.absolute()))

    # Assemble the Mitsuba scene
    emitter    = create_emitter(emitter_type)
    integrator = create_integrator()
    sensor     = create_sensor(render_resolution)

    # Passing the initial lens as file to Mitsuba
    # allows setting custom material parameters for the mesh
    # TODO: Is this detour still necessary?
    lens_filename = "lens.obj"
    save_mesh(output_dir / lens_filename, v.cpu().numpy(), f=f.cpu().numpy())
    
    scene = create_scene(integrator, sensor, emitter, lens_filename)
    scene = mi.load_dict(scene)

    # Define the function for rendering the caustics
    params = mi.traverse(scene)
    @drjit.wrap_ad(source='torch', target='drjit')
    def render_caustics(vertices, spp=256, seed=1):
        params["lens.vertex_positions"] = mi.Float(drjit.ravel(vertices))
        params["lens.faces"] = mi.UInt(mi.Int(f.ravel()))
        params.update()
        return mi.render(scene, params, seed=seed, spp=2*spp, spp_grad=spp)

    # Render the initial state and save the slab and lens
    spp = 32
    img = render_caustics(v, spp=spp, seed=0)
    imageio.imwrite(output_dir / "initial.png", mi.util.convert_to_bitmap(img))

    lens_mesh = [m for m in scene.shapes() if m.id() == 'lens'][0]
    lens_mesh.write_ply(str(output_dir / "lens.ply"))

    slab_mesh = [m for m in scene.shapes() if m.id() == 'slab'][0]
    slab_mesh.write_ply(str(output_dir / "slab.ply"))

    # Load the reference image and resize it to match the sensor
    sensor    = scene.sensors()[0]
    crop_size = sensor.film().crop_size()
    img_ref   = load_ref_image(image_path, crop_size, output_dir=output_dir)
    imageio.imwrite(output_dir/"image_reference.png", img_ref.cpu())

    # Create output directories 
    lenses_output_dir = output_dir / "lenses"
    render_output_dir = output_dir / "renderings"
    lenses_output_dir.mkdir(parents=True, exist_ok=True)
    render_output_dir.mkdir(parents=True, exist_ok=True)

    # Optimize only the y-coordinate of the control points
    y_opt = torch.zeros_like(patches.V[..., 0])

    use_large_steps = True
    factor          = n/40
    refinements = {it: (lambd, factor*step_size) for (it, lambd, step_size) in zip(refinement_iterations, large_steps_lambdas, step_sizes)}
    def setup_optimizer(y_opt, lr, lambda_=1):
        y_opt = y_opt.detach()
        if use_large_steps:
            # Optimize vertices with gradient preconditioning
            from largesteps.parameterize import to_differential, from_differential
            from largesteps.optimize import AdamUniform

            # Compute the system matrix
            F = generate_grid_faces(n, m, quads=True).to(device, dtype=torch.long)
            M = compute_lsig_matrix(y_opt.reshape(-1, 1), F, lambda_=lambda_)

            # Parameterize
            u_opt = to_differential(M, y_opt.reshape(-1, 1))
            u_opt.requires_grad = True
            optimizer = AdamUniform([u_opt], lr=lr)
        else:
            y_opt.requires_grad = True
            optimizer = torch.optim.Adam([y_opt], lr=lr)

        def get_variables():
            if use_large_steps:
                from largesteps.parameterize import from_differential
                return from_differential(M, u_opt, 'Cholesky').reshape_as(patches.V[..., 0])
            return y_opt
            
        return optimizer, get_variables

    # NOTE: For the default parameters, this setup is a no-op because 
    #       iteration 0 is marked as refinement iteration and triggers
    #       a setup inside of the optimization loop
    optimizer, get_variables = setup_optimizer(y_opt, lr=0.1)

    losses = []
    progress_range = tqdm(range(num_iterations))
    for it in progress_range:
        if it in refinements:
            optimizer, get_variables = setup_optimizer(y_opt, lr=refinements[it][1], lambda_=refinements[it][0])

        y_opt = get_variables()

        patches_opt           = copy.deepcopy(patches)
        patches_opt.V[..., 1] = y_opt.clamp(max=0.05)

        # Pin the border
        patches_opt.V[patches_opt.F[:, :,  0], 1] = 0
        patches_opt.V[patches_opt.F[:, :, -1], 1] = 0
        patches_opt.V[patches_opt.F[:,  0, :], 1] = 0
        patches_opt.V[patches_opt.F[:, -1, :], 1] = 0

        # Tessellate the lens and render the caustics image
        v_opt, _, _, _ = tessellator.tessellate(patches_opt, degree=degree)
        img_opt        = render_caustics(v_opt, spp=spp, seed=it)

        loss = scale_independent_loss(img_opt, img_ref)
        
        if it % output_interval == 0:
            imageio.imwrite(output_dir / f"optimized_{it}.png", mi.util.convert_to_bitmap(img_opt))

            frame_index = it // output_interval
            imageio.imwrite(render_output_dir / f"{frame_index}.png", mi.util.convert_to_bitmap(img_opt))
            if save_intermediate_lenses:
                frame_output_dir = lenses_output_dir / str(frame_index)
                frame_output_dir.mkdir(parents=True, exist_ok=True)
                P_opt = patches_opt.V[patches_opt.F]
                F = generate_grid_faces(P_opt.shape[1], P_opt.shape[2], quads=True)
                write_quad_mesh(frame_output_dir / f"controlmesh_0.obj", P_opt.reshape(-1, 3).cpu(), F)
                save_mesh(frame_output_dir / f"surface_0.obj", v_opt, f)

        if viewer:
            P_opt = patches_opt.V[patches_opt.F]
            viewer.set_mesh(v_opt.detach().cpu().numpy(), f=f.detach().cpu().numpy(), c=[0, 1, 0], object_name="surface_points")
            viewer.set_points(P_opt.reshape(-1, 3).detach().cpu().numpy(), c=[0, 0, 1], object_name="control_points")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += [ float(loss.item()) ]
        progress_range.set_postfix({"loss": f"{losses[-1]:0.6f}"})

    # Save the control points of the lens
    np.save(output_dir / "control_points.npy", {
        'P': patches_opt.V[patches_opt.F].detach().cpu().numpy(),
        'degree': degree,
    })

    # Save the tessellated lens and the final rendered caustics
    save_mesh(output_dir / "optimized.obj", v_opt, f)
    imageio.imwrite(output_dir / "optimized.png", mi.util.convert_to_bitmap(img_opt))

    plt.plot(losses)
    if display_plot:
        plt.show()
    else:
        plt.savefig(output_dir / "loss.pdf")

    return

def render_caustics(lens_path, render_resolution=(256, 256), emitter_type='gray', viewer: MeshViewer=None):
    lens_path = Path(lens_path)

    mi.set_variant('cuda_ad_rgb')
    mi.Thread.thread().file_resolver().append("./data")
    mi.Thread.thread().file_resolver().append(str(lens_path.parent))

    emitter    = create_emitter(emitter_type)
    integrator = create_integrator()
    sensor     = create_sensor(render_resolution)
    scene      = create_scene(integrator, sensor, emitter, lens_path.name)

    scene = mi.load_dict(scene)

    spp = 128
    img = mi.render(scene, spp=spp)

    imageio.imwrite(lens_path.parent / "rendering.png", mi.util.convert_to_bitmap(img))


if __name__ == "__main__":
    parser = ArgumentParser("Lens Design with Caustics", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("image_path", type=Path, help="Path to the target image.")
    parser.add_argument("output_dir", type=Path, help="Path to the output directory.")
    parser.add_argument("--degree", type=int, default=4, help="Degree of the NURBS patch.")
    parser.add_argument("--resolution_nm", type=int, default=[60, 60], nargs='+', help="Resolution of the control mesh.")
    parser.add_argument("--resolution_uv", type=int, default=[128, 128], nargs='+', help="Resolution of the tessellation.")
    parser.add_argument("--resolution_render", type=int, default=[128, 128], nargs='+', help="Resolution of the rendered images.")
    parser.add_argument("--viewer", default=False, action="store_true", help="Show an interactive viewer")
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of optimization iterations.")
    parser.add_argument("--display_plot", default=False, action="store_true", help="Display the loss plot instead of saving it.")
    parser.add_argument("--refinement_iterations", type=int, default=[0, 100, 400, 2000], nargs='+', help="Iterations at which to perform a refinement of LS lambda and LR.")
    parser.add_argument("--large_steps_lambdas", type=float, default=[5., 5., 0., 0.], nargs='+', help="Large steps lambdas for the refinement iterations")
    parser.add_argument("--step_sizes", type=float, default=[0.1, 0.01, 0.001, 0.0005], nargs='+', help="Gradient descent step sizes for the refinement iterations")
    parser.add_argument("--output_interval", type=int, default=50, help="Interval for outputs.")
    parser.add_argument("--save_intermediate_lenses", default=False, action="store_true", help="Save intermediate lenses as OBJ files.")
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    n, m = args.resolution_nm
    u, v = args.resolution_uv

    viewer = MeshViewer() if args.viewer else None

    run_caustics_design(image_path=args.image_path, output_dir=output_dir, degree=args.degree, n=n, m=m, uv_resolution=args.resolution_uv, num_iterations=args.num_iterations, viewer=viewer, render_resolution=args.resolution_render, 
                        refinement_iterations=args.refinement_iterations, large_steps_lambdas=args.large_steps_lambdas, step_sizes=args.step_sizes, 
                        output_interval=args.output_interval, display_plot=args.display_plot, save_intermediate_lenses=args.save_intermediate_lenses)
    
    render_caustics(output_dir / "optimized.obj", render_resolution=(512, 512))