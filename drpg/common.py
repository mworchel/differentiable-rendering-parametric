import imageio
import json
from pathlib import Path
import torch

def make_grid(sizes, limits=(-1, 1)):
    # Check if limits are intended for all dimensions
    if len(limits) == 2 and not hasattr(limits[0], '__len__'):
        limits = [limits]*len(sizes)

    # Flip the y-axis for images
    if len(sizes) == 2:
        limits[1] = (limits[1][1], limits[1][0])

    xs = []
    for size_x, limits_x in zip(sizes, limits):
        xs += [ torch.linspace(*limits_x, size_x) ]

    return torch.stack(torch.meshgrid(*xs[::-1], indexing='ij')[::-1], dim=-1)

def load_mesh(path, device, normalize=True):
    import trimesh

    mesh: trimesh.Trimesh = trimesh.load_mesh(path)

    v = torch.from_numpy(mesh.vertices).to(device, dtype=torch.float32)
    f = torch.from_numpy(mesh.faces).to(device, dtype=torch.int32)

    if normalize:
        v -= torch.mean(v, dim=0)
        v /= torch.max(torch.linalg.norm(v, dim=-1))

    vertex_normals = torch.from_numpy(mesh.vertex_normals).to(device, dtype=torch.float32)

    return v, f, vertex_normals

def save_mesh(path, v, f, **kwargs):
    """ Save a triangle mesh to a file

    Args:
        path: Path to the *.obj file
        v: Vertices                         [Vx3]
        f: Index set that defines the faces [Fx3]
        **kwargs: Keyword arguments passed to trimesh
    """
    import trimesh

    if torch.is_tensor(v):
        v = v.detach().cpu().numpy()

    if torch.is_tensor(f):
        f = f.detach().cpu().numpy()

    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False, **kwargs)
    mesh.export(path)

def write_quad_mesh(path, v, f):
    """ Save a quad mesh to a file (only recommended for small meshes)

    Args:
        path: Path to the *.obj file
        v: Vertices                         [Vx3]
        f: Index set that defines the faces [Fx4]
    """

    with open(path, "w") as file:
        for i in range(len(v)):
            file.write(f"v {v[i,0]} {v[i,1]} {v[i,2]}\n")
            
        for i in range(len(f)):
            file.write(f"f {f[i,0]+1} {f[i,1]+1} {f[i,2]+1} {f[i,3]+1}\n")

def save_polyline(path, points):
    """ Save a polyline to a numpy txt file.

    Args:
        path: Path to the *.txt file
        points: Points on the polyline [Vx3]
    """
    import numpy as np
    np.savetxt(path, points)

def load_environment_map(path, device):
    path = Path(path)
    format, scale = ('HDR-FI', 1.0) if path.suffix == '.hdr' else (None, 255.0)
    envmap        = torch.tensor(imageio.imread(path, format=format)/scale, dtype=torch.float32, device=device)
    if (envmap.shape[2]) == 4:
        envmap = envmap[:, :, :3]
    # alpha         = torch.ones_like(envmap[..., :1])
    # torch.cat([envmap, alpha], dim=-1)
    return envmap.contiguous()

def load_cubemap(path, device):
    path = Path(path)
    format, scale = (None, 1.0) if path.suffix == '.exr' else (None, 255.0)
    cubemap       = torch.tensor(imageio.imread(path, format=format)/scale, dtype=torch.float32, device=device)
    if (cubemap.shape[2]) == 4:
        cubemap = cubemap[:, :, :3]
    return torch.stack(torch.chunk(cubemap, 6, dim=1), dim=0).contiguous()

def load_bpt(filepath: Path, transpose: bool=False, verbose: bool=False, device: torch.device='cpu') -> torch.Tensor:
    """ Load a *.bpt file, storing a collection of Bézier patches

    Args:
        filepath: Path to the input *.bpt file
        transpose: Whether to transpose the control grid of each patch (e.g. required for the teapot)
        verbose: Indicator for verbose output about read file
        device: Device hosting the returned tensor

    Returns:
        A collection of Bézier patches defined by their control points with shape (B, n, m, 3),
        where n is the number of rows and m the number of columns of the control grid and B
        is the total number of patches.
    """

    patches = []
    with open(filepath, 'r') as f:    
        num_patches = int(f.readline())
        if verbose:
            print(f"{num_patches=}")

        for i in range(num_patches):
            n, m = [int(v) for v in f.readline().split(' ')]

            if verbose and i == 0:
                print(f"{n=} {m=}")

            P = []
            for j in range((n+1)*(m+1)):
                P += [ [float(v) for v in  f.readline().split(' ')] ]

            patches += [ P ]
    
    patches = torch.tensor(patches, dtype=torch.float32, device=device).reshape(num_patches, n+1, m+1, -1)

    if transpose:
            patches.transpose_(1, 2)
        
    return patches

def save_bpt(filepath: Path, patches: torch.Tensor, transpose: bool=False):
    """ Save a *.bpt file, storing a collection of Bézier patches

    Args:
        filepath: Path to the output *.bpt file
        patches: collection of Bézier patches defined by their control points with shape (B, n, m, 3)
        transpose: Whether to transpose the control grid of each patch (e.g. required for the teapot)
    """

    num_patches = patches.shape[0]

    if num_patches == 0:
        raise RuntimeError("The tensor of patches is empty.")

    # Download the patches to the CPU and transpose the grids if necessary
    patches = patches.to(device='cpu')
    if transpose:
        patches.transpose_(1, 2)

    n = patches.shape[1] - 1
    m = patches.shape[2] - 1

    with open(filepath, 'w') as f:
        f.write(f"{num_patches}\n")
        
        for patch_id in range(num_patches):
            # Store the degree
            f.write(f"{n} {m}\n") 

            for i in range(n+1):
                for j in range(m+1):
                    x, y, z = patches[patch_id, i, j]
                    f.write(f"{x} {y} {z}\n")

def load_json(path):
    with open(path, 'r') as file:
        return json.load(file)
    
def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
def convert_quad_mesh_to_patches(vertices: torch.Tensor, faces: torch.Tensor, n: int, m: int) -> torch.Tensor:
    u = 1-torch.linspace(0, 1, m, device=vertices.device)
    v = 1-torch.linspace(0, 1, n, device=vertices.device)

    p0 = torch.multiply(*torch.meshgrid([v, u], indexing='ij')[::-1])
    p1 = p0.flip(0)
    p2 = p1.flip(1)
    p3 = p2.flip(0)

    weights = torch.stack([p0, p1, p2, p3], dim=0)
    
    P = torch.matmul(vertices[faces.to(dtype=torch.long)].transpose(1, 2).reshape(-1, 4), weights.reshape(4, -1)).reshape(-1, 3, n, m).permute(0, 2, 3, 1)

    return P