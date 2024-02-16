import math
import torch

from .geometry import create_planar_grid
from .spline_surface import ParametricPatches, Side, CoordinateMergeMethod

def teaser_surface(device: torch.device):
    """ Build the surface from the teaser
    """

    patches = ParametricPatches(3, device=device)

    i = patches.add(torch.tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                                 [[0, 0, 1], [1, 2, 1], [2, 1, 1]],
                                 [[0, 0, 2], [1, 0, 2], [2, 0, 2]]], dtype=torch.float32, device=device))
    j = patches.add(torch.tensor([[[2, 0.0, 0], [3, 0, 0], [4, 0, 0]],
                                 [[2, 1.0, 1], [3, 2, 1], [4, 0, 1]],
                                 [[2, 0.0, 2], [3, 0, 2], [4, 0, 2]]], dtype=torch.float32, device=device))
    k = patches.add(torch.tensor([[[0, 0, 2], [1, 0, 2], [2, 0, 2]],
                                 [[0, 0, 3], [1, -0.5, 3], [2, 0, 3]],
                                 [[0, 0, 4], [1, 0, 4], [2, 0, 4]]], dtype=torch.float32, device=device))
    l = patches.add(torch.tensor([[[2, 0.5, 2], [3, 0, 2], [4, 0, 2]],
                                 [[2, 0.5, 3], [3, 1, 3], [4, 0, 3]],
                                 [[2, 0.5, 4], [3, 0, 4], [4, 0, 4]]], dtype=torch.float32, device=device))

    # Manually connect the patches
    patches.connect(i, j, Side.Right,  Side.Left, merge_method=CoordinateMergeMethod.Average)
    patches.connect(i, k, Side.Bottom, Side.Top,  merge_method=CoordinateMergeMethod.Average)
    patches.connect(j, l, Side.Bottom, Side.Top,  merge_method=CoordinateMergeMethod.Average)
    patches.connect(k, l, Side.Right,  Side.Left, merge_method=CoordinateMergeMethod.Average)

    # Enforce C1 continuity
    patches.V[patches.F[j, :, 1]] = 2*patches.V[patches.F[i, :, 2]] - patches.V[patches.F[i, :, 1]]
    patches.V[patches.F[k, 1, :]] = 2*patches.V[patches.F[i, 2, :]] - patches.V[patches.F[i, 1, :]]
    patches.V[patches.F[l, :, 1]] = 2*patches.V[patches.F[k, :, 2]] - patches.V[patches.F[k, :, 1]]
    patches.V[patches.F[l, 1, :]] = 2*patches.V[patches.F[j, 2, :]] - patches.V[patches.F[j, 1, :]]

    patches.merge_duplicate_vertices()

    return patches

def bezier_sphere(num_patches: int=4, n: int=3, device: torch.device = None):
    """ Build a sphere-like object from Bézier tensor product surfaces
    """

    m        = n
    phi_step = 2*math.pi/num_patches

    patches = ParametricPatches(n, device=device)
    for i in range(num_patches):
        # Generate the polar coordinates for the control points
        phi        = torch.linspace(i*phi_step, (i+1)*phi_step, m)
        theta      = torch.linspace(0, math.pi/2, n)
        theta, phi = torch.meshgrid([theta, phi], indexing='ij')

        # Convert polar coordinates to euclidean coordinates
        patch = torch.stack([torch.sin(theta)*torch.sin(phi), torch.cos(theta), torch.sin(theta)*torch.cos(phi)], dim=-1)
        patches.add(patch)

        # Now also add the bottom part of this patch
        theta = theta + math.pi/2
        patch = torch.stack([torch.sin(theta)*torch.sin(phi), torch.cos(theta), torch.sin(theta)*torch.cos(phi)], dim=-1)
        patches.add(patch)

    patches.merge_duplicate_vertices()

    return patches

def bezier_cube(n: int=3, m: int=None, scale: float=1, device: torch.device=None):
    """ Build a cube from Bézier tensor product surfaces
    """
    
    if m is None:
        m = n

    patches = ParametricPatches(n, device=device)
    patches.add(create_planar_grid(n, m, [0, scale, 0], [0, scale, 0], [scale, 0,  0],    scale=scale, device=device))
    patches.add(create_planar_grid(n, m, [0, 0, scale], [0, 0, scale], [scale, 0,  0],    scale=scale, device=device))
    patches.add(create_planar_grid(n, m, [scale, 0, 0], [scale, 0, 0], [0, 0, -scale],    scale=scale, device=device))
    patches.add(create_planar_grid(n, m, [0, 0, -scale], [0, 0, -scale], [-scale, 0,  0], scale=scale, device=device))
    patches.add(create_planar_grid(n, m, [-scale, 0, 0], [-scale, 0, 0], [0, 0, scale],   scale=scale, device=device))
    patches.add(create_planar_grid(n, m, [0, -scale, 0], [0, -scale, 0], [-scale, 0,  0], scale=scale, device=device))
    patches.merge_duplicate_vertices()

    return patches

def bezier_helix(scale: float=1.0, windings: int=1, height: float=1.0, device: torch.device = None):
    P = torch.tensor([
        [[-1, 0.00,  0], [-1, 0.25,  1], [ 0,  0.5,  1]],
        [[ 0, 0.50,  1], [ 1, 0.75,  1], [ 1,  1.0,  0]],
        [[ 1, 1.00,  0], [ 1, 1.25, -1], [ 0,  1.5, -1]],
        [[ 0, 1.50, -1], [-1, 1.75, -1], [-1,  2.0,  0]],
    ], dtype=torch.float32, device=device)

    # Repeat the curves for the desired number of times
    P = P.repeat(windings, 1, 1)

    for i in range(1, windings):
        P[4*i:4*(i+1), :, 1] += i*2

    P[:, :, 1] *= height

    return scale*P

def triangle_profile(scale: float, device: torch.device = None):
    return scale * torch.tensor([[1, 0.0], [-1, 1], [-1, -1]], dtype=torch.float32, device=device)

def square_profile(scale: float, device: torch.device = None):
    return scale * torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float32, device=device)

def circle_profile(radius: float, n: int=16, device: torch.device = None):
    phi = torch.linspace(0, 2*math.pi, n, device=device)
    return radius * torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)

def heart_profile(radius: float, n: int=16, device: torch.device = None):
    u = torch.linspace(0, 2*math.pi, n, device=device)
    x = 16*torch.sin(u)**3
    y = 13*torch.cos(u) - 5*torch.cos(2*u) - 2*torch.cos(3*u) - torch.cos(4*u)
    return radius*torch.stack([y, x], dim=-1)