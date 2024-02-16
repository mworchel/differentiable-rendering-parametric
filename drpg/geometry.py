import torch

def dot(v: torch.Tensor, w: torch.Tensor, dim: int=-1, keepdim: bool=True, **kwargs):
    return torch.sum(v*w, dim=dim, keepdim=keepdim, **kwargs)

def compute_frames_double_reflection(x: torch.Tensor, v: torch.Tensor, a: torch.Tensor):
    """ Compute rotation minimizing frames along a curve with the Double Reflection algorithm [1]

    [1] Wang et al. (2008) - Computation of rotation minimizing frames

    Args:
        x: Positions along a space curve      [(Bx)Nx3]
        v: Velocity at discrete positions     [(Bx)Nx3]
        a: Acceleration at discrete positions [(Bx)Nx3]
    """
    is_batched = len(x.shape) == 3
    if not is_batched:
        x = x.unsqueeze(0)
        v = v.unsqueeze(0)
        a = a.unsqueeze(0)

    B, N = x.shape[0:2]

    t = torch.nn.functional.normalize(v, p=2, dim=-1)

    # TODO: Handle collinear case
    b = torch.cross(a, v, dim=-1)
    n = torch.cross(v, b, dim=-1)

    t = torch.nn.functional.normalize(v, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    n = torch.nn.functional.normalize(n, dim=-1)

    # Align frames with double reflection
    with torch.no_grad():
        # Set the alignment for the first frame
        a = torch.zeros((B, N, 2), dtype=torch.float32, device=x.device)
        a[:, 0, 0] = 0
        a[:, 0, 1] = 1

        for i in range(0, N-1):
            bi  = a[:, i, 0:1]*n[:, i] + a[:, i, 1:2]*b[:, i]
            ti  = t[:, i]
            tii = t[:, i+1]

            v1 = x[:, i+1] - x[:, i]
            c1 = dot(v1, v1, keepdim=True)
            if (c1 == 0).any():
                # If two points share the same position (connected curves),
                # they should share the frame
                print("WARNING: Same points detected")
                bi = n[:, i]*a[:, i, 0] + b[:, i]*a[:, i, 1]
                a[:, i+1, 0] = dot(n[:, i+1], bi, keepdim=True)
                a[:, i+1, 1] = dot(b[:, i+1], bi, keepdim=True)
                continue
            bL = bi - (2/c1)*dot(v1, bi, keepdim=True)*v1
            tL = ti - (2/c1)*dot(v1, ti, keepdim=True)*v1
            v2 = tii - tL
            c2 = dot(v2, v2, keepdim=True)
            if (c2 == 0).any():
                raise RuntimeError("c2...")
            bii = bL - (2/c2)*dot(v2, bL, keepdim=True)*v2
            nii = torch.cross(tii, bii, dim=-1)
            a[:, i+1, 0:1] = dot(n[:, i+1], bii, keepdim=True)
            a[:, i+1, 1:2] = dot(b[:, i+1], bii, keepdim=True)

    b = torch.nn.functional.normalize(a[:, :, 0:1] * n + a[:, :, 1:2] * b, dim=-1, p=2)
    n = torch.nn.functional.normalize(torch.cross(t, b, dim=-1), dim=-1, p=2)

    if not is_batched:
        t = t.squeeze(0)
        n = n.squeeze(0)
        b = b.squeeze(0)

    return t, n, b

def recompute_frames(x: torch.Tensor, v: torch.Tensor, a: torch.Tensor, 
                     t_prev: torch.Tensor, n_prev: torch.Tensor, b_prev: torch.Tensor):
    """ Compute new frames from the current curve velocity and previous frames
        According to Section 3.5 in the paper.
    
    Args:
        x: Positions along a space curve      [(Bx)Nx3]
        v: Velocity at discrete positions     [(Bx)Nx3]
        a: Acceleration at discrete positions [(Bx)Nx3]
        t_prev: Previous frame tangents       [(Bx)Nx3]
        n_prev: Previous frame normals        [(Bx)Nx3]
        b_prev: Previous frame binormals      [(Bx)Nx3]

    Returns:
        New frames (tangent, normal, binormal).
    """

    # v ~ t
    # a ~ n
    n = torch.cross(v, b_prev.detach(), dim=-1)
    b = torch.cross(n, v, dim=-1)

    t = torch.nn.functional.normalize(v, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    n = torch.nn.functional.normalize(n, dim=-1)

    return t, n, b

def generate_grid_faces(n: int, m: int, quads: bool=False):
    """ Generate an indexed face set for vertices arranged on a regular nxm grid

    Args:
        n: Grid rows
        m: Grid columns
        quads: If true, return quad faces, otherwise triangle faces (default: False)

    Returns:
        Indices with shape [Fx3] (triangles) or [Fx4] (quads).
    """

    index_n, index_m = torch.meshgrid([
        torch.arange(0, n-1)*m,
        torch.arange(0, m-1),
    ], indexing='ij')

    if not quads:
        f_quad = torch.tensor([
            [1, 0, m], [1, m, m+1]
        ])
    else:
        f_quad = torch.tensor([
            [1, 0, m, m+1]
        ])
    
    f = ((index_n + index_m)[:, :, None, None] + f_quad[None, None, :, :])

    if not quads:
        f = f.reshape(-1, 3)
    else:
        f = f.reshape(-1, 4)

    return f

def transform_affine(points, position, normal, tangent, scale):
    device = points.device

    position = position if torch.is_tensor(position) else torch.tensor(position, dtype=torch.float32, device=device)

    normal = normal if torch.is_tensor(normal) else torch.tensor(normal, dtype=torch.float32, device=device)
    normal = torch.nn.functional.normalize(normal, p=2, dim=0)

    tangent = tangent if torch.is_tensor(tangent) else torch.tensor(tangent, dtype=torch.float32, device=device)
    tangent = torch.nn.functional.normalize(tangent, p=2, dim=0)

    bitangent = torch.cross(tangent, normal, dim=-1)
    bitangent = torch.nn.functional.normalize(bitangent, p=2, dim=0)

    R = torch.stack([tangent, normal, bitangent], dim=1)

    return (scale*points @ R.T) + position[None, :]

def create_xz_grid(n: int, m: int, device: torch.device):
    """ Generate vertices of a regular nxm grid in the xz plane (extent [-1, 1]^2)

                 ___m___(1, 1)
                |  x    |
              n |  |__ z|
                |_______|
        (-1, -1)

    Args:
        n: Number of vertices in the z direction
        m: Number of vertices in the x direction
        device: Device 
         
    Returns:
        Two coordinate arrays (x,z), each with shape [n,m].
        x is the faster varying coordinate.
    """

    x = torch.linspace(-1, 1, m, device=device)
    z = torch.linspace(-1, 1, n, device=device)
    x, z = torch.meshgrid([x, z], indexing="xy")
    return x, z

def create_planar_grid(n: int, m: int, position: torch.Tensor, normal: torch.Tensor, tangent: torch.Tensor, scale: float, device: torch.device):
    """ Generate vertices of a flat nxm grid. The y coordinate is computed as f(x, z) = 0.
     
        The grid is laid out in the xz-plane [-scale, scale]^2 and transformed such that it is centered at `position`, 
        its y-axis aligns with `normal` and its x-axis aligns with `tangent`.
    """

    x, z = create_xz_grid(n, m, device)
    y    = torch.zeros_like(x)
    return transform_affine(torch.stack([x, y, z], dim=-1), position, normal, tangent, scale)

def create_quadratic_grid(n: int, m: int, a: float, b: float, position: torch.Tensor, normal: torch.Tensor, tangent: torch.Tensor, scale: float, device: torch.device):
    """ Generate vertices of a flat nxm grid. The y coordinate is computed as f(x, z) = -a*x^2 - b*z^2 + a + b.
     
        The grid is laid out in the xz-plane [-scale, scale]^2 and transformed such that it is centered at `position`, 
        its y-axis aligns with `normal` and its x-axis aligns with `tangent`.
    """

    x, z = create_xz_grid(n, m, device)
    y    = -a*x**2 - b*z**2 + (a+b)
    return transform_affine(torch.stack([x, y, z], dim=-1), position, normal, tangent, scale)