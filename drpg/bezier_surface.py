from typing import Tuple
import torch

from .bezier_curve import evaluate_bezier_curve, elevate_bezier_curve_degree, subdivide_bezier_curve

def evaluate_bezier_surface_on_grid(P: torch.Tensor, u: torch.Tensor, v: torch.Tensor, with_tangents=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Evaluate a bezier surface patch on a grid in the parameter domain.

    Args:
        P: Control points of the patch [(Bx)NxMx3]
        u: Parameter values            [(Bx)U]
        v: Parameter values            [(Bx)V]
        with_tangents: Indicator if the tangents are computed

    Returns:
        p: Evaluated points         [(Bx)VxUx3]
        tu: Tangents in u direction [(Bx)VxUx3]
        tv: Tangents in v direction [(Bx)VxUx3]
    """
    is_batched_P = len(P.shape) == 4
    if not is_batched_P:
        P = P.unsqueeze(0)

    B, N, M, C = P.shape
    U = u.shape[-1]
    V = v.shape[-1]

    is_batched_u = len(u.shape) == 2
    is_batched_v = len(v.shape) == 2

    if is_batched_u and u.shape[0] != B:
        raise RuntimeError(f"Batch size must match for control points ({B}) and u parameter values ({u.shape[0]})")

    if is_batched_v and v.shape[0] != B:
        raise RuntimeError(f"Batch size must match for control points ({B}) and v parameter values ({v.shape[0]})")

    u_ = torch.repeat_interleave(u, N, dim=0) if is_batched_u else u
    v_ = torch.repeat_interleave(v, U, dim=0) if is_batched_v else v

    # p_u [(B*N)xUx3]
    p_u, t_u, _ = evaluate_bezier_curve(P.reshape(B*N, M, C), u_)
    
    # (B*N, U, C) -> (B, N, U, C) -> (B, U, N, C) -> (B*U, N, C)
    p_u          = p_u.reshape(B, N, U, -1)
    p_uv, t_v, _ = evaluate_bezier_curve(p_u.transpose(1, 2).reshape(B*U, N, -1), v_)
    # (B*U, V, C) -> (B, U, V, C) -> (B, V, U, C)
    p_uv = p_uv.reshape(B, U, V, -1).transpose(1, 2)

    if with_tangents:
        t_v = t_v.reshape(B, U, V, -1).transpose(1, 2)

        # (B*N, U, C) -> (B, N, U, C) -> (B, U, N, C) -> (B*U, N, C)
        t_u       = t_u.reshape(B, N, U, -1)
        t_u, _, _ = evaluate_bezier_curve(t_u.transpose(1, 2).reshape(B*U, N, -1), v_)
        t_u       = t_u.reshape(B, U, V, -1).transpose(1, 2)

        if not is_batched_P:
            t_u = t_u.squeeze(0)
            t_v = t_v.squeeze(0)
    else:
        t_u = None
        t_v = None

    if not is_batched_P:
        p_uv = p_uv.squeeze(0)

    return p_uv, t_u, t_v

def evaluate_bezier_surface(P: torch.tensor, u: torch.tensor, v: torch.tensor):
    """ Evaluate a parametric surface at certain parameter values

    Args:
        P: Control points of the surface [NxMxC]
        u: Parameter values              [U]
        v: Parameter values              [U]

    Returns:
        P_uv: Evaluated points                               [Ux3]
        t_u: Tangents in u direction at the evaluated points [Ux3]
        t_v: Tangents in v direction at the evaluated points [Ux3]
    """
    P_u,  t_u, _ = evaluate_bezier_curve(P, u)                            # P_u.shape  = NxUx3
    P_uv, t_v, _ = evaluate_bezier_curve(P_u.transpose(0, 1), v[:, None]) # P_uv.shape = Ux1x3
    t_u, _, _    = evaluate_bezier_curve(t_u.transpose(0, 1), v[:, None]) # t_u.shape  = t_v.shape = Ux1x3

    return P_uv[:, 0, :], t_u[:, 0], t_v[:, 0]

def elevate_bezier_surface_degree(P: torch.tensor):
    """ Elevate the degree of a Bezier tensor product surface

    Args:
        P: Control points of the surface [(Bx)NxMxC]

    Returns:
        Q: New control points of the surface [(Bx)N+1xM+1xC]
    """

    is_batched = len(P.shape) == 4
    if not is_batched:
        P = P.unsqueeze(0)

    B, N, M, C = P.shape

    Q = elevate_bezier_curve_degree(P.reshape(B*N, M, C)).reshape(B, N, M+1, C)
    Q = elevate_bezier_curve_degree(Q.permute(0, 2, 1, 3).reshape(B*(M+1), N, C)).reshape(B, M+1, N+1, C)
    Q = Q.permute(0, 2, 1, 3)

    if not is_batched:
        Q = Q.squeeze(0)

    return Q

def subdivide_bezier_surface(P: torch.Tensor, flat: bool=True) -> torch.Tensor:
    """ Subdivide a Bezier tensor product surface

    Args:
        P: Control points of the surface [(Bx)NxMxC]
        flat: Indicator whether to flatten the batch dimension of the result tensor

    Returns:
        Q: New control points of the surface [(B*4x)NxMxC] if `flat = True` else 
           [Bx2x2xNxMxC] with the follwing order for dimensions 1 and 2:
           [00|01]
           [10|11]
    """

    is_batched = len(P.shape) == 4
    if not is_batched:
        P = P.unsqueeze(0)

    B, N, M, C = P.shape

    Q = subdivide_bezier_curve(P.reshape(B*N, M, C)).reshape(B, N, 2, M, C)
    Q = subdivide_bezier_curve(Q.permute(0, 2, 3, 1, 4).reshape(B*2*M, N, C)).reshape(B, 2, M, 2, N, C)
    Q = Q.permute(0, 3, 1, 4, 2, 5)

    if not is_batched:
        Q = Q.squeeze(0)

    if flat:
        Q = Q.reshape(B*4, N, M, C)

    return Q

def get_normals_from_tangents(tu, tv):
    n = torch.cross(tv, tu, dim=-1)
    return torch.nn.functional.normalize(n, dim=-1)