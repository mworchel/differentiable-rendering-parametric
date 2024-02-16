import torch

from .casteljau import decasteljau_forward, decasteljau_backward

class BezierCurveEvaluation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, control_points, t):
        """
        Args:
            control_points: Control point positions [BxNxC]
            t: Parameter values                     [U]

        Returns:
            Tuple of
                positions     [BxUx3] 
                velocities    [BxUx3]
                accelerations [BxUx3]
        """

        p, tangent, normal = decasteljau_forward(control_points, t) 
        ctx.save_for_backward(control_points, p, tangent, normal, t)
        return p, tangent, normal

    @staticmethod
    def backward(ctx, grad_position, grad_tangent, grad_normal):
        """
        Args:
            ctx: The PyTorch autograd context
            grad_position: Gradients w.r.t. to the position   [BxUx3]
            grad_tangent: Gradients w.r.t. to the velocity    [BxUx3]
            grad_normal: Gradients w.r.t. to the acceleration [BxUx3]
        """
        control_points, p, tangent, normal, t = ctx.saved_tensors
        grad_P, grad_t = decasteljau_backward(control_points, t, p, tangent, normal, grad_position, grad_tangent, grad_normal)
        return grad_P, grad_t

def evaluate_bezier_curve(P: torch.tensor, u: torch.tensor):
    """ Evaluate a bezier curve of degree N-1 defined by N control points.

    Args:
        P: Control points of the curve [(Bx)Nx3]
        u: Parameter values            [(Bx)U]

    Returns:
        A tuple with 
            positions     [(Bx)Ux3] 
            velocities    [(Bx)Ux3]
            accelerations [(Bx)Ux3]
    """
    is_batched = len(P.shape) == 3

    if not is_batched:
        P = P.unsqueeze(0)

    position, velocity, acceleration = BezierCurveEvaluation.apply(P, u)

    if not is_batched:
        position = position.squeeze(0)
        velocity = velocity.squeeze(0)
        acceleration = acceleration.squeeze(0)

    return position, velocity, acceleration

def elevate_bezier_curve_degree(P: torch.tensor):
    """ Elevate the degree of a Bezier curve

    Args:
        P: Control points of the curve [(Bx)Nx3]

    Returns:
        Q: New control points of the curve [(Bx)N+1x3]
    """
    is_batched = len(P.shape) == 3
    if not is_batched:
        P = P.unsqueeze(0)

    shape = list(P.shape)
    n = shape[-2] - 1         # Current degree
    shape[-2] = shape[-2] + 1
    Q = torch.zeros(shape, dtype=P.dtype, device=P.device)

    # First and last control points remain unchanged
    Q[:,  0] = P[:, 0]
    Q[:, -1] = P[:, -1]

    i = torch.arange(1, n+1, device=P.device)[None, :, None]
    a = i/(n+1)
    # Q_i = i/(n+1)*P_{i-1} + (1-i/(n+1))*P_i
    Q[:, 1:(n+1)] = a * torch.roll(P, 1, 1)[:, 1:] + (1 - a) * P[:, 1:]

    if not is_batched:
        Q = Q.squeeze(0)

    return Q

def subdivide_bezier_curve(P: torch.Tensor, u: float = 0.5) -> torch.Tensor:
    """ Subdivide a Bezier curve using DeCasteljau's algorithm

    Args:
        P: Control points of the curve              [(Bx)Nx3]
        u: Parameter value used for the subdivision

    Returns:
        Q: New control points of the curve [(Bx)2xNx3]
    """

    n     = P.shape[1]
    u_inv = (1-u)
    Q = []
    R = []
    for k in range(0, n):
        Q += [ P[:,    0] ]
        R += [ P[:, -k-1] ]
        P = u_inv*P + u*torch.roll(P, shifts=-1, dims=1)

    return torch.stack([
        torch.stack(Q,       dim=1), # BxNx3
        torch.stack(R[::-1], dim=1)  # BxNx3
    ], dim=1)