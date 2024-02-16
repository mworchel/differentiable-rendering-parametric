import torch

def decasteljau_forward(P: torch.Tensor, u: torch.Tensor):
    """ Forward pass of the de Casteljau algorithm
    Args:
        P: Control points with shape    [BxNxC]
        u: Parameter values: With shape [(Bx)U]

    Returns:
        Curves evaluated at the parameter values [BxUxC]
    """

    has_batched_parameter_values = len(u.shape) == 2
    if has_batched_parameter_values and P.shape[0] != u.shape[0]:
        raise RuntimeError(f"Batch size must match for control points ({P.shape[0]}) and parameter values ({u.shape[0]})")

    if not has_batched_parameter_values:
        u = u[None, :]

    points = P
    # points (B, N, U, C)
    # u      (B, 1, U, 1)
    points = points[:, :, None, :].repeat(1, 1, u.shape[1], 1)
    u      = u[:, None, :, None]

    normal = torch.zeros_like(points[:, 0])

    n     = P.shape[1]
    u_inv = (1-u)
    for k in range(0, n):
        if k == n - 1:
            tangent = points[:, 1] - points[:, 0]
        if n > 2 and k == n - 2:
            normal = points[:, 2] - 2*points[:, 1] + points[:, 0]

        if k == 0:
            continue

        points = u_inv*points + u*torch.roll(points, shifts=-1, dims=1)

    return points[:, 0], tangent, normal

def torch_binom(n, k):
    mask = n >= k
    n = mask * n
    k = mask * k
    a = torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1)
    return torch.exp(a) * mask

def bernstein(t, i, n):
    # TODO: Make efficient
    return torch.where((i < 0) | (i > n), torch.zeros_like(t), torch_binom(n, i) * t**i * (1-t)**(n-i))

def bernstein_recursive(n: int, u: torch.tensor):
    """ Evaluate the Bernstein basis polynomials at given parameter values

    Using the recursion, this function is fast for small n and numerically stable.
    The implementation of the recursive definition is iterative.
    
    Args:
        n: Degree of the Bernstein basis polynomials
        u: Parameter values                          [B, U]

    Returns:
        Bu: Evaluated Bernstein basis polynomials                                                        [B, n+1, U]
        Bt: Evaluated Bernstein basis for tangent derivatives B^{n-1}_{i-1} - B^{n-1}_i                  [B, n+1, U]
        Bn: Evaluated Bernstein basis for normal derivatives B^{n-2}_{i-2} - 2*B^{n-2}_{i-1} + B^{n-2}_i [B, n+1, U]
    """

    is_batched = len(u.shape) == 2

    if not is_batched:
        u = u[None, :]

    B, U = u.shape[0], u.shape[1]
    Bu = torch.zeros((B, n+1, U), dtype=u.dtype, device=u.device)
    Bu[:, 0, :] = 1 # B^0_0
    
    Bt = torch.zeros_like(Bu)
    Bn = torch.zeros_like(Bu)

    u = u[:, None, :]
    u_inv = (1-u)
    for k in range(1, n+1):
        if k == n:
            # B^(n-1)_i-1 - B^(n-1)_i
            Bt = torch.roll(Bu, shifts=1, dims=1) - Bu
        
        if k == n-1:
            # B^(n-2)_i-2 - 2*B^(n-2)_i-1 + B^(n-2)_i
            Bn = torch.roll(Bu, shifts=2, dims=1) - 2*torch.roll(Bu, shifts=1, dims=1) + Bu

        # Recursion: u*B^k_i-1 + (1-u)*B^k_i
        Bu = u*torch.roll(Bu, shifts=1, dims=1) + u_inv*Bu

    if not is_batched:
        Bu = Bu.squeeze(0)
        Bt = Bt.squeeze(0)
        Bn = Bn.squeeze(0)

    return Bu, Bt, Bn

def decasteljau_backward(P, u, position, tangent, normal, grad_position, grad_tangent, grad_normal, recursive_bernstein=True):
    """ Backward pass of the de Casteljau algorithm

    Args:
        P: Control points from the forward pass          [BxNxC]
        u: Parameter values from the forward pass        [(Bx)U]
        position: Positions from the forward pass        [BxUxC]
        tangent: Tangents from the forward pass          [BxUxC]
        grad_position: Gradient w.r.t. to the position   [BxUxC]
        grad_tangent: Gradient w.r.t. to the tangent     [BxUxC]
        grad_normal: Gradient w.r.t. to the normal       [BxUxC]
    """

    num_control_points = P.shape[1]
    n = num_control_points - 1

    # u = [B, U]
    if len(u.shape) == 1:
        u = u[None, :]

    # Gradient w.r.t parameter values is almost free
    grad_u = (n*tangent * grad_position).sum(dim=-1)
    grad_u += (n*(n-1)*normal * grad_tangent/n).sum(dim=-1)
    # TODO: Scatter the normal gradients back to the parameters

    if recursive_bernstein == True:
        Bu, Bt, Bn = bernstein_recursive(n, u)
    else:
        # i = [1, N, 1]
        i = torch.arange(num_control_points)[None, :, None].to(P.device)
        # u = [B, 1, U]
        # B = [B, N, U]
        Bu = bernstein(u[:, None, :], i, n)              #  B     N  U  C
        Bt = bernstein(u[:, None, :], i-1, n-1) - bernstein(u[:, None, :], i, n-1)
        Bn = bernstein(u[:, None, :], i-2, n-2) - 2*bernstein(u[:, None, :], i-1, n-2) + bernstein(u[:, None, :], i, n-2)

    grad_P = (Bu[:, :, :, None] * grad_position[:, None, :, :]).sum(dim=2)

    # Scatter grad_tangent to the control points
    # Tangent is missing the 'n *' factor in the forward pass, so it is not needed here either
    grad_P += (Bt[:, :, :, None] * grad_tangent[:, None, :, :]).sum(dim=2)

    # Scatter grad_normal to the control points
    # Normal is missing the 'n * (n-1)' factor in the forward pass, so it is not needed here either
    grad_P += (Bn[:, :, :, None] * grad_normal[:, None, :, :]).sum(dim=2)

    return grad_P, grad_u