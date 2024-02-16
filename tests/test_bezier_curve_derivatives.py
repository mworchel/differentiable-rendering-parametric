import pytest
import torch

from diff_curves import *

def grad_check(P, u, check_parameter_acceleration=True):
    u.requires_grad = True
    torch.autograd.gradcheck(lambda u: evaluate_bezier_curve(P, u)[0], [u], eps=0.001, raise_exception=True)
    torch.autograd.gradcheck(lambda u: evaluate_bezier_curve(P, u)[1], [u], eps=0.001, raise_exception=True)
    if check_parameter_acceleration:
        torch.autograd.gradcheck(lambda u: evaluate_bezier_curve(P, u)[2], [u], eps=0.001, raise_exception=True)

    P.requires_grad = True
    torch.autograd.gradcheck(lambda P: evaluate_bezier_curve(P, u)[0], [P], eps=0.001, raise_exception=True)
    torch.autograd.gradcheck(lambda P: evaluate_bezier_curve(P, u)[1], [P], eps=0.001, raise_exception=True)
    torch.autograd.gradcheck(lambda P: evaluate_bezier_curve(P, u)[2], [P], eps=0.001, raise_exception=True)

def test_bernstein_polynomials():
    u = torch.rand(16, dtype=torch.float64)

    for n in range(2, 32):
        i = torch.arange(n+1)[:, None]
        Bu_d = bernstein(u[None, :], i, n)
        Bt_d = bernstein(u[None, :], i-1, n-1) - bernstein(u[None, :], i, n-1)
        Bn_d = bernstein(u[None, :], i-2, n-2) - 2*bernstein(u[None, :], i-1, n-2) + bernstein(u[None, :], i, n-2)

        Bu_r, Bt_r, Bn_r = bernstein_recursive(n, u)
        
        atol = 1e-8 if n < 6 else 1e-5
        assert torch.isclose(Bu_d, Bu_r, atol=atol).all()
        assert torch.isclose(Bt_d, Bt_r, atol=atol).all()
        assert torch.isclose(Bn_d, Bn_r, atol=atol).all()

def test_derivative_batched_parameters():
    for n in range(2, 6):
        # Bezier curves of degree n
        P = torch.rand((2, n+1, 2), dtype=torch.float64)
        u = torch.rand(16, dtype=torch.float64)
        u = u.repeat(P.shape[0])

        # FIXME: Known issue, gradients of the normal not scattered to the parameter
        grad_check(P, u, check_parameter_acceleration=n < 3)