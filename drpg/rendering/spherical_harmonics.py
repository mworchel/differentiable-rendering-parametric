import torch

# This class is adapted from the official code of the paper "Large Steps in Inverse Rendering of Geometry". 
# Please see their license for restrictions (https://github.com/rgl-epfl/large-steps-pytorch/blob/master/LICENSE)
# Copyright (c) 2021 Baptiste Nicolet
class SphericalHarmonics:
    """
    Environment map approximation using spherical harmonics.

    This class implements the spherical harmonics lighting model of [Ramamoorthi
    and Hanrahan 2001], that approximates diffuse lighting by an environment map.
    """

    def __init__(self, envmap):
        """
        Precompute the coefficients given an envmap.

        Parameters
        ----------
        envmap : torch.Tensor
            The environment map to approximate.
        """
        h,w = envmap.shape[:2]

        # NOTE: theta, phi, x, z, y differ from the original code
        # Compute the grid of theta, phi values
        theta = (torch.linspace(0, torch.pi, h,   device=envmap.device)).repeat(w, 1).t()
        phi   = (torch.linspace(0, 2*torch.pi, w, device=envmap.device)).repeat(h,1)

        # Compute the value of sin(theta) once
        sin_theta = torch.sin(theta)
        # Compute x,y,z
        # This differs from the original formulation as here the up axis is Y
        x = sin_theta * torch.sin(phi)
        z = sin_theta * torch.cos(phi)
        y = torch.cos(theta)

        # Compute the polynomials
        Y_0 = 0.282095
        # The following are indexed so that using Y_n[-p]...Y_n[p] gives the proper polynomials
        Y_1 = [
            0.488603 * z,
            0.488603 * x,
            0.488603 * y
            ]
        Y_2 = [
            0.315392 * (3*z.square() - 1),
            1.092548 * x*z,
            0.546274 * (x.square() - y.square()),
            1.092548 * x*y,
            1.092548 * y*z
        ]
        import matplotlib.pyplot as plt
        area = w*h
        radiance = envmap[..., :3]
        dt_dp = 2.0 * torch.pi**2 / area

        # Compute the L coefficients
        L = [ [(radiance * Y_0 * (sin_theta)[..., None] * dt_dp).sum(dim=(0,1))],
            [(radiance * (y * sin_theta)[..., None] * dt_dp).sum(dim=(0,1)) for y in Y_1],
            [(radiance * (y * sin_theta)[..., None] * dt_dp).sum(dim=(0,1)) for y in Y_2]]

        # Compute the R,G and B matrices
        c1 = 0.429043
        c2 = 0.511664
        c3 = 0.743125
        c4 = 0.886227
        c5 = 0.247708

        self.M = torch.stack([
            torch.stack([ c1 * L[2][2] , c1 * L[2][-2], c1 * L[2][1] , c2 * L[1][1]           ]),
            torch.stack([ c1 * L[2][-2], -c1 * L[2][2], c1 * L[2][-1], c2 * L[1][-1]          ]),
            torch.stack([ c1 * L[2][1] , c1 * L[2][-1], c3 * L[2][0] , c2 * L[1][0]           ]),
            torch.stack([ c2 * L[1][1] , c2 * L[1][-1], c2 * L[1][0] , c4 * L[0][0] - c5 * L[2][0]])
        ]).movedim(2,0)

    def eval(self, n):
        """
        Evaluate the shading using the precomputed coefficients.

        Parameters
        ----------
        n : torch.Tensor
            Array of normals at which to evaluate lighting.
        """
        normal_array = n.view((-1, 3))
        h_n = torch.nn.functional.pad(normal_array, (0,1), 'constant', 1.0)
        l = (h_n.t() * (self.M @ h_n.t())).sum(dim=1)
        return l.t().view(n.shape)