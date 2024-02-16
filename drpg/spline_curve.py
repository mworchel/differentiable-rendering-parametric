from enum import Enum, IntEnum
import torch
from typing import Union, List

class ParametricCurves:
    """ A collection of parametric curves

    Args:
        d: Dimension of the curve (2 or 3)
        n: Number of control points for each curve (degree = n-1)
    """

    def __init__(self, n: int, d:int, device: torch.device):
        self.n = n
        self.d = d
        self.V = torch.zeros((0, d),      dtype=torch.float32, device=device) # Control point positions
        self.F = torch.zeros((0, self.n), dtype=torch.long, device=device)    # Indices for the curves

    def with_control_points(self, V: torch.Tensor):
        assert V.shape == self.V.shape
        curves_new   = ParametricCurves(self.n, self.d, device=self.V.device)
        curves_new.V = V
        curves_new.F = self.F
        return curves_new

    def add(self, P: torch.Tensor) -> Union[int, List[int]]:
        """ Add the control mesh of a curve
        
        Args:
            P: Tensor with control points with shape ((B,)n,d)

        Returns:
            index: Global index of the curve
        """
        if len(P.shape) == 2:
            P = P.unsqueeze(0)

        assert P.shape[1] == self.n
        assert P.shape[2] == self.d

        num_curves = P.shape[0]

        v = P.reshape(-1, self.d)
        f = (torch.arange(v.shape[0]) + self.V.shape[0]).reshape(num_curves, self.n)

        index = self.F.shape[0] + torch.arange(num_curves, dtype=torch.long, device=P.device)
        self.V = torch.cat([self.V, v.to(self.V.device, dtype=self.V.dtype)])
        self.F = torch.cat([self.F, f.to(self.F.device, dtype=self.F.dtype)])

        return index if num_curves > 1 else index[0]

    def merge_duplicate_vertices(self, tolerance=1e-6):
        V_truncated = torch.round(self.V / tolerance) * tolerance
        V_truncated_unique, indices, counts = torch.unique(V_truncated, sorted=False, return_inverse=True, return_counts=True, dim=0)

        V_unique = torch.zeros_like(V_truncated_unique)
        V_unique.scatter_add_(0, indices.unsqueeze(1).expand_as(self.V), self.V)
        self.V = V_unique / counts.unsqueeze(1)

        self.F = indices[self.F]