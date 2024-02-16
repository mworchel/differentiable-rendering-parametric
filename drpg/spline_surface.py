from enum import Enum, IntEnum
import numpy as np
from pathlib import Path
import torch
from typing import Union, List

class Side(IntEnum):
    Top    = 0
    Left   = 1
    Bottom = 2
    Right  = 3

class CoordinateMergeMethod(Enum):
    PrimaryPrecedence = 0
    Average           = 1

class ParametricPatches:
    """ A collection of parametric patches with connectivity data

    Args:
        n: Number of control points in vertical direction   (degree = n-1)
        m: Number of control points in horizontal direction (degree = m-1)
           If m is None, it is assumed that m=n.
        device: 
    """

    def __init__(self, n: int, m: int = None, device: torch.device = None):
        self.n = n
        self.m = n if m is None else m
        self.V = torch.zeros((0, 3), dtype=torch.float32, device=device)           # Control point positions
        self.F = torch.zeros((0, self.n, self.m), dtype=torch.long, device=device) # Indices for the patches

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'n': self.n,
            'm': self.m,
            'V': self.V.cpu().detach().numpy(),
            'F': self.F.cpu().detach().numpy(),
        }
        np.savez(path, **data)

    @classmethod
    def load(cls, path: Path, device=torch.device):
        data = np.load(path)
        n         = int(data['n'])
        m         = int(data['m']) if 'm' in data else None # Legacy: did not include m
        patches   = cls(n=n, m=m, device=device)
        patches.V = torch.from_numpy(data['V']).to(device=device)
        patches.F = torch.from_numpy(data['F']).to(device=device)
        return patches

    def with_control_points(self, V: torch.Tensor):
        assert V.shape == self.V.shape
        patches_new   = ParametricPatches(self.n, self.m, device=self.V.device)
        patches_new.V = V
        patches_new.F = self.F
        return patches_new

    def add(self, P: torch.Tensor) -> Union[int, List[int]]:
        """ Add the control mesh of a patch or multiple patches
        
        Args:
            P: Tensor with control points with shape ((B,)n,m,3)

        Returns:
            index: Global index of the patch(es)
        """
        if len(P.shape) == 3:
            P = P.unsqueeze(0)

        assert P.shape[1] == self.n
        assert P.shape[2] == self.m
        assert P.shape[3] == 3

        num_patches = P.shape[0]

        v = P.reshape(-1, 3)
        f = (torch.arange(v.shape[0]) + self.V.shape[0]).reshape(num_patches, self.n, self.m)

        index = [self.F.shape[0] + i for i in range(num_patches)]
        self.V = torch.cat([self.V, v.to(self.V.device, dtype=self.V.dtype)])
        self.F = torch.cat([self.F, f.to(self.F.device, dtype=self.F.dtype)])

        return index if num_patches > 1 else index[0]

    def merge_duplicate_vertices(self, tolerance=1e-6):
        V_truncated = torch.round(self.V / tolerance) * tolerance
        V_truncated_unique, indices, counts = torch.unique(V_truncated, sorted=False, return_inverse=True, return_counts=True, dim=0)

        V_unique = torch.zeros_like(V_truncated_unique)
        V_unique.scatter_add_(0, indices.unsqueeze(1).expand_as(self.V), self.V)
        self.V = V_unique / counts.unsqueeze(1)

        self.F = indices[self.F]

    def get_side_indices(self, i: int, side: Side, offset: int = 0) -> torch.Tensor:
        """ Get the control point indices of a specific side of patch i

        Args:
            i: Identifier of the patch
            side: The side
            offset: Offset to the side (e.g. to get the 2nd row from the top)

        Returns:
            Array with control point indices of shape (3,)
        """

        if side == Side.Top:
            return self.F[i, 0 + offset, :]
        if side == Side.Left:
            return self.F[i, :, 0 + offset]
        if side == Side.Bottom:
            return self.F[i, -1 - offset, :]
        if side == Side.Right:
            return self.F[i, :, -1 - offset]
        
        return None
    
    def set_side_indices(self, i: int, side: Side, f: torch.Tensor):
        if side == Side.Top:
            self.F[i, 0, :] = f
        if side == Side.Left:
            self.F[i, :, 0] = f
        if side == Side.Bottom:
            self.F[i, -1, :] = f
        if side == Side.Right:
            self.F[i, :, -1] = f

    def connect(self, i: int, j: int, side_i: Side, side_j: Side, merge_method: CoordinateMergeMethod = CoordinateMergeMethod.PrimaryPrecedence):
        """ Connect two parametric patches

        Args:
            i:      Index of the primary patch
            j:      Index of the secondary patch
            side_i: Connection side of the primary patch
            side_j: Connection side of the secondary patch
        """
        assert i >= 0 and i < self.F.shape[0]
        assert j >= 0 and j < self.F.shape[0]

        print("WARNING: ParametricPatches::connect is deprecated in the current form.")

        Fi = self.get_side_indices(i, side_i)
        Fj = self.get_side_indices(j, side_j)
        if merge_method == CoordinateMergeMethod.Average:
            self.V[Fi] = 0.5*(self.V[Fi] + self.V[Fj])

        self.set_side_indices(j, side_j, Fi)