from enum import Enum, IntEnum
import torch
from typing import Optional, Tuple

from .casteljau import torch_binom
from .geometry import generate_grid_faces
from .spline_surface import ParametricPatches
from .spline_curve import ParametricCurves

def is_side_contracted(f: torch.Tensor) -> bool:
    return f[0] == f[1] and f[1] == f[2]

def build_tangent_map(patches: ParametricPatches):
    """ Build the tangent/edge map data structure from a set of patches
    """
    N = patches.n

    # Shape (4*B,N)
    edges = torch.cat([
        patches.F[:,  0,  :], # Top    (B,N)
        patches.F[:,  :,  0], # Left
        patches.F[:, -1,  :], # Bottom
        patches.F[:,  :, -1], # Right
    ], dim=0)

    borders = torch.cat([
        patches.F[:,  1,  :], # Top    (B,N)
        patches.F[:,  :,  1], # Left
        patches.F[:, -2,  :], # Bottom
        patches.F[:,  :, -2], # Right
    ], dim=0)

    # Fully ignore contracted sides
    # We detect contracted sides by flipping the edge indices and compare for equality
    # TODO: Could run into some degenerate cases
    uncontracted_mask = ~((edges == edges.flip(-1)).all(dim=-1))
    edges   = edges[uncontracted_mask]
    borders = borders[uncontracted_mask]

    # Define an order for the sides by having the smaller index first
    # [[2, 0, 1], [1, 0, 2]] -> [[1, 0, 2], [1, 0, 2]]
    flip_mask          = edges[:, 0] > edges[:, -1]
    edges[flip_mask]   = edges[flip_mask].flip(1)
    borders[flip_mask] = borders[flip_mask].flip(1)

    # Find unique edges
    # `inverse` defines a mapping from the unique array to the original array
    _, inverse, counts = edges.unique(dim=0, return_inverse=True, return_counts=True)

    # The edge mask defines for each *unique* edge if it is incident on two patches.
    # Scatter this information to all edges using `inverse`
    edge_mask = (counts == 2)[inverse]

    # `inverse` contains the *unique* index each edge is mapped to
    # `inverse` has the same shape as `edges.shape[0]`
    # and contains the index to the unique edge for each edge 
    # Sorting this array gives us `indices` which we can use to group
    # edges in the original array by their unique edge.
    _, indices = torch.sort(inverse)

    # Now gather edges and border points
    Fi0Gi0 = edges[indices][edge_mask[indices]]
    Fi1Gi1 = borders[indices][edge_mask[indices]]

    Fi0, Gi0 = Fi0Gi0.reshape(-1, 2, N).unbind(1)
    Fi1, Gi1 = Fi1Gi1.reshape(-1, 2, N).unbind(1)

    return torch.stack([
        torch.stack([Fi0, Fi1], dim=1),
        torch.stack([Gi0, Gi1], dim=1)
    ], dim=1)

class C1Method(Enum):
    Strict  = 0
    Relaxed = 1

def c1_loss(patches: ParametricPatches, tangent_map: torch.Tensor, method: C1Method = C1Method.Strict) -> torch.Tensor:
    """
    Args:
        patches: Set of parametric patches
        tangent_map: Edge map storing patch adjacency data (Ex2x2x3)
        method: Method used for computing the C1 regularization
    """

    P0, P1 = patches.V[tangent_map].unbind(2)
    F, G   = (P1 - P0).unbind(1)

    if method == C1Method.Strict:
        return ((F + G)**2).sum(-1).mean()
    elif method == C1Method.Relaxed:
        eps: float     = 1e-12
        F_norm = F.norm(dim=-1, keepdim=True).clamp_min(eps)
        G_norm = G.norm(dim=-1, keepdim=True).clamp_min(eps)
        FdotG = (F/F_norm*(-G)/G_norm).sum(-1)
        return (1 - FdotG).mean() + ((F_norm - G_norm)**2).mean()
        #return  (1 - torch.cosine_similarity(F, -G, dim=1)).mean()
        
class G1Method(Enum):
    Determinant        = 0
    CrossProduct       = 1 # This method corresponds to Equation (24) in the paper

def g1_loss(patches: ParametricPatches, tangent_map: torch.Tensor, method: G1Method = G1Method.CrossProduct) -> torch.Tensor:
    """ Compute the G1 regularization
    """

    # The realized tangent map as shape (E,2,2,N,3)
    tangent_map_realized = patches.V[tangent_map]

    # M contains the control point positions along the edges
    # Note that `tangent_map_realized[:, 0, 0]` == `tangent_map_realized[:, 1, 0]`
    M = tangent_map_realized[:, 0, 0]
    H = M.roll(shifts=(-1), dims=(1,)) - M
    H = H[:, :-1]

    P0, P1 = tangent_map_realized.unbind(2)
    F, G   = (P1 - P0).unbind(1)

    # H    has shape   (E,N-1,3)
    # F, G have shapes (E,N,3)

    # Pre-multiply H, F, G with the coefficients
    H = torch_binom(H.shape[1]-1, torch.arange(H.shape[1], device=H.device))[None, :, None] * H
    F = torch_binom(F.shape[1]-1, torch.arange(F.shape[1], device=F.device))[None, :, None] * F
    G = torch_binom(G.shape[1]-1, torch.arange(G.shape[1], device=G.device))[None, :, None] * G

    n = patches.n-1 # n is the degree here
    i = torch.arange(n-1 + 1, dtype=torch.long, device=patches.V.device)
    j = torch.arange(n   + 1, dtype=torch.long, device=patches.V.device)

    if method == G1Method.Determinant:
        # Original: Determinant-based version

        # This code generates a tensor with indices (n=2 in this example; biquadratic patches)
        # i | 0 0 0 0 0 0 0 0 0 1 1 1 ...
        # j | 0 0 0 1 1 1 2 2 2 0 0 0 ...
        # k | 0 1 2 0 1 2 0 1 2 0 1 2 ...
        k   = torch.arange(n + 1, dtype=torch.long, device=patches.V.device)
        jk  = torch.stack(torch.meshgrid(j, k, indexing='ij'), dim=-1).reshape(-1, 2)
        ijk = torch.stack([
            *torch.meshgrid(i, jk[:, 0], indexing='ij'), 
            torch.meshgrid(i, jk[:, 1], indexing='ij')[1]
        ], dim=-1).reshape(-1, 3)

        # HFG has shape (E,B,3,3), where B is the size of the ijk tensor
        HFG = torch.stack([H[:, ijk[:, 0]], F[:, ijk[:, 1]], G[:, ijk[:, 2]]], dim=-1)
        
        # cm has shape (E,D+1,3) where D is the degree of the Bezier curve (Equation X in the paper)
        D  = n-1 + n + n
        cm = torch.zeros((HFG.shape[0], D + 1, 3), dtype=torch.float32, device=patches.V.device)

        # Compute the indices m=i+j+k for each permutation of indices
        # ijk.shape=(B,3), so m.shape=(1,B,1)
        m = ijk.sum(dim=-1, keepdim=True)[None]

        # detHFG has shape (E,B,1)
        detHFG = torch.linalg.det(HFG.reshape(-1, 3, 3)).reshape(HFG.shape[0], HFG.shape[1], 1)
        cm     = torch.scatter_add(cm, 1, m.expand_as(detHFG), detHFG)

        loss = (cm**2).mean() 
    elif method == G1Method.CrossProduct:
        # Modified: Cross product-based version

        ij = torch.stack(torch.meshgrid(i, j, indexing='ij'), dim=-1).reshape(-1, 2)

        D   = n-1 + n
        Fcm = torch.zeros((H.shape[0], D + 1, 3), dtype=torch.float32, device=patches.V.device)
        Gcm = torch.zeros((H.shape[0], D + 1, 3), dtype=torch.float32, device=patches.V.device)
        m  = ij.sum(dim=-1, keepdim=True)[None] # Determine the index

        # HxF and GxH have shapes (E,B,3)
        HxF = torch.cross(H[:, ij[:, 0]], F[:, ij[:, 1]], dim=-1)
        GxH = torch.cross(G[:, ij[:, 1]], H[:, ij[:, 0]], dim=-1)

        Fcm = torch.scatter_add(Fcm, 1, m.expand_as(HxF), HxF)
        Gcm = torch.scatter_add(Gcm, 1, m.expand_as(GxH), GxH)

        # mean over coordinates and all quantities to compare
        loss = ((Fcm - Gcm)**2).sum(-1).mean()
    else:
        raise RuntimeError(f"Method '{method}' not implemented")

    return loss

def build_curve_tangent_map(curves: ParametricCurves):
    """ Build the tangent/joint map data structure from a set of curves
    """

    joints = torch.cat([
        curves.F[:, -1],
        curves.F[:,  0],
    ], dim=0)

    borders = torch.cat([
        curves.F[:, -2],
        curves.F[:,  1],
    ], dim=0)

    _, inverse, counts = joints.unique(dim=0, return_inverse=True, return_counts=True)
    _, indices = torch.sort(inverse)

    # print(indices)

    joint_mask = (counts == 2)[inverse]

    # print(joint_mask)

    F0G0   = joints[indices][joint_mask[indices]]
    F1G1 = borders[indices][joint_mask[indices]]
    
    F0, G0 = F0G0.reshape(-1, 2).unbind(1)
    F1, G1 = F1G1.reshape(-1, 2).unbind(1)

    return torch.stack([
        torch.stack([F0, F1], dim=1),
        torch.stack([G0, G1], dim=1)
    ], dim=1)

def c1_curve_loss(curves: ParametricPatches, tangent_map: torch.Tensor) -> torch.Tensor:
    """
    Args:
        curves: Set of parametric curves
        tangent_map: Joint map storing curve adjacency data (Ex2x2)
    """

    P0, P1 = curves.V[tangent_map].unbind(2)
    F, G   = (P1 - P0).unbind(1)
    return ((F + G)**2).sum(-1).mean()

def g1_curve_loss(curves: ParametricPatches, tangent_map: torch.Tensor) -> torch.Tensor:
    """
    Args:
        curves: Set of parametric curves
        tangent_map: Joint map storing curve adjacency data (Ex2x2)
    """

    P0, P1 = curves.V[tangent_map].unbind(2)
    F, G   = (P1 - P0).unbind(1)
    return (1 - torch.cosine_similarity(F, -G, dim=-1)).mean()

class MeshType(IntEnum):
    Triangle = 3
    Quad     = 4

def convert_patches_to_mesh(patches: ParametricPatches, type: MeshType, faces_per_patch: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Convert a set parametric patches to a quad mesh.
    
    Args:
        patches: Data structure for the patches
        type: Type of the output mesh 

    Returns:
        V: Vertices of the mesh (Vx3)
        F: Faces of the mesh    (Fx3) or (Fx4)
    """

    num_control_points_per_patch = patches.n*patches.m

    # We first generate a index set that defines a quad mesh for one patch
    F_grid = generate_grid_faces(patches.n, patches.m, quads=(type==MeshType.Quad)).to(patches.V.device)

    # Now access all patches with the index set to generate quads from the control point indices
    # (Order of the indices in F_grid matches the order of the control points)
    # A -- B -- C                                             [1, 0, 4, 5]            [B, A, D, E]
    # |    |    |                                             [2, 1, 5, 6]            [C, B, E, F]
    # D -- E -- F --reshape--> [A, B, C, D, E, F, G, H, I] --      ...     --index-->      ...
    # |    |    |
    # G -- H -- I
    #    patch
    F = patches.F.reshape(-1, num_control_points_per_patch)[:, F_grid]

    vertices_per_face = int(type)
    F = F.reshape(-1, vertices_per_face) if not faces_per_patch else F

    return patches.V, F

def convert_curves_to_graph(curves: ParametricCurves):
    """ Convert a set parametric curves to a graph.
    
    Args:
        curves: Data structure for the curves

    Returns:
        V: Vertices (Vx3)/(Vx2)
        E: Edges    (Ex2)
    """

    F = torch.stack([curves.F[:, :-1], curves.F[:, 1:]], dim=-1).reshape(-1, 2)

    return curves.V, F

def laplacian_uniform_all(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """ Compute the uniform Laplacian for a triangle or quad mesh, or a general graph.

    Code adapted from https://github.com/rgl-epfl/large-steps-pytorch/blob/master/largesteps/geometry.py#L65
    
    Args:
        V: Vertices of the mesh (Vx3)
        F: Faces of the mesh    (Fx3) or (Fx4) or (Fx2)

    Returns:
        L: Laplacian matrix     (VxV) (sparse)
    """
    
    if F.shape[1] == 3:   # Triangle mesh
        ii = F[:, [0, 1, 2]].flatten()
        jj = F[:, [1, 2, 0]].flatten()
    elif F.shape[1] == 4: # Quad mesh
        ii = F[:, [0, 1, 2, 3]].flatten()
        jj = F[:, [1, 2, 3, 0]].flatten()
    elif F.shape[1] == 2: # Graph
        ii = F[:, 0]
        jj = F[:, 1]

    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    
    adj_values = torch.ones(adj.shape[1], device=V.device, dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx    = torch.cat([adj, torch.stack([diag_idx, diag_idx], dim=0)], dim=1)
    values = torch.cat([-adj_values, adj_values])

    # The coalesce operation sums the duplicate indices, resulting in the correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V.shape[0], V.shape[0])).coalesce()

def compute_lsig_matrix(verts: torch.Tensor, faces: torch.Tensor, lambda_: float, alpha: Optional[float]=None) -> torch.Tensor:
    """
    Build the parameterization matrix for a triangle mesh, quad mesh, or a general graph.

    If alpha is defined, then we compute it as (1-alpha)*I + alpha*L otherwise
    as I + lambda*L as in the paper. The first definition can be slightly more
    convenient as it the scale of the resulting matrix doesn't change much
    depending on alpha.

    Code from https://github.com/rgl-epfl/large-steps-pytorch/blob/master/largesteps/geometry.py#L96-L133

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    lambda_ : float
        Hyperparameter lambda of our method, used to compute the
        parameterization matrix as (I + lambda_ * L)
    alpha : float in [0, 1[
        Alternative hyperparameter, used to compute the parameterization matrix
        as ((1-alpha) * I + alpha * L)
    """
    L = laplacian_uniform_all(verts, faces)

    idx = torch.arange(verts.shape[0], dtype=torch.long, device=verts.device)
    eye = torch.sparse_coo_tensor(torch.stack((idx, idx), dim=0), torch.ones(verts.shape[0], dtype=torch.float, device=verts.device), (verts.shape[0], verts.shape[0]))
    if alpha is None:
        M = torch.add(eye, lambda_*L) # M = I + lambda_ * L
    else:
        if alpha < 0.0 or alpha >= 1.0:
            raise ValueError(f"Invalid value for alpha: {alpha} : it should take values between 0 (included) and 1 (excluded)")
        M = torch.add((1-alpha)*eye, alpha*L) # M = (1-alpha) * I + alpha * L
    return M.coalesce()