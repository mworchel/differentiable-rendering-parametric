from enum import Enum
import torch
from typing import Tuple, Optional

from .bezier_curve import evaluate_bezier_curve
from .bezier_surface import evaluate_bezier_surface_on_grid, evaluate_bezier_surface
from .geometry import generate_grid_faces, compute_frames_double_reflection, recompute_frames
from .spline_curve import ParametricCurves
from .spline_surface import ParametricPatches

class SurfaceType(Enum):
    Bezier = 0
    NURBS  = 1

class SurfaceTessellator:
    def __init__(self, uv_grid_resolution: Tuple[int, int], type: SurfaceType):
        self.uv_grid_resolution = uv_grid_resolution
        self.type = type
        self.eval_nurbs_cache = {}

    def tessellate(self, patches: ParametricPatches, degree: Optional[int] = None, weights: Optional[torch.Tensor] = None, adaptive_threshold: bool = None, return_per_patch: bool = False, max_iterations=10, u: Optional[torch.Tensor] = None, v: Optional[torch.Tensor] = None, return_uv: bool = False, no_face_offset: bool =False) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """ Tessellate a spline surface
        
        Args:

        """

        device = patches.V.device

        if u is None:
            u = torch.linspace(0, 1, self.uv_grid_resolution[0], device=device)
        if v is None:
            v = torch.linspace(0, 1, self.uv_grid_resolution[1], device=device)

        if self.type == SurfaceType.Bezier:
            # p has shape (F,N,M,3) where F is the number of patches
            p, tu, tv = evaluate_bezier_surface_on_grid(patches.V[patches.F], u, v, with_tangents=True) 
        else: # self.type == SurfaceType.NURBS
            if degree is None:
                raise ValueError("Parameter `degree` is required for NURBS tessellation.")

            # Lazily create module for evaluating NURBS surfaces
            from NURBSDiff.surf_eval import SurfEval
            nurbsdiff_device = 'cpu' if patches.V.device == torch.device('cpu') else 'cuda'
            key = (patches.n, patches.m, degree, nurbsdiff_device)
            if not key in self.eval_nurbs_cache:
                self.eval_nurbs_cache[key] = SurfEval(patches.m, patches.n, p=degree, q=degree, out_dim_u=self.uv_grid_resolution[0], out_dim_v=self.uv_grid_resolution[1], dvc=nurbsdiff_device)
            eval_nurbs: SurfEval = self.eval_nurbs_cache[key]

            P = patches.V[patches.F] # (F,N,M,3)
            P = P.transpose(1, 2)

            if weights is None:
                weights = torch.ones_like(P[..., :1])                

            p = eval_nurbs(torch.cat([P, weights], dim=-1)) # (F,V,U,3)
            p = p.transpose(2, 1) # (F,U,V,3)
    	    
            tu = None
            tv = None

        # 
        num_patches            = patches.F.shape[0]
        num_vertices_per_patch = p.shape[1]*p.shape[2]

        f_grid   = generate_grid_faces(v.shape[0], u.shape[0]).to(device)
        uvs_grid = torch.stack(torch.meshgrid([v, u], indexing='ij')[::-1], dim=-1).reshape(-1, 2)

        if adaptive_threshold is None or self.type == SurfaceType.NURBS:
            # Early exit when no adaptive tessellation
            vertices = p.reshape(-1, 3)  if not return_per_patch else p
            
            if no_face_offset:
                f    = f_grid[None].repeat(num_patches, 1, 1)
            else:
                f    = f_grid.unsqueeze(0) + num_vertices_per_patch * torch.arange(0, num_patches, device=device).view(-1, 1, 1)
            faces    = f.reshape(-1, 3)  if not return_per_patch else f

            tu       = tu.reshape(-1, 3) if ((not return_per_patch) and (tu is not None)) else tu
            tv       = tv.reshape(-1, 3) if ((not return_per_patch) and (tv is not None)) else tv

            result = [vertices, faces, tu, tv]
            if return_uv:
                uv = uvs_grid[None, :, :].repeat(num_patches, 1, 1).reshape(num_patches, v.shape[0], u.shape[0], 2)
                result += [ uv.reshape(-1, 2) if not return_per_patch else uv ]

            return *result,

        # Adaptive tessellation
        # Currently refines each patch individually (TODO: vectorize)
        V_all  = []
        f_all  = []
        tu_all = []
        tv_all = []
        uv_all = []
        num_vertices = 0
        for i in range(p.shape[0]):
            # Get vertex positions and their uv coordinates
            V   = p[i].reshape(-1, 3)
            uvs = uvs_grid
            tus = tu.reshape(-1, 3)
            tvs = tv.reshape(-1, 3)

            # Subdivide as long as necessary
            faces_current = f_grid
            faces         = torch.zeros((0, 3), dtype=torch.long, device=f_grid.device)
            j = 0
            while faces_current.shape[0] != 0 and j < max_iterations:
                #print(j, faces_current.shape[0])
                j += 1

                edges                       = torch.cat([faces_current[:, [0, 1]], faces_current[:, [1, 2]], faces_current[:, [2, 0]]], dim=0)
                edges, _                    = torch.sort(edges, dim=1)
                edges_unique, edges_inverse = torch.unique(edges, return_inverse=True, dim=0)

                # Compute edge midpoints and their parameter values
                edge_midpoints    = V[edges_unique].mean(dim=1)
                edge_midpoints_uv = uvs[edges_unique].mean(dim=1)

                # Evaluate the surface at the edge midpoint parameter values
                # FIXME: Do not re-test edges that we have already tested
                edge_midpoints_on_surface, tu_edge, tv_edge = evaluate_bezier_surface(patches.V[patches.F[i]], edge_midpoints_uv[:, 0], edge_midpoints_uv[:, 1])

                # Determine the edges to split
                edge_error      = (edge_midpoints_on_surface - edge_midpoints).norm(dim=-1)
                edge_split_mask = edge_error > adaptive_threshold


                # Insert edge midpoints as new vertices
                V_edges = edge_midpoints_on_surface[edge_split_mask]

                # Assign indices to the newly created vertices
                edge_vertex_indices                  = torch.full(edge_split_mask.shape, -1, dtype=torch.long, device=edge_split_mask.device)
                edge_vertex_indices[edge_split_mask] = (V.shape[0] + torch.arange(V_edges.shape[0], device=edge_vertex_indices.device))

                # Get vertex indices on the (potentially split edges)
                edge_vertex_indices_global = edge_vertex_indices[edges_inverse].reshape(3, -1)

                edge_split_mask_global = edge_split_mask[edges_inverse].reshape(3, -1)
                face_split_count = edge_split_mask_global.sum(dim=0)

                helper_idxs = torch.tensor([[1], [2], [3]], dtype=torch.long, device=face_split_count.device)
                split_edge_index_in_face, permutation = ((helper_idxs * edge_split_mask_global) - 1).sort(dim=0)

                edge_vertex_indices_in_face = torch.gather(edge_vertex_indices_global, 0, permutation)

                # Case of one split per face: get the index of the edge that is split
                mask_1 = face_split_count == 1
                mask_2 = face_split_count == 2
                mask_3 = face_split_count == 3

                # This is not a typo: edge indices are sorted ascending
                ve1 = edge_vertex_indices_in_face[-1]
                ve2 = edge_vertex_indices_in_face[-2]
                ve3 = edge_vertex_indices_in_face[-3]

                e1 = split_edge_index_in_face[-1]
                e2 = split_edge_index_in_face[-2] 
                e3 = split_edge_index_in_face[-3]
                vi = torch.gather(faces_current, 1, ((e1 + 0) % 3).unsqueeze(1))[:, 0]
                vj = torch.gather(faces_current, 1, ((e1 + 1) % 3).unsqueeze(1))[:, 0]
                vk = torch.gather(faces_current, 1, ((e1 + 2) % 3).unsqueeze(1))[:, 0]

                mask_2_d1 = mask_2 & ((e1 - e2) == 1)
                mask_2_d2 = mask_2 & ((e1 - e2) == 2)

                V = torch.cat([
                    V,
                    V_edges
                ])

                uvs = torch.cat([
                    uvs,
                    edge_midpoints_uv[edge_split_mask]
                ])

                tus = torch.cat([
                    tus,
                    tu_edge[edge_split_mask]
                ])

                tvs = torch.cat([
                    tvs,
                    tv_edge[edge_split_mask]
                ])

                faces_new = torch.cat([
                    # Faces with one subdivided edge
                    torch.stack([ve1[mask_1], vk[mask_1], vi[mask_1]], dim=-1),
                    torch.stack([ve1[mask_1], vj[mask_1], vk[mask_1]], dim=-1),
                    # # Faces with two subdivided edge
                    torch.stack([vi[mask_2],    ve1[mask_2],    ve2[mask_2]], dim=-1),
                    torch.stack([vj[mask_2],    ve2[mask_2],    ve1[mask_2]], dim=-1),
                    torch.stack([vk[mask_2_d1], ve2[mask_2_d1], vj[mask_2_d1]], dim=-1),
                    torch.stack([ve2[mask_2_d2], vk[mask_2_d2], vi[mask_2_d2]], dim=-1),
                    # Faces with three subdivided edges
                    torch.stack([vi[mask_3], ve1[mask_3], ve2[mask_3]], dim=-1),
                    torch.stack([ve2[mask_3], ve1[mask_3], ve3[mask_3]], dim=-1),
                    torch.stack([vk[mask_3], ve2[mask_3], ve3[mask_3]], dim=-1),
                    torch.stack([ve1[mask_3], vj[mask_3], ve3[mask_3]], dim=-1),
                ])

                faces         = torch.cat([faces, faces_current[face_split_count == 0]])
                faces_current = faces_new

            # Add remaining faces (those not subdivided after the stopping criterion)
            if faces_current.shape[0] > 0:
                faces = torch.cat([faces, faces_current])

            f_all  += [ faces + num_vertices if not no_face_offset else faces ]
            V_all  += [ V ]
            tu_all += [ tus ]
            tv_all += [ tvs ]
            uv_all += [ uvs ]

            num_vertices += V.shape[0]

        V  = torch.cat(V_all)  if not return_per_patch else V_all
        f  = torch.cat(f_all)  if not return_per_patch else f_all
        tu = torch.cat(tu_all) if not return_per_patch else tu_all
        tv = torch.cat(tv_all) if not return_per_patch else tv_all
        uv = torch.cat(uv_all) if not return_per_patch else uv_all

        results = [ V, f, tu, tv ]

        if return_uv:
            results += [ uv ]

        return *results,

def extrude(positions, t, n, b, shape, size=None, open_profile_curve=False, caps=False):
    """ Extrude a discrete curve into a triangle mesh

    Args:
        positions: Discrete positions on the curve         ((B,)N,3)
        frames: Frames at the discrete positions (t, n, b) ((B,)N,3,3)
        shape: Discrete points describing the extrusion shape in normal-space in counterclockwise order [Sx2]
        size (optional): Size of the extrusion operation [N]
        open_profile_curve: Indicator if the profile curve should be considered open
    """

    is_batched = len(positions.shape) == 3
    if not is_batched:
        positions = positions.unsqueeze(0)
        t         = t.unsqueeze(0)
        n         = n.unsqueeze(0)
        b         = b.unsqueeze(0)

    B, num_samples = positions.shape[:2]
    shape_size     = len(shape)

    if size == None:
        size = 1
    else:
        size = size[None, :, None, None]

    # Generate the vertices
    vertices = positions[:, :, None, :] + size * n[:, :, None, :] * shape[None, None, :, 0:1] + size * b[:, :, None, :] * shape[None, None, :, 1:2]
    vertices = vertices.view(B,-1, 3)

    # Generate the faces for each segment
    i  = torch.arange(num_samples-1, device=positions.device)[:, None]
    ii = i+1
    j  = torch.arange(shape_size, device=positions.device)[None, :]
    jj = (j+1) % shape_size

    if open_profile_curve:
        # Remove the last segment
        j  = j[:, :-1]
        jj = jj[:, :-1]

    faces = torch.stack([
         i * shape_size + j, 
        ii * shape_size + j, 
        ii * shape_size + jj,

         i * shape_size + j, 
        ii * shape_size + jj, 
         i * shape_size + jj
    ], dim=-1).view(-1, 3)

    if not open_profile_curve and caps:
        # FIXME: Implement
        # Add start and end caps to the curve
        vertices = torch.cat([vertices, positions[:1], positions[-1:]], dim=0)

    if not is_batched:
        vertices = vertices[0]

    return vertices, faces

class CurveTessellator:
    def __init__(self, u_resolution):
        self.u_resolution = u_resolution

    def tessellate(self, curves: ParametricCurves, profile: torch.Tensor, components: Optional[torch.Tensor] = None, t_prev: Optional[torch.Tensor] = None, n_prev: Optional[torch.Tensor] = None, b_prev: Optional[torch.Tensor] = None):
        """ Tessellate a set of parametric curves
        
        Args:
            curves: Set of parametric curves
            profile: The profile curve with shape                            (P,2)
            components: Ordered indices of (connected) components with shape (C,M)
            t_prev: Tangent of the previous frame ((C,),U,3)
            n_prev: Normal of the previous frame ((C,),U,3)
            b_prev: Binormal of the previous frame ((C,),U,3)
        """

        device = curves.V.device

        epsilon = 1e-4
        u       = torch.linspace(0, 1-epsilon, self.u_resolution, device=device)

        use_components = components is not None
        if not use_components:
            components = torch.arange(curves.F.shape[0], dtype=torch.long, device=curves.F.device)[None, :]

        C, M = components.shape[0:2]
        N    = curves.F.shape[1]

        # Group curves by components 
        # P has shape (C,M,N,3) where N is the number of control points and D the dimension
        P = curves.V[curves.F[components]]

        # p has shape (C*M)xUx3
        p, v, a = evaluate_bezier_curve(P.reshape(-1, N, 3), u)

        # Group evaluation results by components
        p = p.reshape(C, -1, 3)
        v = v.reshape(C, -1, 3)
        a = a.reshape(C, -1, 3)

        if t_prev is not None:
            assert n_prev is not None
            assert b_prev is not None
            t, n, b = recompute_frames(p, v, a, t_prev, n_prev, b_prev)
        else:
            t, n, b = compute_frames_double_reflection(p, v, a)
            
        v, f = extrude(p, t, n, b, profile)
        
        if not use_components:
            v = v[0]

        return v, f, t, n, b