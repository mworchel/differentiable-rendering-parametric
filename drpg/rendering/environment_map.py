import nvdiffrast.torch as dr
import torch

class EnvironmentMap:
    def __init__(self, envmap: torch.Tensor, is_cube_map: bool = False):
        self.envmap = envmap
        self.is_cube_map = is_cube_map

    def eval(self, rays: torch.Tensor):
        rays_array = rays.reshape(-1, 3)
        rays_array = torch.nn.functional.normalize(rays_array, dim=-1)
        
        if not self.is_cube_map:
            x, y, z = rays_array.unbind(dim=-1)
            theta   = torch.acos(y)
            phi     = torch.atan2(x, z)

            # NOTE: nvdiffrast uv space is
            #  --> u
            # |
            # v
            u = 1 - phi/(2*torch.pi)
            v = theta / torch.pi

            uv = torch.stack([u, v], dim=-1).reshape(rays.shape[0], rays.shape[1], 2)
            img = dr.texture(self.envmap[None], uv[None])[0]
        else:
            img = dr.texture(self.envmap[None], uv=rays[None], boundary_mode='cube')[0]

        return img