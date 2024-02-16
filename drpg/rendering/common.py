import torch
from typing import Tuple

from diffshadow.filtering import make_grid
from diffshadow.transformation import hnormalized, apply_transformation

def get_view_rays(view_matrix: torch.Tensor, projection_matrix: torch.Tensor, resolution: Tuple[int, int]) -> torch.Tensor:
    """ Get world space view rays for a camera 

    Args:
        view_matrix [4,4]: View matrix of the camera
        projection_matrix [4,4]: Projection matrix of the camera (NOTE: must be perspective projection)
        resolution: Size of the image (height, width)

    Returns:
        World space view rays [height,width,3]
    """

    h, w = resolution
    rays = torch.cat([make_grid([h, w], limits=(-1, 1)), torch.zeros((h, w, 1))], dim=-1).to(view_matrix.device)
    rays = hnormalized(apply_transformation(rays, torch.inverse(projection_matrix)))
    rays = torch.nn.functional.normalize(rays @ view_matrix[:3, :3], p=2, dim=-1)
    return rays