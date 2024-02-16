import math
import nvdiffrast.torch as dr
import torch
from typing import List, Optional, Tuple, Union

from diffshadow import dot, map_points_to_light_space
from diffshadow.simple_renderer import SimpleRenderer, merge_meshes, Mesh, Camera, DirectionalLight, SpotLight, PointLight

from .environment_map import EnvironmentMap
from .spherical_harmonics import SphericalHarmonics

def render_ibl(context: dr.RasterizeGLContext, meshes: List[Mesh], lights: List[SphericalHarmonics], camera: Camera, resolution: Tuple[int, int], ambient=0.1, return_gbuffer=False):
    mesh = merge_meshes(meshes) if len(meshes) > 1 else meshes[0]

    # G-Buffer pass
    renderer = SimpleRenderer(context)
    gbuffer  = renderer._gbuffer_pass(mesh, camera, resolution=resolution)

    color = gbuffer['diffuse_albedo'] * ambient

    if color.shape[2] != 3:
        color = color.repeat(1, 1, 3)

    for light in lights:
        color += light.eval(gbuffer['normal']) * gbuffer['diffuse_albedo'] / math.pi

    color = dr.antialias((color * gbuffer['mask'][:, :, None])[None], gbuffer['rast_out'], gbuffer['v_clipspace'], mesh.faces)[0]

    if not return_gbuffer:
        return color
    else:
        return color, gbuffer
    
def render_reflection(context: dr.RasterizeGLContext, mesh: Mesh, envmap: EnvironmentMap, camera: Camera, resolution: Tuple[int, int]):
    background = 0.01

    renderer = SimpleRenderer(context)
    gbuffer  = renderer._gbuffer_pass(mesh, camera, resolution=resolution)

    normal = gbuffer['normal']

    # Compute the reflection vector
    R      = camera.view_matrix[:3, :3]
    t      = camera.view_matrix[:3,  3]
    center = -R.T @ t
    V      = torch.nn.functional.normalize(center - gbuffer['position'], p=2, dim=-1)

    R = 2*gbuffer['normal']*dot(gbuffer['normal'], V, keepdim=True) - V

    mask = gbuffer['mask'][:, :, None]
    img  = torch.where(mask > 0, envmap.eval(R), background)
    img  = dr.antialias(img[None], gbuffer['rast_out'], gbuffer['v_clipspace'], gbuffer['f'])[0]

    return img, normal, mask

def render_facet(context: dr.RasterizeGLContext, mesh: Mesh, camera: Camera, resolution: Tuple[int, int], 
                 light: Optional[Union[PointLight, DirectionalLight, SpotLight]] = None, sh: Optional[SphericalHarmonics] = None, envmap: Optional[EnvironmentMap] = None):
    device = mesh.vertices.device

    # G-Buffer pass
    renderer = SimpleRenderer(context)
    gbuffer  = renderer._gbuffer_pass(context, mesh, camera, resolution=resolution)

    # Flat/facetted normals
    rast_out     = gbuffer['rast_out']
    bu, bv, zw, tid = rast_out.unbind(-1)
    bw = 1 - bu - bv
    bu = torch.where(rast_out[:, :, :, -1] > 0, (bu+0.001)**(1/2), 0)
    bv = torch.where(rast_out[:, :, :, -1] > 0, (bv+0.001)**(1/2), 0)
    bw = torch.where(rast_out[:, :, :, -1] > 0, (bw+0.001)**(1/2), 0)
    weight = (bu + bv + bw)
    bu = bu / weight
    bv = bv / weight
    rast_out_flat = torch.stack([bu, bv, zw, tid], dim=-1)
    n_worldspace = (mesh.normals @ mesh.transform_inv_transposed.T[:3, :3])
    normals_flat, _ = dr.interpolate(n_worldspace[None], rast_out_flat, mesh.faces)
    normals_flat = normals_flat[0]
    normals_flat = torch.nn.functional.normalize(normals_flat, p=2, dim=-1)

    # Ambient term
    color = gbuffer['diffuse_albedo']*0.01

    # Diffuse + Specular
    if light is not None:
        light_ = renderer._create_core_light_source(light, device=device)
        light_to_point, L, uv = map_points_to_light_space(light_, gbuffer['position'][None])
        light_to_point = light_to_point[0]
        L              = L[0]

        # Point lights have radial fall-off 
        falloff   = 1.0 if not isinstance(light, PointLight) else 1/light_to_point.norm(dim=-1, keepdim=True)**2

        # Intensity is the "color" of the light; can be spatially varying
        intensity = renderer._get_light_intensity(light, light_, uv)

        NdotL      = dot(normals_flat, L, keepdims=True)

        # Diffuse component
        color += gbuffer['diffuse_albedo'] / math.pi * intensity * NdotL.clamp(min=0, max=1) * falloff

        # Specular component (Blinn-Phong)
        R      = camera.view_matrix[:3, :3]
        t      = camera.view_matrix[:3,  3]
        center = -R.T @ t
        V      = torch.nn.functional.normalize(center - gbuffer['position'], p=2, dim=-1)
        H      = torch.nn.functional.normalize(L + V, dim=-1)
        specAngle = torch.clamp(dot(H, normals_flat), min=1e-10)
        specular = pow(specAngle, 16)
        color += specular * NdotL.clamp(min=0, max=1)

    # Diffuse image-based lighting
    if sh is not None:
        color += sh.eval(normals_flat) * gbuffer['diffuse_albedo'] / math.pi

    # Reflection mapping
    if envmap is not None:
        R      = camera.view_matrix[:3, :3]
        t      = camera.view_matrix[:3,  3]
        center = -R.T @ t
        V      = torch.nn.functional.normalize(center - gbuffer['position'], p=2, dim=-1)
    
        R = 2*normals_flat*dot(normals_flat, V) - V

        color = envmap.eval(R)

    color = torch.where(gbuffer['mask'][:, :, None] > 0, color, torch.tensor(0.8, device=device))
    color = dr.antialias(color[None], gbuffer['rast_out'], gbuffer['v_clipspace'], mesh.faces)[0]

    return color

def render_silhouette(context: dr.RasterizeGLContext, mesh: Mesh, camera: Camera, resolution: Tuple[int, int], spp: int):
    renderer = SimpleRenderer(context)
    gbuffer = renderer._gbuffer_pass(mesh, camera, resolution=(spp*resolution[0], spp*resolution[1]))

    mask = gbuffer['mask'][:, :, None].to(dtype=torch.float32)
    mask = dr.antialias(mask[None], gbuffer['rast_out'], gbuffer['v_clipspace'], gbuffer['f'])[0]

    silhouette = 1 - mask
    if spp > 1:
        silhouette_nhwc = silhouette[None].permute(0, 3, 1, 2)
        silhouette_nhwc = torch.nn.functional.avg_pool2d(silhouette_nhwc, spp)
        silhouette = silhouette_nhwc.permute(0, 2, 3, 1)[0].contiguous()

    return silhouette