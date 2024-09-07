import torch
import torch.nn.functional as F
import numpy as np

from rendering.PointsRendererWithFragments import PointsRendererWithFragments
from point_cloud_utils.lift_pcd import lift_pcd
from point_cloud_utils.homogeneous_coordinates import *
from point_cloud_utils.renderer import *
from point_cloud_utils.point_cloud_reigstration import *
from rendering.realistic_projection import *

try:
    import pytorch3d
    from pytorch3d.structures import Pointclouds
    from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVOrthographicCameras,
        FoVPerspectiveCameras,
        OrthographicCameras,
        PointsRasterizationSettings,
        PointsRenderer,
        PulsarPointsRenderer,
        PointsRasterizer,
        AlphaCompositor,
        NormWeightedCompositor
    )

    pytorch3d_available = True
except:
    pytorch3d_available = False


def render_and_map_pyrender(
        point_cloud,
        orientation,
        camera_type='intrinsic',
        canvas_width=600, canvas_height=600,
        fov=np.pi / 6.2,
        fx=1000, fy=1000,
        cx=600/2, cy=600/2,
        point_size=5.0,
        light_intensity=1.0,
        show_registration_result=False,
        device='cpu'
):
    """
    Render the given point cloud and map pixels in the render to
    points in the given point cloud.
    Mapping is by index, so for each pixel in the rendered image the mapping
    returns the index of the corresponding point in the point cloud.

    :param point_cloud: the point cloud to render, as a Nx3 or Nx6 array/tensor.
        If Nx6, first three values are treated as coordinates, last three values as
        RGB channels representing the color of the point.
    :param orientation: array/list/tuple representing x, y, z rotations respectively in degrees
    :param camera_type: string, either 'intrinsic', 'perspective', or 'orthographic'
    :param canvas_width: integer, in pixels
    :param canvas_height: integer, in pixels
    :param fov: float, field of view
    :param fx:
    :param fy:
    :param cx:
    :param cy:
    :param point_size: float, how large should points be when rendered. Increasing this value
        makes them look more as a continuous surface
    :param light_intensity: float, light intensity for the renderer
    :param show_registration_result: boolean, default to False, whether to show the result of
        registration in an interactive viewer
    :param device: string, default to 'cpu', where to store tensors for computations
    :return: the rendered point cloud, the corresponding depth map, and the mapping from
        pixels to points. The mapping sets to -1 all the pixels that do not correspond to anything
        in the point cloud. All positive values in mapping instead represent the index of the point
        in the point cloud that has been rendered to that location
    """

    # ============ #
    # Pre-processing
    # ============ #

    # Extract coordinates
    coordinates = point_cloud[:, :3]
    # Extract colors
    colors = point_cloud[:, 3:]

    # Make the point cloud an Open3D object
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.array(coordinates, copy=True))

    # Shift the point cloud to the origin
    source_pcd.translate(-source_pcd.get_center())

    # All points should be contained in [-1, 1]
    max_absolute_value = np.abs(np.array(source_pcd.points)).max()
    source_pcd.scale(1 / max_absolute_value, center=source_pcd.get_center())

    # Rotate
    rotation = source_pcd.get_rotation_matrix_from_xyz(np.deg2rad(np.array(orientation)))
    T = np.eye(4)
    T[:3, :3] = rotation
    source_pcd.transform(T)

    # Translate, so that the camera can actually see the object
    source_pcd.translate((0.0, 0.0, -5))

    # Render the image and get the depth map
    rendered_image, depth = renderer(
        source_pcd, colors=colors,
        camera_type=camera_type,
        CANVAS_WIDTH=canvas_width, CANVAS_HEIGHT=canvas_height,
        FOV=fov, FX=fx, FY=fy, CX=cx, CY=cy,
        point_size=point_size, light_intensity=light_intensity)

    # Get the camera matrix and lift the point cloud
    camera_matrix = torch.tensor(get_camera_matrix(fx, fy, cx, cy), device=device)
    depth_rgb = torch.tensor(depth).unsqueeze(-1).to(device)
    extracted_pcd = lift_pcd(depth_rgb, remove_homogeneous(camera_matrix).reshape(-1))

    # Keep only depths for pixel where the point cloud landed
    filtered_indices = ~torch.all(extracted_pcd == 0.0, dim=1)
    filtered_indices = filtered_indices.nonzero().squeeze()
    filtered_extracted_pcd = extracted_pcd[filtered_indices]

    # Make it an Open3D PointCloud
    lifted_pcd = o3d.geometry.PointCloud()
    lifted_pcd.points = o3d.utility.Vector3dVector(np.array(filtered_extracted_pcd.cpu().numpy(), copy=True))

    if show_registration_result:
        draw_registration_result(source_pcd, lifted_pcd, np.eye(4))

    # Compute the pixel-point mapping
    closest_points = map_points(lifted_pcd, source_pcd)
    mappings = -torch.ones(len(extracted_pcd), dtype=torch.int, device=device)
    mappings[filtered_indices] = torch.tensor(closest_points, device=device)
    mappings = mappings.view([canvas_width, canvas_height])

    # Return the data
    return rendered_image, depth, mappings

def render_and_map_pytorch3d(
        point_cloud,
        rotations, translations,
        canvas_width=600, canvas_height=600,
        point_size=0.007,
        points_per_pixel=10,
        perspective=True,
        device='cpu'
    ):
    # Prepare the point cloud
    verts = point_cloud[:, :3].to(device)
    verts -= verts.mean(dim=0)
    verts /= verts.norm(dim=-1).max()
    rgb = (point_cloud[:, 3:] / 255.).to(device)

    point_cloud_object = Pointclouds(points=[verts], features=[rgb])
    point_cloud_stacked = point_cloud_object.extend(len(rotations))

    # Prepare the cameras
    if perspective:
        cameras = FoVPerspectiveCameras(device=device, R=rotations, T=translations, znear=0.01)
    else:
        cameras = FoVOrthographicCameras(device=device, R=rotations, T=translations, znear=0.01)

    # Prepare the rasterizer
    raster_settings = PointsRasterizationSettings(
        image_size=(canvas_width, canvas_height),
        radius=point_size,
        points_per_pixel=points_per_pixel
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRendererWithFragments(
        rasterizer=rasterizer,
        compositor=NormWeightedCompositor(background_color=(1., 1., 1.))
    )

    # Get mappings and rendered images
    rendered_images, fragments = renderer(point_cloud_stacked)
    for idx in range(len(rotations)):
        fragments.idx[idx, fragments.idx[idx] != -1] -= (idx * len(verts))

    return (rendered_images[..., :3] * 255).int(), fragments.zbuf[..., 0], fragments.idx[..., 0]


def manual_projection(point_cloud, projector):
    img, is_seen, point_loc_in_img = projector.get_img(point_cloud)
    img = img[:, :, 20:204, 20:204]
    point_loc_in_img = torch.ceil((point_loc_in_img - 20) * 224. / 184.)
    img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=True)
    return img, is_seen, point_loc_in_img

def render_and_map_manual(
        point_cloud,
        device='cpu'
    ):
    projector = Realistic_Projection()
    with torch.no_grad():
        images, is_seen, point_loc_in_img = manual_projection(point_cloud[:, :3].unsqueeze(0), projector)

    # Get the mappings
    mappings = -1 * torch.ones((10, 224, 224), dtype=torch.int).to(device)
    point_in_view = torch.repeat_interleave(torch.arange(0, 10)[:, None], len(point_cloud)).view(-1, ).long().to(device)
    point_indices = torch.arange(0, len(point_cloud)).int().repeat(10).to(device)
    yy = point_loc_in_img[:, :, 0].view(-1).long()
    xx = point_loc_in_img[:, :, 1].view(-1).long()

    is_seen[is_seen == 0] = -1
    is_visible = is_seen.view(-1).int()
    mappings[point_in_view, xx, yy] = point_indices * is_visible
    mappings = torch.clamp(mappings, min=-1)

    return (images * 255).int(), (xx, yy, point_in_view, is_seen), mappings

def render_with_orientations(
        point_cloud,
        orientations=None,
        rotations=None, translations=None,
        backend='pyrender',
        camera_type='intrinsic',
        canvas_width=600, canvas_height=600,
        fov=np.pi / 6.2,
        fx=1000, fy=1000,
        cx=600 / 2, cy=600 / 2,
        point_size=5.0,
        light_intensity=1.0,
        show_registration_result=False,
        points_per_pixel=10,
        perspective=True,
        device='cpu'
    ):
    assert backend in ['pyrender', 'pytorch3d']

    if backend == 'pytorch3d':
        if not pytorch3d_available:
            raise RuntimeError('PyTorch3D not available')

        return render_and_map_pytorch3d(
            point_cloud, rotations, translations,
            canvas_width=canvas_width, canvas_height=canvas_height,
            point_size=point_size,
            points_per_pixel=points_per_pixel,
            perspective=perspective,
            device=device
        )

    elif backend == 'manual':
        return render_and_map_manual(point_cloud, device=device)

    elif backend == 'pyrender':
        rendered_images = np.empty((len(orientations), canvas_width, canvas_height, 3), dtype=int)
        depth_maps = np.empty((len(orientations), canvas_width, canvas_height), dtype=int)
        mappings = torch.empty((len(orientations), canvas_width, canvas_height), dtype=int).to(device)

        pcd_as_numpy = point_cloud.cpu().numpy()

        for idx, orientation in enumerate(orientations):
            rendered_image, depth_map, mapping = render_and_map_pyrender(
                pcd_as_numpy, orientation,
                camera_type=camera_type, canvas_width=canvas_width, canvas_height=canvas_height,
                fov=fov, fx=fx, fy=fy, cx=cx, cy=cy, point_size=point_size,
                light_intensity=light_intensity, show_registration_result=show_registration_result,
                device=device
            )

            rendered_images[idx] = rendered_image
            depth_maps[idx] = depth_map
            mappings[idx] = mapping

        return torch.tensor(rendered_images, device=device), torch.tensor(depth_maps, device=device), mappings
