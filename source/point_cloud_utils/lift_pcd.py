import numpy as np
import torch


def lift_pcd(
        depth: torch.Tensor,
        camera: torch.Tensor,
        xy_idxs=None,
        invert_axes=[False, True, True]
):
    """
    Given a depth image and relative camera, lifts the depth to a point cloud.
    If depth has 4 channels, the last 3 are used as RGB and an RGB point cloud is produced in output.
    Image size is implicitly given as depth image size.
    Optionally a set of xy coordinates can be passed to lift only these points.

    :param depth: the depth map image
    :param camera: the intrinsic parameters of the camera
    :param xy_idxs: set of (x, y) coordinates to lift only those points
    :param invert_axes: list of 3 Booleans representing whether the corresponding axis
        (x, y, z respectively) should be inverted
    """

    H, W, D = depth.shape

    d = depth[:, :, 0]

    if xy_idxs is not None:

        xmap = xy_idxs[0].to(d.device)
        ymap = xy_idxs[1].to(d.device)

        pt2 = d[ymap, xmap]
        xmap = xmap.to(torch.float32)
        ymap = ymap.to(torch.float32)

    else:

        # make coordinate grid
        xs = torch.linspace(0, W - 1, steps=W)
        ys = torch.linspace(0, H - 1, steps=H)

        # modify to be compatible with PyTorch 1.8
        xmap, ymap = np.meshgrid(xs.numpy(), ys.numpy(), indexing='xy')

        xmap = torch.tensor(xmap)
        ymap = torch.tensor(ymap)

        xmap = xmap.flatten().to(d.device).to(torch.float32)
        ymap = ymap.flatten().to(d.device).to(torch.float32)
        pt2 = d.flatten()

    # get camera info
    fx = camera[0]
    fy = camera[4]
    cx = camera[2]
    cy = camera[5]

    # perform lifting
    pt0 = (xmap - cx) * pt2 / fx
    pt1 = (ymap - cy) * pt2 / fy

    pcd_depth = torch.stack((pt0, pt1, pt2), dim=1)

    if D > 1:
        feats = depth[ymap.long(), xmap.long(), 1:]
        if xy_idxs is None:
            feats = feats.reshape(H * W, D - 1)
        pcd_depth = torch.cat([pcd_depth, feats], dim=1)

    for axis in range(pcd_depth.shape[-1]):
        pcd_depth[:, axis] *= -1 if invert_axes[axis] else 1

    return pcd_depth
