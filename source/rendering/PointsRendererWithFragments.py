"""
This source code is adapted from PyTorch3D. Differently from the original
implementation, this code returns `fragments` from the `forward()` function.
Source code: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/renderer/points/renderer.html#PointsRenderer
"""

import torch
import torch.nn as nn


class PointsRendererWithFragments(nn.Module):
    """
    A class for rendering a batch of points. The class should
    be initialized with a rasterizer and compositor class which each have a forward
    function.

    The points are rendered with varying alpha (weights) values depending on
    the distance of the pixel center to the true point in the xy plane. The purpose
    of this is to soften the hard decision boundary, for differentiability.
    See Section 3.2 of "SynSin: End-to-end View Synthesis from a Single Image"
    (https://arxiv.org/pdf/1912.08804.pdf) for more details.
    """

    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def to(self, device):
        # Manually move to device rasterizer as the cameras
        # within the class are not of type nn.Module
        self.rasterizer = self.rasterizer.to(device)
        self.compositor = self.compositor.to(device)
        return self

    def forward(self, point_clouds, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images, fragments
