import torch
from point_cloud_utils.backprojection import backproject, backproject_on_existing_tensor
from point_cloud_utils.feature_interpolation import interpolate_point_cloud


def aggregate_features(features, mappings, point_cloud, interpolate_missing=True, average=True, device='cpu'):
    """
    Aggregate features from different views for each point in the point cloud

    :param features: features from a backbone 2D model. Shape: (#views, width, height, embedding_dimensionality)
        #views is the same as len(mappings), as each feature map comes with its corresponding mapping
    :param mappings: tensor mapping from pixels to a point in `point_cloud`. Each "pixel"
        in `mappings` provides the index to a point in `point_cloud`, or -1 if it does not
        map to anything. Shape is (#views, width, height)
    :param point_cloud: tensor of shape (#points, 3) or (#points, 6)
    :param interpolate_missing: whether to interpolate missing features for points after aggregation.
        For more control, set this to False and call `interpolate_point_cloud` instead, passing all
        the desired parameters. Default to True
    :param average: whether to average the features collected for each point or not. Default to True
    :param device: where to perform the computation. Default to 'cpu'
    :return: the aggregated features, one feature vector per point in `point_cloud`
    """

    # Instead of stacking point-associated features and then using torch.nan_mean() on them,
    # this code aggregates them progressively to avoid consuming too much VRAM, which causes
    # the GPU to run out of memory with very large point clouds (>110_000 points).
    # This code can handle efficiently even larger points clouds (tested up to >230_000 points).
    feature_pcd_aggregated = torch.zeros((len(point_cloud), features.shape[-1]), device=device, dtype=torch.double)
    count = torch.zeros(len(feature_pcd_aggregated), device=device)

    for features_from_view, mapping in zip(features, mappings):
        feature_pcd = backproject(mapping, point_cloud, features_from_view, device=device)
        nan_mask = ~torch.all(feature_pcd == 0.0, dim=-1)
        feature_pcd_aggregated[nan_mask] += feature_pcd[nan_mask]
        count[nan_mask] += 1

    # Avoid division by zero
    count[count == 0] = 1

    # Compute the mean
    if average:
        feature_pcd_aggregated /= count.unsqueeze(-1)

    # Interpolate features
    if interpolate_missing:
        feature_pcd_aggregated = interpolate_point_cloud(point_cloud[:, :3], feature_pcd_aggregated, neighbors=20)

    return feature_pcd_aggregated
