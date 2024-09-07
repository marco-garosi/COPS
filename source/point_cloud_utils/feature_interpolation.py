import copy
import torch
import numpy as np
import open3d as o3d
import torch.nn.functional as F
from torch_geometric.nn import knn


def interpolate_point_cloud(pcd, features, points_with_missing_features=None, neighbors=10, copy_features=False, zero_nan=True):
    """
    Interpolate features associate to each point for points that are missing them.
    A point misses a feature if all the feature vector associated to it is made up
    of NaN values.
    Features are therefore interpolated using neighboring points.

    Note that if all the closest neighbors of a point are missing features as well,
    then the feature vector will still be composed only of NaN values. The `zero_nan` parameter allows to control
    this behavior: setting it to True (which is the default value) will substitute all NaN values with 0.

    :param pcd: the point cloud tensor, with shape (#points, 3)
    :param features: the features. Each element is a tensor representing the features associated to the
        point with the same index in `pcd`
    :param points_with_missing_features: indices of points which are missing features. If None, it will be
        automatically determined by finding all the feature vectors which have all features (along feature
        dimension) set to 0.0. Default to None
    :param neighbors: how many neighbors to consider in the interpolation
    :param copy_features: whether to copy the feature tensor or modify directly the one being passed
    :param zero_nan: if a feature vector is still NaN after interpolation (because all its neighbors are NaN, too),
        set it to 0 anyway. True: set NaN to 0 after interpolation. Defaults to True
    :return: the interpolated tensor
    """

    if copy_features:
        features = copy.deepcopy(features)

    if points_with_missing_features is None:
        points_with_missing_features = torch.all(features == 0, dim=-1).nonzero().view(-1)

    if len(points_with_missing_features) == 0:
        return features

    # k-NN between all the points
    neighbors += 1
    knn_on_cluster_assignment = knn(pcd, pcd, neighbors)

    # Get the features only for the points that will have to be used to interpolate feature
    # vectors, and then compute the average between all the neighbors for each point
    knn_on_cluster_assignment = knn_on_cluster_assignment[1].view(len(pcd), neighbors)
    neighbors_features = features[knn_on_cluster_assignment[points_with_missing_features].view(-1)].view(
        len(points_with_missing_features), neighbors, -1
    )
    # Setting to NaN makes it possible to compute the average only of those points which actually
    # have features. If this was left to 0, they would influence the mean, while this makes
    # it possible not to take them into account
    neighbors_features[torch.all(neighbors_features == 0, dim=-1)] = float('nan')
    features[points_with_missing_features] = neighbors_features.nanmean(dim=1)

    # Adjust feature vectors for points whose neighbors are all NaN (no feature vector assigned to them)
    if zero_nan:
        features = torch.nan_to_num(features)

    return features


def interpolate_feature_map(features, width, height, mode='bicubic'):
    """
    Interpolate a patchy feature map to the specified size (width, height)

    :param features: tensor of shape (batch_size, #patches + 1 for [CLS], embedding_dimension)
    :param width: width of the output
    :param height: height of the output
    :param mode: interpolation method. Default to 'bicubic'
    :return: interpolated feature map
    """

    if isinstance(features, dict) and 'last_hidden_state' in features.keys():
        features = features.last_hidden_state
    features = features[:, 1:, :]

    # R: renders
    # L: length
    R, L, _ = features.shape
    W = H = np.sqrt(L).astype(int)

    with torch.no_grad():
        interpolated_features = F.interpolate(
            features.view(R, W, H, -1).permute(3, 0, 1, 2),
            size=(width, height),
            mode=mode,
            align_corners=False if mode not in ['nearest', 'area'] else None,
        )
    interpolated_features = interpolated_features.permute(1, 2, 3, 0)

    return interpolated_features
