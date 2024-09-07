import torch


def backproject(mapping, point_cloud, pixel_features, device='cpu'):
    """
    Back-project features to points in the point cloud using the given mapping.

    :param mapping: tensor with shape (CANVAS_HEIGHT, CANVAS_WIDTH)
    :param point_cloud: tensor, points in the point cloud, with shape (#points, 3)
    :param pixel_features: features extracted with a backbone model. Shape is
        (CANVAS_HEIGHT, CANVAS_WIDTH, feature_dimensionality)
    :param device: device on which to place tensors and perform operations.
        Default to `cpu`

    :return: a new tensor of shape (#points, feature_dimensionality) that associates
        to each point in the point cloud a feature vector. Feature vector is all 0
        if no feature is being associated to a point.
        It can be indexed as `point_cloud`, so the i-th feature vector is associated
        to the i-th point in `point_cloud`.
    """

    # Feature vector: (points, feature_dimension)
    # For example: for 10,000 points and embedding dimension of 768,
    # features will be a (10000, 768) tensor
    features = torch.zeros((len(point_cloud), pixel_features.shape[-1]), dtype=torch.float, device=device)

    # Get pixel coordinates of pixels on which the point cloud has been projected
    yx_coords_of_pcd = (mapping != -1).nonzero()

    # Map features to the points
    # Explanation: `mapping` is a (HEIGHT, WIDTH) map with the same dimensionality
    # of the render. Each entry is either `-1` (no point has been mapped to the corresponding pixel)
    # or the index of the point in `point_cloud` that was mapped/projected to the corresponding
    # pixel.
    # So: `mapping != -1` returns a boolean mask to tell if a pixel is "empty" or if something
    # was projected there. Then, `mapping[mapping != -1]` returns the indices of the points
    # in `point_cloud` that have been mapped to a point. They're returned from top-left to
    # bottom-right order. Since `features` has a "row" for each point in `point_cloud`,
    # it can be indexed via the same indices as `point_cloud`. Therefore, `features[mapping[mapping != -1]]`
    # accesses the features of all the points that have been rendered (are visible) in the rendering.
    # Lastly, the assignment simply assigns features to those points. Features come from an "image"
    # (where number of channels is arbitrary, depending on the backbone model).
    # Note that a point may be mapped to multiple pixels, especially if using a large enough `point_size`.
    # In this case, a point will be assigned just the "last" features: if `pixel_features` has
    # two distinct feature vectors (e.g. [1, 2] and [3, 4]) for the point (x, y), the point (x, y)
    # will be ultimately assigned features [3, 4]. While this may sound like a problem, it is actually
    # not in most practical applications: if a point is projected into multiple pixels, they are certainly
    # neighbouring pixels. Therefore, they most likely have very similar feature vectors: so overwriting
    # the features and just keeping the "last" that comes (usually the most bottom-right in the
    # `pixel_features` "image") is not an actual problem.
    features[mapping[mapping != -1]] = pixel_features[yx_coords_of_pcd[:, 0], yx_coords_of_pcd[:, 1]]

    return features


def backproject_on_existing_tensor(mapping, features, pixel_features):
    """
    Back-project features to points in the point cloud using the given mapping.

    :param mapping: tensor with shape (CANVAS_HEIGHT, CANVAS_WIDTH)
    :param features: feature vector to store features onto
    :param pixel_features: features extracted with a backbone model. Shape is
        (CANVAS_HEIGHT, CANVAS_WIDTH, feature_dimensionality)

    :return: indices of points that received a feature vector
    """

    # Get pixel coordinates of pixels on which the point cloud has been projected
    mask = (mapping != -1)
    yx_coords_of_pcd = mask.nonzero()
    points = mapping[mask]

    # Map features to the points
    features[points] += pixel_features[yx_coords_of_pcd[:, 0], yx_coords_of_pcd[:, 1]]

    return points
