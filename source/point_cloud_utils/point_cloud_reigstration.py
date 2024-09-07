import copy
import open3d as o3d
import numpy as np


def draw_registration_result(source, target, transformation):
    """
    Draw the two point clouds in the same 3D space

    :param source: source point cloud (first point)
    :param target: target point cloud (second pcd)
    :param transformation: transformation
    :return:
    """

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def map_points(source, target):
    # Get the KDTree for the target
    target_pcd_tree = o3d.geometry.KDTreeFlann(target)

    # For each point in the source point cloud, make room to store the index
    # of target's closest point
    closest_points = np.empty(len(source.points), dtype=int)

    for source_idx, source_point in enumerate(source.points):
        [k, idx, _] = target_pcd_tree.search_knn_vector_3d(source_point, 1)
        closest_points[source_idx] = idx[0]

    return closest_points
