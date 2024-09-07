import numpy as np


def rotate_x(theta):
    """
    Generate a rotation matrix along the x-axis

    :param theta: rotation angle, in radians
    :return: rotation matrix
    """

    return np.array([
        [1.0, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)],
    ])


def rotate_y(theta):
    """
    Generate a rotation matrix along the y-axis

    :param theta: rotation angle, in radians
    :return: rotation matrix
    """

    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1.0, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])


def rotate_z(theta):
    """
    Generate a rotation matrix along the z-axis

    :param theta: rotation angle, in radians
    :return: rotation matrix
    """

    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1.0],
    ])


def rotate(theta, degrees=True, axis='x', homogeneous=True):
    """
    Rotate along the specified axis

    :param theta: rotation angle. Default in degrees, but can be expressed in radians
    :param degrees: whether to interpret the angle as being expressed in degrees (True, default) or in radians (default)
    :param axis: axis around which to rotate. Accepted values: 'x', 'y', 'z'. Default to 'x'
    :param homogeneous: whether to return the rotation matrix with homogeneous coordinates
    :return: rotation matrix
    """

    assert axis in ['x', 'y', 'z'], 'axis should be: x, y, or z'

    if degrees:
        theta = np.deg2rad(theta)

    if axis == 'x':
        rotation_matrix = rotate_x(theta)
    elif axis == 'y':
        rotation_matrix = rotate_y(theta)
    elif axis == 'z':
        rotation_matrix = rotate_z(theta)
    else:
        # Should never happen
        rotation_matrix = np.eye(3)

    if homogeneous:
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :3] = rotation_matrix
        rotation_matrix = homogeneous_matrix

    return rotation_matrix
