import numpy as np


def make_homogeneous(matrix):
    """
    Make the given matrix with homogeneous coordinates

    :param matrix: a 3x3 matrix to make homogeneous
    :return: the homogeneous matrix
    """

    assert matrix.shape == (3, 3)

    homogeneous = np.zeros((4, 4))
    homogeneous[:3, :3] = matrix
    homogeneous[-1, -1] = 1

    return homogeneous


def remove_homogeneous(matrix):
    """
    Remove homogenous coordinates and go back to a 3x3 matrix

    :param matrix: a 4x4 matrix to make 3x3 by removing the last dimension on both axes
    :return: the non-homogeneous matrix
    """

    assert matrix.shape == (4, 4)

    return matrix[:3, :3]
