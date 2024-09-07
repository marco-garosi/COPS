import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_algorithm(matrix_parts_iou):
    """
    Computes the Hungarian algorithm.
    
    params: matrix_parts_iou: IoU matrix over the parts
    
    :return: matrix_parts_iou: IoU matrix over the parts
    :return: ids_pred: predicted ids
    :return: ids_gt: ground truth ids
    
    """    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-matrix_parts_iou)
    
    # get the ids
    ids_pred = np.zeros(matrix_parts_iou.shape[1])
    ids_gt = np.zeros(matrix_parts_iou.shape[0])
    
    for i, j in zip(row_ind, col_ind):
        ids_pred[j] = i
        ids_gt[i] = j
        
    return matrix_parts_iou, ids_pred, ids_gt