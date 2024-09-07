import numpy as np


def iou_matrix_over_parts(ids_pred, ids_gt):
    """
    Computes the IoU matrix over the parts.
    
    params: ids_pred: predicted ids
    params: ids_gt: ground truth ids
    
    :return: matrix_parts_iou: IoU matrix over the parts
    
    """
    
    unique_ids_pred = np.unique(ids_pred)
    unique_ids_gt = np.unique(ids_gt)

    matrix_parts_iou = np.zeros((len(unique_ids_gt)+1, len(unique_ids_pred)+1))

    for id_gt in unique_ids_gt:
        
        for id_pred in unique_ids_pred:
            
            intersection = np.sum(np.array((ids_pred == id_pred) & (ids_gt == id_gt)))
            union = np.sum(np.array((ids_pred == id_pred) | (ids_gt == id_gt)))
            
            iou = intersection / union
            
            matrix_parts_iou[id_gt, id_pred] = iou
            
    return matrix_parts_iou