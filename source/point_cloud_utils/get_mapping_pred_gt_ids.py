
from point_cloud_utils.iou_matrix_over_parts import iou_matrix_over_parts
from point_cloud_utils.hungarian_algorithm import hungarian_algorithm


def get_mapping_ids_pred_gt(ids_pred, ids_gt):
    """
    Gets the mapping between the predicted ids and the ground truth ids.
    
    params: ids_pred: predicted ids
    params: ids_gt: ground truth ids
    
    :return: matrix_parts_iou: IoU matrix over the parts
    :return: ids_pred_h: mapping between the predicted ids and the ground truth ids
    :return: ids_gt_h: mapping between the ground truth ids and the predicted ids
        
    """
    # compute the IoU matrix over the parts
    matrix_parts_iou = iou_matrix_over_parts(ids_pred, ids_gt)

    # ids_pred_h, ids_gt_h contains the correct assignment of the ids
    # between the ground truth and the predicted ids
    matrix_parts_iou, ids_pred_h, ids_gt_h = hungarian_algorithm(matrix_parts_iou)
    
    return matrix_parts_iou, ids_pred_h, ids_gt_h