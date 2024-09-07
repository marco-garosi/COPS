import torch
from scipy.optimize import linear_sum_assignment


def hungarian_algorithm(matrix_parts_iou):
    """
    Computes the Hungarian algorithm.

    :param matrix_parts_iou: IoU matrix over the parts
    :return: predicted ids and ground truth ids
    """

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-matrix_parts_iou)

    # Get the IDs
    ids_pred = torch.zeros(matrix_parts_iou.shape[1])
    ids_gt = torch.zeros(matrix_parts_iou.shape[0])

    for i, j in zip(row_ind, col_ind):
        ids_pred[j] = i
        ids_gt[i] = j

    return matrix_parts_iou, ids_pred, ids_gt


def compute_iou_matrix_over_parts(ids_pred, ids_gt):
    """
    Compute the IoU matrix over the parts.

    :param ids_pred: predicted ids
    :param ids_gt: ground truth ids

    :return matrix_parts_iou: IoU matrix over the parts
    """

    unique_ids_pred = torch.unique(ids_pred)
    unique_ids_gt = torch.unique(ids_gt)

    matrix_parts_iou = torch.zeros((
        max(unique_ids_gt.max().item(), len(unique_ids_gt)) + 1,
        max(unique_ids_pred.max().item(), len(unique_ids_pred)) + 1
    ))

    for id_gt in unique_ids_gt:

        for id_pred in unique_ids_pred:
            intersection = torch.sum((ids_pred == id_pred) & (ids_gt == id_gt))
            union = torch.sum((ids_pred == id_pred) | (ids_gt == id_gt))

            iou = intersection / union

            matrix_parts_iou[id_gt, id_pred] = iou

    return matrix_parts_iou


def map_cluster_ids_to_gt(ids_pred, ids_gt):
    """
    Compute the mapping between the predicted ids and the ground truth ids.

    :param ids_pred: predicted ids
    :param ids_gt: ground truth ids

    :return ids_pred_h: mapping between the predicted ids and the ground truth ids
    :return ids_gt_h: mapping between the ground truth ids and the predicted ids

    """

    # Compute the IoU matrix over the parts
    matrix_parts_iou = compute_iou_matrix_over_parts(ids_pred, ids_gt)

    # ids_pred_h, ids_gt_h contains the correct assignment of the ids
    # between the ground truth and the predicted ids
    matrix_parts_iou, ids_pred_h, ids_gt_h = hungarian_algorithm(matrix_parts_iou)

    return matrix_parts_iou, ids_pred_h, ids_gt_h
