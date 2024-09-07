import torch
import numpy as np
from sklearn.metrics.cluster import rand_score, adjusted_rand_score, adjusted_mutual_info_score, mutual_info_score
from metrics.preprocess_ground_truth import preprocess_ground_truth
from torchmetrics.functional import jaccard_index


def standard_mIoU_point_cloud_seg_mask(predicted_cluster, ground_truth_cluster):
    """
    Computes the mean intersection over union of the point cloud
    segmentation mask.

    :param predicted_cluster: 1D tensor with the predicted clusters
    :param ground_truth_cluster: 1D tensor with the target clusters
    :return: mIoU (torch.tensor): mean intersection over union
    """

    # Mask for each unique cluster
    mask = torch.unique(ground_truth_cluster)

    # Initialize the tensors to store the results
    mIoU = torch.zeros(mask.shape[0])

    # Compute the mean IoU for each cluster
    for i, m in enumerate(mask):
        intersection = torch.sum((predicted_cluster == m) & (ground_truth_cluster == m))
        union = torch.sum((predicted_cluster == m) | (ground_truth_cluster == m))
        mIoU[i] = intersection / union

    return torch.mean(mIoU)


def mask_invariant_mIoU_point_cloud_seg_mask(predicted_cluster, ground_truth_cluster):
    """
    Computes the mean intersection over union of the point cloud
    segmentation mask considering as target id the predominant id
    in the predicted cluster mask.

    :param predicted_cluster: 1D tensor with the predicted clusters
    :param ground_truth_cluster: 1D tensor with the target clusters
    :return: mIoU (torch.tensor): mean intersection over union
    """

    # Mask for each unique cluster
    mask = torch.unique(ground_truth_cluster)

    # Initialize the tensors to store the results
    mIoU = torch.zeros(mask.shape[0])

    # Compute the mean IoU for each cluster
    for i, m in enumerate(mask):
        # Get the most frequent id in the predicted cluster
        assigned_id = int(torch.mode(predicted_cluster[ground_truth_cluster == m]).values)

        # Compute the intersection and union
        intersection = torch.sum((predicted_cluster == assigned_id) & (ground_truth_cluster == m))
        union = torch.sum((predicted_cluster == m) | (ground_truth_cluster == m))
        mIoU[i] = intersection / union

    return torch.mean(mIoU)


def mask_invariant_std_MAE(predicted_cluster, ground_truth_cluster):
    """
    Given the mask of the clusters assigned, it computes the pair
    wise distance for each cluster with MAE. Then, for each cluster
    we compute the standard deviation of the distances.

    If the distances computed within each cluster has some discrepancy
    the std is higher. On the other hand, if the std is peaked over
    the mean. It means that the conformation of the mask highly
    correlate with the target mask.


    :param predicted_cluster: 1D tensor with the predicted clusters
    :param ground_truth_cluster: 1D tensor with the target clusters
    :return: mean clusters and std for each cluster (uncertainty of the prediction)
    """

    distances = torch.abs(predicted_cluster - ground_truth_cluster)

    # Mask for each unique cluster
    mask = torch.unique(ground_truth_cluster)

    # Initialize the tensors to store the results
    mean_c = torch.zeros(mask.shape[0])
    std_c = torch.zeros(mask.shape[0])

    # Compute the mean and std for each cluster
    for i, m in enumerate(mask):
        mean_c[i] = torch.mean(distances[ground_truth_cluster == m])
        std_c[i] = torch.std(distances[ground_truth_cluster == m])

    return mean_c.cpu().numpy().tolist(), std_c.cpu().numpy().tolist()


def mask_invariant_std_MSE(predicted_cluster, ground_truth_cluster):
    """
    Given the mask of the clusters assigned, it computes the pair
    wise distance for each cluster with MSE. Then, for each cluster
    we compute the standard deviation of the distances.

    If the distances computed within each cluster has some discrepancy
    the std is higher. On the other hand, if the std is peaked over
    the mean. It means that the conformation of the mask highly
    correlate with the target mask.


    :param predicted_cluster: 1D tensor with the predicted clusters
    :param ground_truth_cluster: 1D tensor with the target clusters
    :return: mean clusters and std for each cluster (uncertainty of the prediction)
    """

    distances = torch.pow(predicted_cluster - ground_truth_cluster, 2)

    # Mask for each unique cluster
    mask = torch.unique(ground_truth_cluster)

    # Initialize the tensors to store the results
    mean_c = torch.zeros(mask.shape[0])
    std_c = torch.zeros(mask.shape[0])

    # Compute the mean and std for each cluster
    for i, m in enumerate(mask):
        mean_c[i] = torch.mean(distances[ground_truth_cluster == m])
        std_c[i] = torch.std(distances[ground_truth_cluster == m])

    return mean_c.cpu().numpy().tolist(), std_c.cpu().numpy().tolist()


def calc_iou(pred, gt) -> float:
    """
    PartSLIP's implementation
    """

    I = np.logical_and(pred, gt).sum()
    U = np.logical_or(pred, gt).sum()

    if U == 0.:
        return 1.

    return I / U


def compute_miou_partslip(predicted_cluster, ground_truth_cluster, part_names):
    print(predicted_cluster.unique(), ground_truth_cluster.unique())
    cnt_iou = 0
    cnt = 0
    ground_truth_cluster = ground_truth_cluster.cpu().numpy()

    # Ground truth
    for i, part in enumerate(part_names):
        if (ground_truth_cluster == i).sum() == 0:
            continue
        # load predictions
        sem_pred = (predicted_cluster == i).cpu().numpy()  # True where predicted cluster is current part
        iou = calc_iou(sem_pred, ground_truth_cluster == i)
        cnt += 1
        cnt_iou += iou

    if cnt == 0:
        return 1.
    return cnt_iou / cnt


def mean_iou(pred, target, n_classes):
    pred = pred.unsqueeze(0)
    target = target.unsqueeze(0)
    segm_pred = torch.nn.functional.one_hot(pred, num_classes=n_classes).to(torch.float32)  # (bs, n_verts, n_classes)
    segm_target = torch.nn.functional.one_hot(target, num_classes=n_classes).to(torch.float32)  # (bs, n_verts, n_classes)
    inter = (segm_pred * segm_target).to(torch.float32).detach().cpu().numpy()
    union = (segm_pred + segm_target > 0).to(torch.float32).detach().cpu().numpy()
    ious_shapes = list()
    for batch in range(union.shape[0]):
        ious_parts = list()
        for part in range(union.shape[2]):
            if np.sum(union[batch, :, part]) == 0.0:
                iou = 1.0  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = np.sum(inter[batch, :, part]) / np.sum(union[batch, :, part])
            ious_parts.append(iou)
        ious_shapes.append(np.mean(ious_parts))
    ious_shapes = np.array(ious_shapes)
    return np.sum(ious_shapes)


def calculate_iou(ground, prediction, num_labels):
    """
    Computes IOU of point cloud
    :param ground: numpy array consisting of ground truth labels
    :param prediction: numpy array of predicted labels
    :param num_labels: int, total number of labels
    :return:
    """

    label_iou, intersection, union = {}, {}, {}
    # Ignore undetermined
    prediction = np.copy(prediction)
    prediction[ground == 0] = 0

    for i in range(1, num_labels):
        # Calculate intersection and union for ground truth and predicted labels
        intersection_i = np.sum((ground == i) & (prediction == i))
        union_i = np.sum((ground == i) | (prediction == i))

        # If label i is present either on the gt or the pred set
        if union_i > 0:
            intersection[i] = float(intersection_i)
            union[i] = float(union_i)
            label_iou[i] = intersection[i] / union[i]

    metrics = {"label_iou": label_iou, "intersection": intersection, "union": union}

    return metrics


def calculate_shape_iou(ious):
    """
    Computes average shape IOU
    :param ious: dictionary containing for each shape its label iou,
     intersection and union scores with respect to the ground truth.
    :return:
      aveg_shape_IOU: float
    """
    shape_iou = {}

    for model_name, metrics in ious.items():
        # Average label iou per shape
        L_s = len(metrics["label_iou"])
        shape_iou[model_name] = np.nan_to_num(np.sum([v for v in metrics["label_iou"].values()]) / float(L_s))

    # Dataset avg shape iou
    avg_shape_iou = np.sum([v for v in shape_iou.values()]) / float(len(ious))

    return avg_shape_iou


def calculate_part_iou(ious, num_labels):
    """
    Calculates part IOU
    :param ious: dictionary containing for each shape its label iou,
     intersection and union scores with respect to the ground truth.
    :param num_labels: int, total number of labels in the category
    :return:
      aveg_shape_IOU: float
    """
    intersection = {i: 0.0 for i in range(1, num_labels)}
    union = {i: 0.0 for i in range(1, num_labels)}

    for model_name, metrics in ious.items():
        for label in metrics["intersection"].keys():
            # Accumulate intersection and union for each label across all shapes
            intersection[label] += metrics["intersection"][label]
            union[label] += metrics["union"][label]

    # Calculate part IOU for each label
    part_iou = {}
    for key in range(1, num_labels):
        try:
            part_iou[key] = intersection[key] / union[key]
        except ZeroDivisionError:
            part_iou[key] = 0.0
    # Avg part IOU
    avg_part_iou = np.sum([v for v in part_iou.values()]) / float(num_labels - 1)

    return avg_part_iou


def compute_all_metrics(predicted_cluster, ground_truth_cluster, num_classes=None, device='cpu'):
    """
    Compute all the metrics for the point cloud segmentation task

    :param predicted_cluster: tensor of shape (#points) containing the predicted cluster ids for each point
    :param ground_truth_cluster: tensor of shape (#points) containing the ground truth cluster id for each point
    :param num_classes: number of classes. If None, the function tries to determine it automatically. Default to None.
    :param device: device to perform computations on. Default to 'cpu'
    :return: a dictionary containing all the metrics
    """

    # For the Jaccard index
    if num_classes is None:
        num_classes = max(2, len(set(torch.unique(predicted_cluster).tolist() + torch.unique(ground_truth_cluster).tolist())), predicted_cluster.max().item(), ground_truth_cluster.max().item())

    # Compute the metrics, build a dictionary collecting them, and return it
    return {
        'meanIoU': float(mean_iou(predicted_cluster.long(), ground_truth_cluster.long(), num_classes)),
        # 'jaccard_index_micro': jaccard_index(predicted_cluster, ground_truth_cluster, task='multiclass', num_classes=num_classes, average='micro').item(),
        'jaccard_index_macro': jaccard_index(predicted_cluster, ground_truth_cluster, task='multiclass', num_classes=num_classes, average='macro').item(),
        # 'jaccard_index': jaccard_index(predicted_cluster, ground_truth_cluster, task='multiclass', num_classes=num_classes, average='weighted').item(),
        'jaccard_index_per_part': jaccard_index(predicted_cluster, ground_truth_cluster, task='multiclass', num_classes=num_classes, average='none').cpu().tolist(),

        # 'rand_score': rand_score(ground_truth_cluster.cpu().numpy(), predicted_cluster.cpu().numpy()),
        # 'adjusted_rand_score': adjusted_rand_score(ground_truth_cluster.cpu().numpy(), predicted_cluster.cpu().numpy()),
        # 'mutual_info_score': mutual_info_score(ground_truth_cluster.cpu().numpy(), predicted_cluster.cpu().numpy()),
        # 'adjusted_mutual_info_score': adjusted_mutual_info_score(ground_truth_cluster.cpu().numpy(), predicted_cluster.cpu().numpy()),
        # 'mask_invariant_std_mae': mask_invariant_std_MAE(predicted_cluster.float(), ground_truth_cluster.float()),
        # 'mask_invariant_std_mse': mask_invariant_std_MSE(predicted_cluster.float(), ground_truth_cluster.float()),
    }
