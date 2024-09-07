import torch


def preprocess_ground_truth(ground_truth):
    """
    Ensure the part ids of the ground truth are in the [0, num_parts - 1] range.
    Some datasets may use arbitrary part numbers, so they may not be in the correct range,
    thus breaking metrics.

    :param ground_truth: tensor of ground truth segmentations, with shape (#points), where each element
        corresponds to the part id of the corresponding point in the point cloud
    :return: remapped part ids to the range [0, num_parts - 1], and the mapping utilized
    """

    part_id2cluster_id = {gt_id.item(): idx for idx, gt_id in enumerate(torch.unique(ground_truth))}

    return torch.tensor([
        part_id2cluster_id[seg.item()]
        for seg in ground_truth
    ], device=ground_truth.device), part_id2cluster_id
