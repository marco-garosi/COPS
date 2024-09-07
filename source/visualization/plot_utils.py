import config

import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, Dinov2Model

import numpy as np


import matplotlib.pyplot as plt
import PIL.Image as Image


from source.point_cloud_utils.point_cloud_reigstration import *
from source.point_cloud_utils.backprojection import backproject
from source.point_cloud_utils.kmeans import kmeans
from source.point_cloud_utils.feature_interpolation import *
from source.point_cloud_utils.feature_aggregation import *
from source.rendering.render_and_map import  render_with_orientations
from source.metrics.compute_metrics import mean_iou, rand_score, adjusted_rand_score, mutual_info_score, adjusted_mutual_info_score


from data.PartNet import PartNet
from data.PartNetMobility import PartNetMobility_Part
from data.ScanObjectNN import ScanObjectNN_Part, scanobjectnn_part_collate_fn


from scipy.optimize import linear_sum_assignment

import open3d as o3d


#######################################################################################################
#######################################################################################################


CANVAS_WIDTH, CANVAS_HEIGHT = 672, 672
FX = FY = 1000
CX = CANVAS_WIDTH / 2
CY = CANVAS_HEIGHT / 2

SPLIT = 'train'
BATCH_SIZE = 1
K_MEANS_CLUSTERS = 5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

orientations = np.array([
    [0.0, 0.0, 0.0],
    [-45.0, 0.0, -55.0],
    [0.0, 90.0, 0.0],
    [20.0, -70.0, 0.0],
    [30.0, 27.0, 10.0],
    [-20.0, 70.0, 30.0],
])


#######################################################################################################
#######################################################################################################

def preprocess_ground_truth(ground_truth):
    """
    Ensure the part ids of the ground truth are in the [0, num_parts - 1] range.
    Some datasets may use arbitrary part numbers, so they may not be in the correct range,
    thus breaking metrics.

    :param ground_truth: tensor of ground truth segmentations, with shape (#points), where each element
        corresponds to the part id of the corresponding point in the point cloud
    :return: remapped part ids to the range [0, num_parts - 1], and the mapping utilized
    """

    part_id2cluster_id = {gt_id: idx for idx, gt_id in enumerate(set(ground_truth.cpu().numpy()))}

    return torch.tensor([
        part_id2cluster_id[seg.item()]
        for seg in ground_truth
    ]), part_id2cluster_id
    

class SegMetrics:
    def __init__(self) -> None:
        pass

    def standard_mIoU_point_cloud_seg_mask(self, predicted_cluster, ground_truth_cluster):
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


    def mask_invariant_mIoU_point_cloud_seg_mask(self, predicted_cluster, ground_truth_cluster):
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


    def mask_invariant_std_MAE(self, predicted_cluster, ground_truth_cluster):
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


    def mask_invariant_std_MSE(self, predicted_cluster, ground_truth_cluster):
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


    def mean_iou(self, predicted_cluster, ground_truth_cluster, num_clusters, device='cpu'):
        """
        Compute the mean Intersection over Union of a point cloud segmentation

        :param predicted_cluster: tensor of shape (#points), where each entry represents the cluster id
            associated to the corresponding point
        :param ground_truth_cluster: tensor of shape (#points), where each entry represents the ground truth
            part id of the corresponding point
        :param num_clusters: how many distinct clusters there are
        :param device: where to perform computations on. Default to 'cpu'
        :return: the mean Intersection over Union metric
        """

        iou_sum = 0.0
        arange = torch.arange(num_clusters)[:, None, None].to(device)

        # Iterate over each cluster
        for cluster_id in range(num_clusters):
            # Mask for points in the cluster
            cluster_mask = (predicted_cluster == cluster_id).float()
            # Masks for each ground truth label
            label_masks = (ground_truth_cluster == arange).float()

            # Compute intersection
            intersection = (cluster_mask[None] * label_masks).sum(dim=(1, 2))
            # Compute union
            union = (cluster_mask + label_masks).clamp(max=1).sum(dim=(1, 2))

            # Compute the IoU for the cluster
            iou = intersection / union
            # Find the best IoU for this cluster
            iou_max, _ = iou.max(dim=0)
            iou_sum += iou_max.item()

        return iou_sum / num_clusters

    def compute_all_metrics(self, predicted_cluster, ground_truth_cluster, device='cpu'):
        """
        Compute all the metrics for the point cloud segmentation task

        :param predicted_cluster: tensor of shape (#points) containing the predicted cluster ids for each point
        :param ground_truth_cluster: tensor of shape (#points) containing the ground truth cluster id for each point
        :param device: device to perform computations on. Default to 'cpu'
        :return: a dictionary containing all the metrics
        """

        # Map the ground truth cluster IDs to the range [0, #clusters - 1]
        # See function documentation for more information
        ground_truth_cluster, mapping = preprocess_ground_truth(ground_truth_cluster)
        ground_truth_cluster = ground_truth_cluster.to(device)

        # Compute the metrics, build a dictionary collecting them, and return it
        return {
            'meanIoU': self.mean_iou(predicted_cluster, ground_truth_cluster, len(mapping), device=device),
            'rand_score': rand_score(ground_truth_cluster.cpu().numpy(), predicted_cluster.cpu().numpy()),
            'adjusted_rand_score': adjusted_rand_score(ground_truth_cluster.cpu().numpy(), predicted_cluster.cpu().numpy()),
            'mutual_info_score': mutual_info_score(ground_truth_cluster.cpu().numpy(), predicted_cluster.cpu().numpy()),
            'adjusted_mutual_info_score': adjusted_mutual_info_score(ground_truth_cluster.cpu().numpy(), predicted_cluster.cpu().numpy()),
            'mask_invariant_std_mae': self.mask_invariant_std_MAE(predicted_cluster.float(), ground_truth_cluster.float()),
            'mask_invariant_std_mse': self.mask_invariant_std_MSE(predicted_cluster.float(), ground_truth_cluster.float()),
        }
        

#######################################################################################################
#######################################################################################################

seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        
def calculate_shape_IoU(pred_np, seg_np, label, class_choice, eva=False):
    label = label.squeeze()
    shape_ious = []
    category = {}
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[int(label[shape_idx])]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[int(label[0])])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
        if label[shape_idx] not in category:
            category[int(label[shape_idx])] = [shape_ious[-1]]
        else:
            category[label[shape_idx]].append(shape_ious[-1])
    if eva:
        return shape_ious, category
    else:
        return shape_ious

#######################################################################################################
#######################################################################################################

def plot_images(rendered_images):
    """
    Plots the rendered images.
    
    params: rendered_images: list of rendered images
    :return: the figure numpy array
    """
    plt.figure(figsize=(10, 10))
    number_of_images = len(rendered_images)
    n_rows = int(np.sqrt(number_of_images))
    n_cols = int(np.ceil(number_of_images / n_rows))
    
    for row in range(n_rows):
        for col in range(n_cols):
            plt.subplot(n_rows, n_cols, row * n_cols + col + 1)
            plt.imshow(rendered_images[row * n_cols + col].cpu().numpy())
            plt.title('orientations' + str(orientations[row * n_cols + col]))
            plt.axis('off')
            
    plt.show()
    
    # get the figure as numpy array
    fig = plt.gcf()
    fig.canvas.draw()
    fig_np = np.array(fig.canvas.renderer._renderer)
    plt.close()
    
    return fig_np

#######################################################################################################
#######################################################################################################
    
def get_rendered_images(point_cloud):
    """
    Renders the point cloud with the given orientations.
    
    parmas: point_cloud: point cloud to render

    :return: rendered_images: list of rendered images
    :return: depth_maps: list of depth maps
    :return: mappings: list of mappings from point cloud to image
    
    """
    
    rendered_images, depth_maps, mappings = render_with_orientations(
        point_cloud, orientations,
        point_size=8.0, light_intensity=0.05,
        canvas_width=CANVAS_WIDTH, canvas_height=CANVAS_HEIGHT, cx=CX, cy=CY,
        device=device
    )
    
    return rendered_images, depth_maps, mappings

#######################################################################################################
#######################################################################################################

def load_the_model(name='facebook/dinov2-base', device='cuda', _eval=True):
    """
    Loads the model with the given name.
    
    params: name: name of the model to load
    
    :return: model: loaded model
    :return: image_processor: loaded image processor
    
    """
    
    image_processor = AutoImageProcessor.from_pretrained(name)
    model = Dinov2Model.from_pretrained(name)
    
    if _eval:
        model.eval()
        
    model.to(device)
    print('Model '+name+' loaded')
    
    return model, image_processor

#######################################################################################################
#######################################################################################################

def random_sampling(point_cloud, segmentation, num_samples=2048):
    """
    Randomly samples points from the point cloud.
    
    params: point_cloud: point cloud to sample from
    params: segmentation: segmentation of the point cloud
    params: num_samples: number of points to sample
    
    :return: sampled_point_cloud: sampled point cloud
    
    """
    mask = torch.randperm(point_cloud.shape[0])[:num_samples]
    sampled_point_cloud = point_cloud[mask]
    segmentation_mask = segmentation[mask]
    
    return sampled_point_cloud, segmentation_mask

#######################################################################################################
#######################################################################################################

def get_the_features(rendered_images, image_obj_in_context, model, image_processor, device='cpu'):
    """
    Gets the features of the rendered images.
    
    params: rendered_images: list of rendered images [n_images, 3, width, height]
    params: image_obj_in_context: image in context [1, 3, width, height]
    
    """

    inputs_1 = image_processor(
        rendered_images,
        return_tensors='pt'
    ).to(device)
    
    inputs_2 = image_processor(
        image_obj_in_context,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs_1 = model(**inputs_1)
        outputs_2 = model(**inputs_2)
        
    del inputs_1, inputs_2

    # [n_images, 768, width, height], [1, 768, width, height]
    return outputs_1, outputs_2 
   
#######################################################################################################
#######################################################################################################

def get_the_aggregated_features(outputs, mappings, point_cloud, CANVAS_HEIGHT=672, CANVAS_WIDTH=672, device='cpu'):
    """
    Gets the interpolated features of the rendered images.
    
    """
    
    # try:
    #     del final_features
    # except:
    #     pass

    final_features = interpolate_feature_map(outputs, CANVAS_WIDTH, CANVAS_HEIGHT)
    del outputs
    print(final_features.shape)
    
    feature_pcd_aggregated = aggregate_features(final_features.cpu(), mappings.cpu(), point_cloud, device=device)
    print(feature_pcd_aggregated.shape)
    del final_features
    
    return feature_pcd_aggregated

#######################################################################################################
#######################################################################################################

id2color = {i: np.random.randint(0, 255, 3) for i in range(10)}

def assign_rgb_colors_to_labels(labels):
    """
    Given a list of labels, assign a unique color to each label.
    
        param labels: a list of labels
        return: a numpy array of shape (#labels, 3) containing the RGB colors assigned to each label
        
    """
    unique_labels = np.unique(labels)
    dict_labels = {label: i for i, label in enumerate(unique_labels)}
    
    
    # assign the colors to each label
    rgb_colors = np.zeros((len(labels), 3))
    for i, label in enumerate(labels):
        rgb_colors[i] = id2color[dict_labels[int(label)]]
     
    return rgb_colors

#######################################################################################################
#######################################################################################################

def visualize_with_open3d(xyz_rgb):
    """
    Visualizes a point cloud with open3d.
    
        param xyz_rgb: a numpy array of shape (#points, 6), where 6 is due to (x, y, z, R, G, B)
        :return: None
                
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz_rgb[:, :3])
    point_cloud.colors = o3d.utility.Vector3dVector(xyz_rgb[:, 3:]/255.0)
    o3d.visualization.draw_geometries([point_cloud])
    
#######################################################################################################
#######################################################################################################

def plot_point_cloud(point_cloud, title='', axis=False, show=True):
    """
    Visualizes a point cloud with matplotlib.

        param point_cloud: a numpy array of shape (#points, 6), where 6 is due to (x, y, z, R, G, B)
        param title: the title of the plot
        param axis: whether to show the axis or not
        :return: None
        
    """
    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    
    # rotate the point cloud to be able to see it from 90 degrees x axis and 30 degrees z axis and 90 degrees y axis
    ax.view_init(30, 0, 90)
    
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=12.0, c=point_cloud[:, 3:]/255.0)
    if axis:
        ax.set(xlabel='X', ylabel='Y', zlabel='Z')
    ax.set_title(title)#, fontsize=23)
    if show:
        plt.show()
    
    # get the figure as numpy array dpi=300
    fig = plt.gcf()
    fig.canvas.draw()
    fig_np = np.array(fig.canvas.renderer._renderer)
    plt.close()
    
    return fig_np

#######################################################################################################
#######################################################################################################

def get_render_clustering(segmentation_id, point_cloud, title):
    """
    Gets the cluster assignment of the rendered images.

    params: segmentation_id: id of the segmentation to load
    params: point_cloud: point cloud to render
    params: title: title of the plot
    
    :return: render_clustering: plot of the point cloud with the cluster assignment
    
    """

    # get the rgb colors for each label
    rgb_colors = assign_rgb_colors_to_labels(segmentation_id)

    # visualize the point cloud
    xyz_rgb = np.concatenate([point_cloud[:, :3], rgb_colors[:point_cloud.shape[0]]], axis=-1)
    
    # visualize_with_open3d(xyz_rgb)
    render_clustering = plot_point_cloud(xyz_rgb, title=title, axis=True, show=False)
    
    return render_clustering

#######################################################################################################
#######################################################################################################

def compute_iou_matrix_over_parts(ids_pred, ids_gt):
    """
    Computes the IoU matrix over the parts.
    
    params: ids_pred: predicted ids
    params: ids_gt: ground truth ids
    
    :return: matrix_parts_iou: IoU matrix over the parts
    
    """
    
    unique_ids_pred = np.unique(ids_pred)
    unique_ids_gt = np.unique(ids_gt)

    # unique_ids_pred, unique_ids_gt
    matrix_parts_iou = np.zeros((len(unique_ids_gt)+1, len(unique_ids_pred)+1))

    for id_gt in unique_ids_gt:
        
        for id_pred in unique_ids_pred:
            
            intersection = np.sum(np.array((ids_pred == id_pred) & (ids_gt == id_gt)))
            union = np.sum(np.array((ids_pred == id_pred) | (ids_gt == id_gt)))
            
            iou = intersection / union
            
            matrix_parts_iou[id_gt, id_pred] = iou
            
    return matrix_parts_iou

#######################################################################################################
#######################################################################################################

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

#######################################################################################################
#######################################################################################################

def get_mapping_ids_pred_gt(ids_pred, ids_gt):
    """
    Gets the mapping between the predicted ids and the ground truth ids.
    
    params: ids_pred: predicted ids
    params: ids_gt: ground truth ids
    
    :return: ids_pred_h: mapping between the predicted ids and the ground truth ids
    :return: ids_gt_h: mapping between the ground truth ids and the predicted ids
        
    """
    # compute the IoU matrix over the parts
    matrix_parts_iou = compute_iou_matrix_over_parts(ids_pred, ids_gt)

    # ids_pred_h, ids_gt_h contains the correct assignment of the ids
    # between the ground truth and the predicted ids
    matrix_parts_iou, ids_pred_h, ids_gt_h = hungarian_algorithm(matrix_parts_iou)
    
    return matrix_parts_iou, ids_pred_h, ids_gt_h

#######################################################################################################
#######################################################################################################

def get_interpolated_features_img_obj_in_context(feature_map_obj_in_context, CANVAS_HEIGHT=672, CANVAS_WIDTH=672, device='cpu'):
    """
    Gets the interpolated features of the image in context.
    
    params: feature_map_obj_in_context: feature map of the image in context
    params: CANVAS_HEIGHT: height of the canvas
    params: CANVAS_WIDTH: width of the canvas
    params: device: device to perform computations on
    
    :return: interpolated_feature_map_obj_in_context: interpolated feature map of the image in context
    
    """
    
    return interpolate_feature_map(feature_map_obj_in_context, CANVAS_WIDTH, CANVAS_HEIGHT)

#######################################################################################################
#######################################################################################################

def get_cluster_assignment_renders_metrics(rendered_images, mappings, point_cloud, context_image, ids_gt, model, image_processor, device='cpu'):#, n_clusters=5):
    """
    Gets the cluster assignment of the rendered images.
    
    params: rendered_images: list of rendered images [n_images, 3, width, height]
    params: mappings: list of mappings from point cloud to image
    params: point_cloud: point cloud to render
    params: chair: image in context [1, 3, width, height]
    params: ids_gt: ground truth ids (segmentation)
    params: n_clusters: number of clusters
    
    :return: cluster_assignment_with_2d_context: cluster assignment with 2d context
    :return: cluster_assignment_without_2d_context: cluster assignment without 2d context
    :return: render_clustering_with_2d_context: plot of the point cloud with the cluster assignment with 2d context
    :return: render_clustering_without_2d_context: plot of the point cloud with the cluster assignment without 2d context
    :return: metrics_with_2d_context: metrics with 2d context
    :return: metrics_without_2d_context: metrics without 2d context
    :return: matrix_parts_iou_with_context: IoU matrix over the parts with 2d context
    :return: matrix_parts_iou_without_context: IoU matrix over the parts without 2d context
    :return: context_image_mask: image in context with the mask of the clusters
    
    """
    # metrics
    seg = SegMetrics()
    
    # extract features and aggregate them
    output_3d, output_2d_context = get_the_features(rendered_images, context_image, model, image_processor, device=device)
    aggregate_features_3d = get_the_aggregated_features(output_3d, mappings, point_cloud, CANVAS_HEIGHT=CANVAS_HEIGHT, CANVAS_WIDTH=CANVAS_WIDTH, device='cpu')
    aggregated_feat_3d_in_context = torch.cat([
                                    aggregate_features_3d.cpu(), 
                                    output_2d_context.last_hidden_state.squeeze(0).cpu()
                                ], dim=0)
    
    # get unique ids_gt ids
    unique_ids_gt = np.unique(ids_gt)
    # print('unique_ids_gt', unique_ids_gt)
    n_clusters = len(unique_ids_gt)
    # print('n_clusters', n_clusters)

    ## get the clusters and compute the metrics
    K_MEANS_CLUSTERS = n_clusters
    
    # Find mask for each cluster in the context image
    CANVAS_HEIGHT_context = context_image.shape[2]
    CANVAS_WIDTH_context = context_image.shape[1]

    # get the interpolated feature map of the image in context
    interpolated_fm = get_interpolated_features_img_obj_in_context(output_2d_context, CANVAS_HEIGHT=CANVAS_HEIGHT_context, CANVAS_WIDTH=CANVAS_WIDTH_context)
    interpolated_fm = torch.flatten(interpolated_fm, start_dim=1, end_dim=-2).squeeze(0)
    cluster_assignment = kmeans(K_MEANS_CLUSTERS, interpolated_fm.cpu(), iterations=30)
    del interpolated_fm

    # get the rgb colors for each label
    rgb_colors = assign_rgb_colors_to_labels(cluster_assignment)

    # reduce the opacity of the mask to place it on top of the image
    mask = rgb_colors.reshape((CANVAS_WIDTH_context, CANVAS_HEIGHT_context, 3)).clip(0, 255).astype(np.uint8)
    mask = Image.fromarray(mask)
    mask.putalpha(150)

    # place the mask on top of the image reducing the opacity
    context_image_mask = context_image.cpu().numpy().squeeze(0)
    context_image_mask = Image.fromarray(context_image_mask.astype(np.uint8))
    context_image_mask.putalpha(255)
    context_image_mask.paste(mask, (0, 0), mask)
    context_image_mask = np.array(context_image_mask) # [width, height, 3] masked image in context
    
    
    # with 2d context
    cluster_assignment_with_2d_context = kmeans(K_MEANS_CLUSTERS , aggregated_feat_3d_in_context, iterations=40)[:point_cloud.shape[0]]
    metrics_with_2d_context = seg.compute_all_metrics(torch.tensor(cluster_assignment_with_2d_context), ids_gt, device='cpu')
    
    # get mapping between the predicted ids and the ground truth ids
    matrix_parts_iou_with_context, ids_pred_h, ids_gt_h = get_mapping_ids_pred_gt(cluster_assignment_with_2d_context, ids_gt)
    cluster_assignment_with_2d_context = np.array([ids_pred_h[int(pred)] for pred in cluster_assignment_with_2d_context])

    # get the rgb colors for each label
    rgb_colors = assign_rgb_colors_to_labels(cluster_assignment_with_2d_context)

    # visualize the point cloud
    xyz_rgb = np.concatenate([point_cloud[:, :3], rgb_colors[:point_cloud.shape[0]]], axis=-1)
    # visualize_with_open3d(xyz_rgb)
    render_clustering_with_2d_context = plot_point_cloud(xyz_rgb, title='mIoU='+str(round(metrics_with_2d_context['meanIoU'],2))+',randScore='+str(round(metrics_with_2d_context['rand_score'],2))+' with 2d context', axis=True, show=False)
    
    
    
    # without 2d context
    cluster_assignment_without_2d_context = kmeans(K_MEANS_CLUSTERS, aggregated_feat_3d_in_context, iterations=40)[:point_cloud.shape[0]]
    metrics_without_2d_context = seg.compute_all_metrics(torch.tensor(cluster_assignment_without_2d_context), ids_gt, device='cpu')
    
    # get mapping between the predicted ids and the ground truth ids
    matrix_parts_iou_without_context, ids_pred_h, ids_gt_h = get_mapping_ids_pred_gt(cluster_assignment_without_2d_context, ids_gt)
    cluster_assignment_without_2d_context = np.array([ids_pred_h[int(pred)] for pred in cluster_assignment_without_2d_context])

    # get the rgb colors for each label
    rgb_colors = assign_rgb_colors_to_labels(cluster_assignment_without_2d_context)

    # visualize the point cloud
    xyz_rgb = np.concatenate([point_cloud[:, :3], rgb_colors[:point_cloud.shape[0]]], axis=-1)
    # visualize_with_open3d(xyz_rgb)
    render_clustering_without_2d_context = plot_point_cloud(xyz_rgb, title='mIoU='+str(round(metrics_without_2d_context['meanIoU'],2))+',randScore='+str(round(metrics_without_2d_context['rand_score'],2))+' without 2d context', axis=True, show=False)
    
    
    
    return cluster_assignment_with_2d_context, cluster_assignment_without_2d_context, render_clustering_with_2d_context, render_clustering_without_2d_context, metrics_with_2d_context, metrics_without_2d_context, matrix_parts_iou_with_context, matrix_parts_iou_without_context, context_image_mask
