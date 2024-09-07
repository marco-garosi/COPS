import sys

import data.PartNetSemanticSegmentation

sys.path.append('../source')

import config

import torch
from torch_geometric.nn import knn
import numpy as np
import json
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from source.point_cloud_utils.feature_interpolation import *
from source.point_cloud_utils.feature_aggregation import *
from source.rendering.render_and_map import render_with_orientations

from source.metrics.compute_metrics import compute_all_metrics, preprocess_ground_truth, compute_miou_partslip
from source.point_cloud_utils.clustering import get_best_cluster_assignment
from source.point_cloud_utils.kmeans import kmeans
from source.point_cloud_utils.FeatureExtractor import FeatureExtractor, FeatureExtractorPointCLIPv2, FeatureExtractorCLIP
from source.models.Geometric import GeometricAwareFeatureAggregation
import source.models.modifiedCLIP as CLIPModified

try:
    from pytorch3d.renderer import look_at_view_transform
except:
    print('PyTorch3D not available')

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from tqdm import tqdm

from utils.setup_script import setup

MODEL_NAME_CLIP = 'ViT-B/16'
METRIC = 'meanIoU'
EARLY_SUBSAMPLE = 100_000

# Tests to run
# To be managed carefully, as disabling one may break others
# due to dependencies
EXTRACT_DINO_FEATURES = True
HUNGARIAN_WITH_GT = True
HUNGARIAN_WITH_GT_REFINEMENT = True
HUNGARIAN_WITH_GT_GEOMETRIC = True
HUNGARIAN_WITH_GT_GEOMETRIC_AND_REFINEMENT = True
POINTCLIPV2 = True
HUNGARIAN_WITH_POINTCLIPV2 = True
HUNGARIAN_WITH_POINTCLIPV2_REFINEMENT = True
HUNGARIAN_WITH_POINTCLIPV2_GEOMETRIC = True
HUNGARIAN_WITH_POINTCLIPV2_GEOMETRIC_AND_REFINEMENT = True

# Prompt settings
USE_PART_NAMES = False
TEMPLATE_PART_NAMES = None
# TEMPLATE_PART_NAMES = 'this is the {part_name} of a {category}'
# TEMPLATE_PART_NAMES = 'this is a depth map of a {part_name} of a {category}'

# from prompts.shapenetpart import best_vweight

# with open(f'../source/prompts/PartNetE_meta.json', 'r') as f:
#     partnete_meta = json.load(f)


def save_results(segmentation_metrics, store_pcd_idx=None, store_ground_truth_segmentations=None, store_predictions=None):
    for idx, segmentation_metrics in enumerate(segmentation_metrics):
        with open(os.path.join('results', f'exp_{args.layer_features}_segmentation_metrics_{args.split}_{idx}.json'), 'w') as f:
            json.dump(segmentation_metrics, f)

    if store_pcd_idx is not None and store_ground_truth_segmentations is not None and store_predictions is not None:
        for idx, gt_seg in enumerate(store_ground_truth_segmentations):
            torch.save(gt_seg, os.path.join('results', f'exp_{args.layer_features}_store_gt_seg_{idx}.pt'))
        torch.save(store_predictions, os.path.join('results', f'exp_{args.layer_features}_store_predictions.pt'))
        torch.save(torch.tensor(store_pcd_idx), os.path.join('results', f'exp_{args.layer_features}_store_pcd_idx.pt'))


def cluster_refinement(point_cloud, neighbors, cluster_assignment):
    if neighbors is None: return

    knn_on_cluster_assignment = knn(point_cloud[:, :3], point_cloud[:, :3], neighbors)
    cluster_assignments_at_each_neighborhood = cluster_assignment[knn_on_cluster_assignment[1]]
    cleaned_cluster_assignment = cluster_assignments_at_each_neighborhood.view(len(point_cloud), -1).mode(dim=-1).values

    return cleaned_cluster_assignment


def evaluate(
        model, image_processor,
        all_prompts,
        orientations,
        dataset, dataloader, split,
        canvas_width, canvas_height,
        fx, fy, cx, cy,
        point_size, light_intensity,
        use_preprocessed_features,
        store_features,
        backbone_model,
        preprocessed_folder=None,
        k_means_iterations=30,
        subsample_point_cloud=25_000,
        early_subsample=True,
        refine_clusters=None,
        use_colorized_renders=True,
        repeat_clustering=5,
        layer_features=None,
        device='cpu'):

    # Collect [CLS] features and ground truths
    gt_classes = []
    gt_cluster_lengths = []

    # Collect predictions
    # Hungarian with ground truth
    segmentation_metrics_hungarian_with_gt = []
    segmentation_metrics_hungarian_with_gt_refinement = []
    segmentation_metrics_hungarian_with_gt_geometric = []
    segmentation_metrics_hungarian_with_gt_geometric_and_refinement = []
    # PointCLIPv2
    segmentation_metrics_pointclipv2 = []
    segmentation_metrics_hungarian_with_pointclipv2 = []
    segmentation_metrics_hungarian_with_pointclipv2_refinement = []
    segmentation_metrics_hungarian_with_pointclipv2_geometric = []
    segmentation_metrics_hungarian_with_pointclipv2_geometric_and_refinement = []

    # Store
    store_ground_truth_segmentations = []
    store_ground_truth_segmentations_pcv2 = []
    store_pcd_idx = []
    store_predictions = {k: [] for k in range(9)}

    # Orientations
    rotations = torch.tensor(orientations['rotations'])
    translations = torch.tensor(orientations['translations'])

    # Load CLIP
    model_clip, image_processor_clip = CLIPModified.clip.load(MODEL_NAME_CLIP, device=device)
    model_clip = model_clip.eval()

    # Encode all prompts to avoid repeating encoding later
    text_features = {}
    for category, prompts in all_prompts.items():
        text_input = torch.cat([CLIPModified.clip.tokenize(prompt) for prompt in prompts]).to(device)
        with torch.no_grad():
            text_feat = model_clip.encode_text(text_input)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        text_features[category] = text_feat

    # Feature extractor
    feature_extractor = FeatureExtractor(image_processor, model, backbone_model, rotations, translations,
                                         canvas_width=canvas_width, canvas_height=canvas_height,
                                         use_colorized_renders=use_colorized_renders,
                                         subsample_point_cloud=subsample_point_cloud, early_subsample=early_subsample,
                                         point_size=point_size, layer_features=layer_features).eval()
    feature_extractor_pointclipv2 = FeatureExtractorPointCLIPv2(model_clip)
    # feature_extractor_pointclipv2 = FeatureExtractorCLIP(model_clip, rotations, translations,
    #                                                      use_colorized_renders=use_colorized_renders,
    #                                                      canvas_width=canvas_width, canvas_height=canvas_height,
    #                                                      subsample_point_cloud=subsample_point_cloud, early_subsample=early_subsample,
    #                                                      improved_depth_maps=True, perspective=False,
    #                                                      point_size=point_size).eval()
    module = GeometricAwareFeatureAggregation().to(device)

    # Iterate the dataset in batches
    pbar = tqdm(dataloader, desc='Batch')
    pcd_idx = -1
    for batch_idx, batch in enumerate(pbar):
        if use_preprocessed_features:
            point_clouds, segmentations, categories, rendered_images_batch, mappings_batch, features_batch, feature_pcd_aggregated_batch = batch
            zip_data = zip(point_clouds, segmentations, categories, rendered_images_batch, mappings_batch, features_batch, feature_pcd_aggregated_batch)
        else:
            point_clouds, segmentations, categories = batch
            zip_data = zip(point_clouds, segmentations, categories)

        # Collect ground truths
        # Using `+` and not `append()` to ensure the ground truth lists are not nested lists
        # but flat lists, so that when later converting them to tensors no issues will arise,
        # in particular due to the last batch of data, that can be smaller than all previous
        # batches
        gt_classes += categories
        gt_cluster_lengths += [len(torch.unique(segmentation)) for segmentation in segmentations]

        # Iterate over each point cloud in the batch, and treat it separately
        for packed_input in tqdm(zip_data, desc='Point Cloud'):
            pcd_idx += 1

            # Prepare storage for tensors
            folder = dataset.get_pcd_id(pcd_idx)
            os.makedirs(os.path.join(preprocessed_folder, folder), exist_ok=True)

            if use_preprocessed_features:
                point_cloud, segmentation, category, rendered_images, mappings, outputs, feature_pcd_aggregated = packed_input
            else:
                point_cloud, segmentation, category = packed_input

            # For PartNetE
            # if category not in [2, 3, 4, 5, 8, 10, 17, 20, 23, 24, 25, 30, 37, 39, 44]:
            #     continue

            # For PartNet
            # if category not in [20, 15, 23, 12, 19, 13, 4, 7, 18, 1, 22, 10, 8, 14, 9, 17, 21]:
            #     continue

            np.random.seed(0)
            if EARLY_SUBSAMPLE is not None and len(point_cloud) > EARLY_SUBSAMPLE:
                indices = np.array(list(range(len(point_cloud))))
                subsampled_point_indices = np.random.choice(indices, EARLY_SUBSAMPLE, replace=False)
                subsampled_point_indices = torch.tensor(subsampled_point_indices)

                point_cloud = point_cloud[subsampled_point_indices]
                segmentation = segmentation[subsampled_point_indices]

            # Store point cloud index
            store_pcd_idx.append(pcd_idx)

            # Move to device
            category = category.item() if isinstance(category, torch.Tensor) else category
            point_cloud = point_cloud.to(device)
            segmentation = segmentation.to(device) - dataset.index_start[category]
            num_classes = max(dataset.get_number_of_parts('max')[category], segmentation.unique().max().item() + 1)
            gt_k = len(segmentation.unique())

            # Extract features
            if EXTRACT_DINO_FEATURES:
                point_cloud, segmentation, images, mappings, feature_pcd_aggregated = feature_extractor(point_cloud, segmentation)

            if HUNGARIAN_WITH_GT:
                cluster_assignment_hungarian_with_gt, metrics_hungarian_with_gt = get_best_cluster_assignment(
                    feature_pcd_aggregated,
                    repeat_clustering,
                    k_means_iterations,
                    gt_k,
                    segmentation,
                    metric=METRIC,
                    num_classes=num_classes,
                    device=device
                )
                store_predictions[0].append(cluster_assignment_hungarian_with_gt)
                segmentation_metrics_hungarian_with_gt.append(metrics_hungarian_with_gt)

            # Refinement
            if HUNGARIAN_WITH_GT_REFINEMENT:
                cluster_assignment_hungarian_with_gt_refinement = cluster_refinement(point_cloud, refine_clusters, cluster_assignment_hungarian_with_gt)
                metrics_hungarian_with_gt_refinement = compute_all_metrics(cluster_assignment_hungarian_with_gt_refinement, segmentation, num_classes=num_classes, device=device)
                store_predictions[1].append(cluster_assignment_hungarian_with_gt_refinement)
                segmentation_metrics_hungarian_with_gt_refinement.append(metrics_hungarian_with_gt_refinement)

            # Geometric module
            if HUNGARIAN_WITH_GT_GEOMETRIC:
                point_cloud_geometric, feature_pcd_aggregated_geometric, other_tensors = module(
                    point_cloud,
                    feature_pcd_aggregated,
                    torch.stack([segmentation])
                )
                cluster_assignment_hungarian_with_gt_geometric, metrics_hungarian_with_gt_geometric = get_best_cluster_assignment(
                    feature_pcd_aggregated_geometric,
                    repeat_clustering,
                    k_means_iterations,
                    gt_k,
                    segmentation,
                    metric=METRIC,
                    num_classes=num_classes,
                    device=device
                )
                store_predictions[2].append(cluster_assignment_hungarian_with_gt_geometric)
                segmentation_metrics_hungarian_with_gt_geometric.append(metrics_hungarian_with_gt_geometric)

            # Geometric module + refinement
            if HUNGARIAN_WITH_GT_GEOMETRIC_AND_REFINEMENT:
                cluster_assignment_hungarian_with_gt_geometric_and_refinement = cluster_refinement(point_cloud, refine_clusters, cluster_assignment_hungarian_with_gt_geometric)
                metrics_hungarian_with_gt_geometric_and_refinement = compute_all_metrics(cluster_assignment_hungarian_with_gt_geometric_and_refinement, segmentation, num_classes=num_classes, device=device)
                store_predictions[3].append(cluster_assignment_hungarian_with_gt_geometric_and_refinement)
                segmentation_metrics_hungarian_with_gt_geometric_and_refinement.append(metrics_hungarian_with_gt_geometric_and_refinement)

            # Predict with PointCLIPv2
            if POINTCLIPV2:
                # vweights = torch.tensor(best_vweight[dataset.class_ids[category].lower()]).to(device).view(1, -1, 1, 1)
                # segmentation = preprocess_ground_truth(segmentation)[0]
                point_cloud_pointclipv2, segmentation_pointclipv2, other_features_pointclipv2, pointclipv2_renders, seg_pred_pointclipv2 = \
                    feature_extractor_pointclipv2(
                        point_cloud, segmentation,
                        text_features[dataset.class_ids[category].lower()],
                        feature_pcd_aggregated.unsqueeze(0) if 'feature_pcd_aggregated' in locals() else None,
                        vweights=vweights if 'vweights' in locals() else None
                    )
                feature_pcd_aggregated_pointclipv2 = other_features_pointclipv2[0]
                pointclipv2_k = len(seg_pred_pointclipv2.unique())
                # For PartNet only
                if isinstance(dataset, data.PartNetSemanticSegmentation.PartNetSemanticSegmentation):
                    seg_pred_pointclipv2 += 1
                print(seg_pred_pointclipv2.unique(), segmentation_pointclipv2.unique(), dataset.class_ids[category])
                metrics_pointclipv2 = compute_all_metrics(seg_pred_pointclipv2.cpu(),
                                                          segmentation_pointclipv2.cpu(),
                                                          num_classes=num_classes,
                                                          device=device)
                store_predictions[4].append(seg_pred_pointclipv2)
                segmentation_metrics_pointclipv2.append(metrics_pointclipv2)


            # PointCLIPv2 as guidance
            if HUNGARIAN_WITH_POINTCLIPV2:
                cluster_assignment_hungarian_with_pointclipv2, metrics_hungarian_pointclipv2 = get_best_cluster_assignment(
                    feature_pcd_aggregated_pointclipv2,
                    repeat_clustering,
                    k_means_iterations,
                    pointclipv2_k,
                    seg_pred_pointclipv2,  # PointCLIPv2's prediction
                    segmentation_gt=segmentation_pointclipv2,  # Ground truth segmentation
                    metric=METRIC,
                    num_classes=num_classes,
                    device=device
                )
                store_predictions[5].append(cluster_assignment_hungarian_with_pointclipv2)
                segmentation_metrics_hungarian_with_pointclipv2.append(metrics_hungarian_pointclipv2)

            # Refinement
            if HUNGARIAN_WITH_POINTCLIPV2_REFINEMENT:
                cluster_assignment_hungarian_with_pointclipv2_refinement = cluster_refinement(point_cloud_pointclipv2, refine_clusters, cluster_assignment_hungarian_with_pointclipv2)
                metrics_hungarian_with_pointclipv2_refinement = compute_all_metrics(cluster_assignment_hungarian_with_pointclipv2_refinement, segmentation_pointclipv2, num_classes=num_classes, device=device)
                store_predictions[6].append(cluster_assignment_hungarian_with_pointclipv2_refinement)
                segmentation_metrics_hungarian_with_pointclipv2_refinement.append(metrics_hungarian_with_pointclipv2_refinement)

            # Geometric module
            if HUNGARIAN_WITH_POINTCLIPV2_GEOMETRIC:
                point_cloud_pointclipv2, feature_pcd_aggregated_pointclipv2, other_tensors = module(
                    point_cloud_pointclipv2,
                    feature_pcd_aggregated_pointclipv2,
                    torch.stack([segmentation_pointclipv2])
                )
                print(pointclipv2_k, seg_pred_pointclipv2.unique())
                cluster_assignment_hungarian_with_pointclipv2_geometric, metrics_hungarian_pointclipv2_geometric = get_best_cluster_assignment(
                    feature_pcd_aggregated_pointclipv2,
                    repeat_clustering,
                    k_means_iterations,
                    pointclipv2_k,
                    seg_pred_pointclipv2,  # PointCLIPv2's prediction
                    segmentation_gt=segmentation_pointclipv2,  # Ground truth segmentation
                    metric=METRIC,
                    num_classes=num_classes,
                    device=device
                )
                print(cluster_assignment_hungarian_with_pointclipv2_geometric.unique())
                store_predictions[7].append(cluster_assignment_hungarian_with_pointclipv2_geometric)
                segmentation_metrics_hungarian_with_pointclipv2_geometric.append(metrics_hungarian_pointclipv2_geometric)

            # Geometric + Refinement
            if HUNGARIAN_WITH_POINTCLIPV2_GEOMETRIC_AND_REFINEMENT:
                cluster_assignment_hungarian_with_pointclipv2_geometric_and_refinement = cluster_refinement(point_cloud_pointclipv2, refine_clusters, cluster_assignment_hungarian_with_pointclipv2_geometric)
                metrics_hungarian_with_pointclipv2_geometric_and_refinement = compute_all_metrics(cluster_assignment_hungarian_with_pointclipv2_geometric_and_refinement, segmentation_pointclipv2, num_classes=num_classes, device=device)
                store_predictions[8].append(cluster_assignment_hungarian_with_pointclipv2_geometric_and_refinement)
                segmentation_metrics_hungarian_with_pointclipv2_geometric_and_refinement.append(metrics_hungarian_with_pointclipv2_geometric_and_refinement)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            store_ground_truth_segmentations += [segmentation.cpu()]
            store_ground_truth_segmentations_pcv2 += [segmentation_pointclipv2.cpu()]

            # Checkpointing...
            save_results([
                segmentation_metrics_hungarian_with_gt,
                segmentation_metrics_hungarian_with_gt_refinement,
                segmentation_metrics_hungarian_with_gt_geometric,
                segmentation_metrics_hungarian_with_gt_geometric_and_refinement,

                segmentation_metrics_pointclipv2,
                segmentation_metrics_hungarian_with_pointclipv2,
                segmentation_metrics_hungarian_with_pointclipv2_refinement,
                segmentation_metrics_hungarian_with_pointclipv2_geometric,
                segmentation_metrics_hungarian_with_pointclipv2_geometric_and_refinement
            ], store_pcd_idx=store_pcd_idx, store_ground_truth_segmentations=[store_ground_truth_segmentations, store_ground_truth_segmentations_pcv2], store_predictions=store_predictions)

            # Free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return [
        segmentation_metrics_hungarian_with_gt,
        segmentation_metrics_hungarian_with_gt_refinement,
        segmentation_metrics_hungarian_with_gt_geometric,
        segmentation_metrics_hungarian_with_gt_geometric_and_refinement,

        segmentation_metrics_pointclipv2,
        segmentation_metrics_hungarian_with_pointclipv2,
        segmentation_metrics_hungarian_with_pointclipv2_refinement,
        segmentation_metrics_hungarian_with_pointclipv2_geometric,
        segmentation_metrics_hungarian_with_pointclipv2_geometric_and_refinement
    ]


if __name__ == '__main__':
    args, model, image_processor, embedding_dimension, prediction_head, orientations, dataset, dataloader, preprocessed_folder, device = setup()

    # Load prompts
    part_names = '_part_names' if USE_PART_NAMES else ''
    with open(f'../source/prompts/{args.dataset}{part_names}.json', 'r') as f:
        all_prompts = json.load(f)
    if USE_PART_NAMES and TEMPLATE_PART_NAMES is not None:
        for category in all_prompts.keys():
            all_prompts[category] = [TEMPLATE_PART_NAMES.format(part_name=part_name, category=category) for part_name in all_prompts[category]]

    # Evaluate the model
    print('\n' + '=' * 89, flush=True)
    segmentation_metrics = evaluate(
        model, image_processor,
        all_prompts,
        orientations,
        dataset, dataloader, args.split,
        args.canvas_width, args.canvas_height,
        args.fx, args.fy, args.cx, args.cy,
        args.point_size, args.light_intensity,
        args.use_preprocessed_features, args.store_features,
        args.backbone_model,
        preprocessed_folder=preprocessed_folder,
        k_means_iterations=args.k_means_iterations,
        subsample_point_cloud=args.subsample_pcd,
        early_subsample=args.early_subsample,
        refine_clusters=args.refine_clusters,
        use_colorized_renders=args.use_colorized_renders,
        repeat_clustering=args.repeat_clustering,
        layer_features=args.layer_features,
        device=device
    )

    print('--> Saving results')
    save_results(segmentation_metrics)
