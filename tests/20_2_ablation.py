import copy
import sys

sys.path.append('../source')

import config

import torch
from torch_geometric.nn import knn
import numpy as np
import json
import itertools
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
POINTCLIPV2 = True

RUN_NO_GFA = True
RUN_GFA_FULL = True

# Prompt settings
USE_GPT_PROMPTS = True
USE_PART_NAMES_TEMPLATES = 'this is a depth map of a {part_name} of a {category}'
USE_PART_NAMES = True

# Orientation settings
NUM_OF_VIEWS = list(reversed(['48', '10', '10_improved', '6']))
ORIENTATIONS = list(reversed(['002.npz', '004.npz', '004.npz', '003.npz']))  # 48, 10, 10, 6 views
IMPROVED_DEPTH_MAPS = list(reversed([False, False, True, False]))
PERSPECTIVE = list(reversed([True, True, False, True]))

# from prompts.shapenetpart import best_vweight

# with open(f'../source/prompts/PartNetE_meta.json', 'r') as f:
#     partnete_meta = json.load(f)


def save_results(segmentation_metrics, store_pcd_idx, store_ground_truth_segmentations, store_predictions, convert_to_tensor=False):
    # segmentation_metrics = copy.deepcopy(segmentation_metrics)
    with open(os.path.join('results', f'segmentation_metrics_{args.split}.json'), 'w') as f:
        json.dump(segmentation_metrics, f)

    if convert_to_tensor:
        store_predictions = {k: torch.stack(v) for k, v in store_predictions.items()}

    torch.save(store_ground_truth_segmentations, os.path.join('results', 'store_gt_seg.pt'))
    torch.save(store_predictions, os.path.join('results', 'store_predictions.pt'))
    torch.save(torch.tensor(store_pcd_idx), os.path.join('results', 'store_pcd_idx.pt'))


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
        device='cpu'):

    # Collect [CLS] features and ground truths
    gt_classes = []
    gt_cluster_lengths = []

    # Collect predictions
    segmentation_metrics = {
        'pointclipv2__gpt': [],

        # No GFA
        'no_gfa__gpt__6_views': [],
        'no_gfa__gpt__10_views': [],
        'no_gfa__gpt__10_improved_views': [],
        'no_gfa__gpt__48_views': [],

        # GFA full
        'gfa_full__gpt__6_views': [],
        'gfa_full__gpt__10_views': [],
        'gfa_full__gpt__10_improved_views': [],
        'gfa_full__gpt__48_views': [],
    }

    # Store
    store_ground_truth_segmentations = []
    store_pcd_idx = []
    store_predictions = copy.deepcopy(segmentation_metrics)

    # Orientations
    rotations = {
        k: torch.tensor(v['rotations'])
        for k, v in orientations.items()
    }
    translations = {
        k: torch.tensor(v['translations'])
        for k, v in orientations.items()
    }

    # Load CLIP
    model_clip, image_processor_clip = CLIPModified.clip.load(MODEL_NAME_CLIP, device=device)
    model_clip = model_clip.eval()

    # Encode all prompts to avoid repeating encoding later
    text_features = {}
    for setting, all_prompts_in_setting in all_prompts.items():
        text_features[setting] = {}
        for category, prompts in all_prompts_in_setting.items():
            text_input = torch.cat([CLIPModified.clip.tokenize(prompt) for prompt in prompts]).to(device)
            with torch.no_grad():
                text_feat = model_clip.encode_text(text_input)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            text_features[setting][category] = text_feat

    # Feature extractor
    feature_extractors = [
        FeatureExtractor(image_processor, model, backbone_model, R, T,
                         canvas_width=canvas_width, canvas_height=canvas_height,
                         use_colorized_renders=use_colorized_renders,
                         subsample_point_cloud=subsample_point_cloud, early_subsample=early_subsample,
                         point_size=point_size,
                         improved_depth_maps=IMPROVED_DEPTH_MAPS[idx], perspective=PERSPECTIVE[idx]).eval()

        for idx, (R, T) in enumerate(zip(rotations.values(), translations.values()))
    ]
    feature_extractor_pointclipv2 = FeatureExtractorPointCLIPv2(model_clip)
    # feature_extractor_pointclipv2 = FeatureExtractorCLIP(model_clip, rotations, translations,
    #                                                      use_colorized_renders=use_colorized_renders,
    #                                                      canvas_width=canvas_width, canvas_height=canvas_height,
    #                                                      subsample_point_cloud=subsample_point_cloud, early_subsample=early_subsample,
    #                                                      improved_depth_maps=True, perspective=False,
    #                                                      point_size=point_size).eval()
    module_gfa_full = GeometricAwareFeatureAggregation().to(device)


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
                # List of M lists, where each of the M lists contains:
                # point_cloud, segmentation, images, mappings, feature_pcd_aggregated
                results = [
                    feature_extractor(point_cloud, segmentation)
                    for feature_extractor in feature_extractors
                ]

                # Take subsampled point cloud and segmentation
                # Thanks to reproducibility in the subsampling process, each call to "feature_extractor"
                # will return exactly the same points
                # print(results)
                point_cloud = results[0][0]
                segmentation = results[0][1]

                # We can safely skip/ignore returned images and mappings, as we don't need them anywhere here
                feature_pcd_aggregated = {
                    k: v[-1]
                    for k, v in zip(NUM_OF_VIEWS, results)
                }

                # point_cloud, segmentation, images, mappings, feature_pcd_aggregated = feature_extractor(point_cloud, segmentation)

            if RUN_GFA_FULL:
                results = [
                    module_gfa_full(
                        point_cloud,
                        feat,
                        segmentation.unsqueeze(0)
                    )
                    for feat in feature_pcd_aggregated.values()
                ]

                feature_pcd_aggregated_gfa_full = {
                    k: v[1]
                    for k, v in zip(NUM_OF_VIEWS, results)
                }

            # Predict with PointCLIPv2
            if POINTCLIPV2:
                if 'feature_pcd_aggregated' not in locals():
                    raise Exception('feature_pcd_aggregated not defined')

                text_features_all_types = torch.stack([v[dataset.class_ids[category].lower()] for v in text_features.values()])

                # vweights = torch.tensor(best_vweight[dataset.class_ids[category].lower()]).to(device).view(1, -1, 1, 1)
                # segmentation = preprocess_ground_truth(segmentation)[0]

                other_tensors = list(itertools.chain.from_iterable([
                    list(feature_pcd_aggregated.values()),
                    list(feature_pcd_aggregated_gfa_full.values())
                ]))
                other_tensors_keys = list(itertools.chain.from_iterable([
                    list(feature_pcd_aggregated.keys()),
                    list(feature_pcd_aggregated_gfa_full.keys())
                ]))

                point_cloud_pointclipv2, segmentation_pointclipv2, other_features_pointclipv2, pointclipv2_renders, seg_pred_pointclipv2 = \
                    feature_extractor_pointclipv2(
                        point_cloud, segmentation,
                        text_features_all_types,
                        other_tensors,
                        vweights=vweights if 'vweights' in locals() else None
                    )
                if len(seg_pred_pointclipv2.shape) == 1:  # (#prompts), without setting as first dimension
                    seg_pred_pointclipv2 = seg_pred_pointclipv2.unsqueeze(0)  # (1, #prompts)

                feature_pcd_aggregated = {
                    k: v
                    for k, v in zip(other_tensors_keys[:len(feature_pcd_aggregated)],
                                    other_features_pointclipv2[:len(feature_pcd_aggregated)])
                }
                feature_pcd_aggregated_gfa_full = {
                    k: v
                    for k, v in zip(other_tensors_keys[len(feature_pcd_aggregated_gfa_full):],
                                    other_features_pointclipv2[len(feature_pcd_aggregated_gfa_full):])
                }

                # feature_pcd_aggregated = other_features_pointclipv2[0]
                # feature_pcd_aggregated_gfa_full = other_features_pointclipv2[1]
                pointclipv2_k = [len(p.unique()) for p in seg_pred_pointclipv2]

                # For PartNet only
                # Because PartNet discards idx 0 (which is not even described), therefore
                # our prompts must be shifted one to the right so that they can actually match
                # the ground truth
                # seg_pred_pointclipv2 += 1
                store_ground_truth_segmentations += [segmentation_pointclipv2.cpu()]
                for idx, task in enumerate(text_features.keys()):
                    # print(seg_pred_pointclipv2[idx].shape, seg_pred_pointclipv2.shape, segmentation_pointclipv2.shape)
                    segmentation_metrics[f'pointclipv2__{task}'] += [compute_all_metrics(
                                                            seg_pred_pointclipv2[idx].cpu(),
                                                            segmentation_pointclipv2.cpu(),
                                                            num_classes=num_classes,
                                                            device=device)]
                    store_predictions[f'pointclipv2__{task}'] += [seg_pred_pointclipv2[idx].cpu()]

            if RUN_NO_GFA:
                for idx, task in enumerate(text_features.keys()):
                    for number_of_views in NUM_OF_VIEWS:
                        cluster_assignment, metrics = get_best_cluster_assignment(
                            feature_pcd_aggregated[number_of_views],  # For current number of views
                            repeat_clustering,
                            k_means_iterations,
                            pointclipv2_k[idx],  # For current task (i.e. GPT)
                            seg_pred_pointclipv2[idx],  # PointCLIPv2's prediction, for current task (i.e. GPT)
                            segmentation_gt=segmentation_pointclipv2,  # Ground truth segmentation
                            metric=METRIC,
                            num_classes=num_classes,
                            device=device
                        )
                        store_predictions[f'no_gfa__{task}__{number_of_views}_views'] += [cluster_assignment.cpu()]
                        segmentation_metrics[f'no_gfa__{task}__{number_of_views}_views'] += [metrics]

            if RUN_GFA_FULL:
                for idx, task in enumerate(text_features.keys()):
                    for number_of_views in NUM_OF_VIEWS:
                        cluster_assignment, metrics = get_best_cluster_assignment(
                            feature_pcd_aggregated_gfa_full[number_of_views],  # For current number of views
                            repeat_clustering,
                            k_means_iterations,
                            pointclipv2_k[idx],  # For current task (i.e. GPT)
                            seg_pred_pointclipv2[idx],  # PointCLIPv2's prediction, for current task (i.e. GPT)
                            segmentation_gt=segmentation_pointclipv2,  # Ground truth segmentation
                            metric=METRIC,
                            num_classes=num_classes,
                            device=device
                        )
                        store_predictions[f'gfa_full__{task}__{number_of_views}_views'] += [cluster_assignment.cpu()]
                        segmentation_metrics[f'gfa_full__{task}__{number_of_views}_views'] += [metrics]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Checkpointing...
            save_results(segmentation_metrics, store_pcd_idx, store_ground_truth_segmentations, store_predictions)

            # Free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return segmentation_metrics


if __name__ == '__main__':
    args, model, image_processor, embedding_dimension, prediction_head, orientations, dataset, dataloader, preprocessed_folder, device = setup()

    # Load prompts
    with open(f'../source/prompts/{args.dataset}.json', 'r') as f:
        all_prompts_gpt = json.load(f)

    with open(f'../source/prompts/{args.dataset}_part_names.json', 'r') as f:
        all_prompts_part_names = json.load(f)

    all_prompts_templates = copy.deepcopy(all_prompts_part_names)
    if USE_PART_NAMES_TEMPLATES is not None:
        for category in all_prompts_templates.keys():
            all_prompts_templates[category] = [USE_PART_NAMES_TEMPLATES.format(part_name=part_name, category=category) for part_name in all_prompts_templates[category]]

    all_prompts = {
        # 'part_names': all_prompts_part_names,
        # 'templates': all_prompts_templates,
        'gpt': all_prompts_gpt,
    }

    # Load orientations
    orientations = {}
    for k, orientation in zip(NUM_OF_VIEWS, ORIENTATIONS):
        orientations[k] = np.load(f'orientations/{orientation}')
    # orientations = np.load(args.orientations)

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
        device=device
    )

    print('--> Saving results')
    save_results(segmentation_metrics)
