import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import knn
import torchvision.transforms.functional as TF

from rendering.render_and_map import render_with_orientations, manual_projection
from rendering.realistic_projection import *
from ..point_cloud_utils.feature_interpolation import interpolate_feature_map
from ..point_cloud_utils.feature_aggregation import aggregate_features


class FeatureExtractor(nn.Module):
    def __init__(self,
                 image_processor, model, model_name,
                 rotations, translations,
                 use_colorized_renders=True, point_size=0.01,
                 canvas_width=224, canvas_height=224,
                 subsample_point_cloud=None,
                 early_subsample=True,
                 use_manual_projection=False,
                 improved_depth_maps=False,
                 perspective=True,
                 layer_features=None
                 ):
        super().__init__()

        if use_colorized_renders and use_manual_projection:
            raise Exception('Use manual projection is available only when not using colorized renders')

        self.image_processor = image_processor
        self.model = model
        self.model_name = model_name
        self.layer_features = layer_features

        self.rotations = rotations
        self.translations = translations
        self.use_colorized_renders = use_colorized_renders
        self.improved_depth_maps = improved_depth_maps
        self.use_manual_projection = use_manual_projection
        self.point_size = point_size
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.subsample_point_cloud = subsample_point_cloud
        self.early_subsample = early_subsample
        self.perspective = perspective

    def forward(self, point_cloud, segmentation, return_outputs=False):
        device = self.model.device

        if self.early_subsample:
            point_cloud, segmentation = self.subsample(point_cloud, segmentation)

        rendered_images, depth_maps, mappings = render_with_orientations(
            point_cloud,
            rotations=self.rotations, translations=self.translations,
            backend='pytorch3d',
            point_size=self.point_size,
            canvas_width=self.canvas_width, canvas_height=self.canvas_height,
            perspective=self.perspective,
            device=device
        )

        if not self.early_subsample:
            point_cloud, segmentation = self.subsample(point_cloud, segmentation, mappings, False)#self.early_subsample)

            # Double mappings to get more precise mappings with a small point size
            _, _, mappings = render_with_orientations(
                point_cloud,
                rotations=self.rotations, translations=self.translations,
                backend='pytorch3d',
                point_size=0.01,
                canvas_width=self.canvas_width, canvas_height=self.canvas_height,
                perspective=self.perspective,
                device=device
            )

        rendered_images = rendered_images.cpu()

        if self.use_colorized_renders:
            inputs_images = rendered_images
        else:
            if self.improved_depth_maps:
                distances = depth_maps - depth_maps[depth_maps != -1].min()
                distances /= distances[depth_maps != -1].max()
                distances[depth_maps == -1] = 1.
                distances = TF.gaussian_blur(distances, 7)
                distances = TF.adjust_gamma(distances.unsqueeze(1), 1.5).squeeze(1)
                inputs_images = (torch.stack([distances] * 3).permute(1, 2, 3, 0) * 255).int()
            else:
                inputs = (depth_maps - depth_maps.view(len(self.rotations), -1).min(dim=-1).values.unsqueeze(-1).unsqueeze(-1))
                inputs /= inputs.view(len(self.rotations), -1).max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
                inputs_images = 255 - (torch.stack([inputs] * 3).permute(1, 2, 3, 0) * 255).int()

        inputs = self.image_processor(images=inputs_images, text=[''], return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        del inputs
        if 'clip' in self.model_name:
            outputs = outputs.vision_model_output

        if self.layer_features is not None:
            outputs = outputs.hidden_states[self.layer_features]
        final_features = interpolate_feature_map(outputs, self.canvas_width, self.canvas_height)
        if not return_outputs:
            del outputs

        feature_pcd_aggregated = aggregate_features(final_features, mappings, point_cloud, device=device, interpolate_missing=True)

        del final_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if return_outputs:
            return point_cloud, segmentation, inputs_images, mappings, feature_pcd_aggregated, outputs
        else:
            return point_cloud, segmentation, inputs_images, mappings, feature_pcd_aggregated

    def subsample(self, point_cloud, segmentation, mappings=None, early=True):
        np.random.seed(0)
        if self.subsample_point_cloud is not None and len(point_cloud) > self.subsample_point_cloud:
            subsampled_point_indices = torch.from_numpy(
                np.random.choice(np.arange(len(point_cloud)), self.subsample_point_cloud, replace=False))

            point_cloud_reduced = point_cloud[subsampled_point_indices]

            # Map each point to a superpoint
            if not early:
                point_to_superpoint_mapping = knn(
                    point_cloud_reduced[:, :3], point_cloud[:, :3], 1
                )[1].int()

                mappings[mappings != -1] = point_to_superpoint_mapping[mappings[mappings != -1]]
                del point_to_superpoint_mapping

            # Perform subsampling
            point_cloud = point_cloud_reduced
            segmentation = segmentation[subsampled_point_indices]

        return point_cloud, segmentation


class FeatureExtractorPointCLIPv2(nn.Module):
    def __init__(self,
                 model,
                 canvas_width=224, canvas_height=224,
                 subsample_point_cloud=2048
                 ):
        super().__init__()

        self.model = model

        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.subsample_point_cloud = subsample_point_cloud

        self.store = {}

        self.projector = Realistic_Projection()

    def forward(self, point_cloud, segmentation, text_features, other_features=None, vweights=None, return_logits=False, reuse_renders=False):
        device = point_cloud.device

        if vweights is None:
            vweights = torch.ones(10).to(device).view(1, -1, 1, 1)

        # if not reuse_renders:
        point_cloud, segmentation, other_features = self.subsample(point_cloud, segmentation, other_features)

        # Project
        inputs_images, is_seen, point_loc_in_images = manual_projection(point_cloud[:, :3].unsqueeze(0), self.projector)

        # Extract features
        with torch.no_grad():
            images_cls, images_feat = self.model.encode_image(inputs_images.to(device))
            images_feat = images_feat / images_feat.norm(dim=-1, keepdim=True)
            B, L, C = images_feat.shape
            images_feat = images_feat.reshape(B, 14, 14, C).permute(0, 3, 1, 2).unsqueeze(0)
            images_feat = images_feat.reshape(-1, len(inputs_images), 196, 512)
        b, nv, hw, c = images_feat.size(0), images_feat.size(1), images_feat.size(2), images_feat.size(3)
        images_feat = images_feat.reshape(b * nv, hw, c)  # (batch, views, 224, 224) -> (batch * view, 224, 224)

        # (#prompts, dim emb)
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(0)  # (1, #prompts, dim emb)

        all_point_logits = []
        all_seg_pred = []
        for text_feat in text_features:
            # Logits
            logits = 100. * images_feat.half() @ text_feat.half().t()
            output = logits.float().permute(0, 2, 1).reshape(-1, len(text_feat), int(hw ** 0.5), int(hw ** 0.5))

            upsample = torch.nn.Upsample(size=224, mode='bilinear') # nearest, bilinear
            avgpool = torch.nn.AvgPool2d(6, 1, 0)
            padding = torch.nn.ReplicationPad2d([2, 3, 2, 3])

            output = avgpool(padding(output))
            output = upsample(output)

            # Back-projection
            point_in_view = torch.repeat_interleave(torch.arange(0, len(inputs_images))[:, None], len(point_cloud)).view(-1, ).long().to(device)
            yy = point_loc_in_images[:, :, 0].view(-1).long()
            xx = point_loc_in_images[:, :, 1].view(-1).long()

            point_logits = output[point_in_view, :, yy, xx]
            point_logits = point_logits.view(b, nv, len(point_cloud), len(text_feat))

            is_seen = is_seen.reshape(b, nv, len(point_cloud), 1)

            point_logits = torch.sum(point_logits * is_seen * vweights, dim=1)
            seg_pred = torch.topk(point_logits, k=1, dim=-1)[1].squeeze()

            all_point_logits.append(point_logits)
            all_seg_pred.append(seg_pred)

        # Squeezing first dimension to ensure backwards compatibility with previous code
        # So that if len(text_features.shape) == 2, predictions are just one-dimensional
        # instead of two-dimensional
        point_logits = torch.stack(all_point_logits).squeeze(0)
        seg_pred = torch.stack(all_seg_pred).squeeze(0)

        if return_logits:
            return point_cloud, segmentation, other_features, inputs_images, seg_pred, point_logits
        else:
            return point_cloud, segmentation, other_features, inputs_images, seg_pred

    def subsample(self, point_cloud, segmentation, other_features):
        np.random.seed(0)
        if self.subsample_point_cloud is not None and len(point_cloud) > self.subsample_point_cloud:
            subsampled_point_indices = torch.from_numpy(np.random.choice(np.arange(len(point_cloud)), self.subsample_point_cloud, replace=False))

            # Perform subsampling
            point_cloud = point_cloud[subsampled_point_indices]
            segmentation = segmentation[subsampled_point_indices]

            if other_features is not None:
                other_features = [o[subsampled_point_indices] for o in other_features]

        return point_cloud, segmentation, other_features


class FeatureExtractorCLIP(nn.Module):
    def __init__(self,
                 model,
                 rotations, translations,
                 use_colorized_renders=True, point_size=0.01,
                 canvas_width=224, canvas_height=224,
                 subsample_point_cloud=None,
                 early_subsample=True,
                 improved_depth_maps=False,
                 perspective=True
                 ):
        super().__init__()

        self.model = model

        self.rotations = rotations
        self.translations = translations
        self.use_colorized_renders = use_colorized_renders
        self.point_size = point_size
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.subsample_point_cloud = subsample_point_cloud
        self.early_subsample = early_subsample
        self.improved_depth_maps = improved_depth_maps
        self.perspective = perspective

    def forward(self, point_cloud, segmentation, text_feat, other_features=None, vweights=None, return_logits=False):
        device = point_cloud.device

        if self.early_subsample:
            point_cloud, segmentation, other_features = self.subsample(point_cloud, segmentation, other_features)

        rendered_images, depth_maps, mappings = render_with_orientations(
            point_cloud,
            rotations=self.rotations, translations=self.translations,
            backend='pytorch3d',
            point_size=self.point_size,
            canvas_width=self.canvas_width, canvas_height=self.canvas_height,
            perspective=self.perspective,
            device=device
        )

        if not self.early_subsample:
            point_cloud, segmentation, other_features = self.subsample(point_cloud, segmentation, other_features)

            # Double mappings to get more precise mappings with a small point size
            _, _, mappings = render_with_orientations(
                point_cloud,
                rotations=self.rotations, translations=self.translations,
                backend='pytorch3d',
                point_size=0.01,
                canvas_width=self.canvas_width, canvas_height=self.canvas_height,
                perspective=self.perspective,
                device=device
            )

        rendered_images = rendered_images.cpu()

        if self.use_colorized_renders:
            inputs_images = rendered_images
        else:
            if self.improved_depth_maps:
                distances = depth_maps - depth_maps[depth_maps != -1].min()
                distances /= distances[depth_maps != -1].max()
                distances[depth_maps == -1] = 1.
                distances = TF.gaussian_blur(distances, 7)
                distances = TF.adjust_gamma(distances.unsqueeze(1), 1.5).squeeze(1)
                inputs_images = (torch.stack([distances] * 3).permute(1, 2, 3, 0) * 255).int()
            else:
                inputs = (depth_maps - depth_maps.view(len(self.rotations), -1).min(dim=-1).values.unsqueeze(-1).unsqueeze(-1))
                inputs /= inputs.view(len(self.rotations), -1).max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
                inputs_images = 255 - (torch.stack([inputs] * 3).permute(1, 2, 3, 0) * 255).int()
        inputs_images_clip = inputs_images.permute(0, 3, 1, 2) / 255.

        # Extract features
        with torch.no_grad():
            images_cls, images_feat = self.model.encode_image(inputs_images_clip.to(device))
            images_feat = images_feat / images_feat.norm(dim=-1, keepdim=True)
            B, L, C = images_feat.shape
            images_feat = images_feat.reshape(B, 14, 14, C).permute(0, 3, 1, 2).unsqueeze(0)
            images_feat = images_feat.reshape(-1, len(inputs_images_clip), 196, 512)
        b, nv, hw, c = images_feat.size(0), images_feat.size(1), images_feat.size(2), images_feat.size(3)
        images_feat = images_feat.reshape(b * nv, hw, c)  # (batch, views, 224, 224) -> (batch * view, 224, 224)

        # Logits
        logits = 100. * images_feat.half() @ text_feat.half().t()
        output = logits.float().permute(0, 2, 1).reshape(-1, len(text_feat), int(hw ** 0.5), int(hw ** 0.5))

        upsample = torch.nn.Upsample(size=224, mode='bilinear')  # nearest, bilinear
        avgpool = torch.nn.AvgPool2d(6, 1, 0)
        padding = torch.nn.ReplicationPad2d([2, 3, 2, 3])

        output = avgpool(padding(output))
        output = upsample(output)

        # Back-projection
        point_logits = aggregate_features(output.permute(0, 2, 3, 1), mappings, point_cloud, device=device, interpolate_missing=True)
        seg_pred = torch.topk(point_logits, k=1, dim=-1)[1].squeeze()

        if return_logits:
            return point_cloud, segmentation, other_features, inputs_images, seg_pred, point_logits
        else:
            return point_cloud, segmentation, other_features, inputs_images, seg_pred

    def subsample(self, point_cloud, segmentation, other_features):
        np.random.seed(0)
        if self.subsample_point_cloud is not None and len(point_cloud) > self.subsample_point_cloud:
            subsampled_point_indices = torch.from_numpy(np.random.choice(np.arange(len(point_cloud)), self.subsample_point_cloud, replace=False))

            # Perform subsampling
            point_cloud = point_cloud[subsampled_point_indices]
            segmentation = segmentation[subsampled_point_indices]

            if other_features is not None:
                other_features = other_features[:, subsampled_point_indices]

        return point_cloud, segmentation, other_features


class FeatureExtractor_PointCLIPv2DINO(nn.Module):
    def __init__(self,
                 image_processor, model, model_name,
                 rotations, translations,
                 use_colorized_renders=True, point_size=0.01,
                 canvas_width=224, canvas_height=224,
                 subsample_point_cloud=None,
                 early_subsample=True,
                 improved_depth_maps=True,
                 perspective=True
                 ):
        super().__init__()

        self.image_processor = image_processor
        self.model = model
        self.model_name = model_name

        self.rotations = rotations
        self.translations = translations
        self.use_colorized_renders = use_colorized_renders
        self.improved_depth_maps = improved_depth_maps
        self.point_size = point_size
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.subsample_point_cloud = subsample_point_cloud
        self.early_subsample = early_subsample
        self.perspective = perspective
        self.device = model.device
        
    def forward(self, point_cloud, segmentation):
        
        if self.early_subsample:
            point_cloud, segmentation = self.subsample(point_cloud, segmentation)

        rendered_images, depth_maps, mappings = render_with_orientations(
            point_cloud,
            rotations=self.rotations, translations=self.translations,
            backend='pytorch3d',
            point_size=self.point_size,
            canvas_width=self.canvas_width, canvas_height=self.canvas_height,
            perspective=self.perspective,
            device=self.device
        )

        if not self.early_subsample:
            point_cloud, segmentation = self.subsample(point_cloud, segmentation, mappings, self.early_subsample)

        rendered_images = rendered_images.cpu()
        mappings = mappings.to(self.device)
        point_cloud = point_cloud.to(self.device)

        if self.use_colorized_renders:
            inputs_images = rendered_images
        else:
            # Depth Maps
            if self.improved_depth_maps:
                distances = depth_maps - depth_maps[depth_maps != -1].min()
                distances /= distances[depth_maps != -1].max()
                distances[depth_maps == -1] = 1.
                distances = TF.gaussian_blur(distances, 7)
                distances = TF.adjust_gamma(distances.unsqueeze(1), 1.5).squeeze(1)
                inputs_images = (torch.stack([distances] * 3).permute(1, 2, 3, 0) * 255).int()
            else:
                inputs = (depth_maps - depth_maps.view(len(self.rotations), -1).min(dim=-1).values.unsqueeze(-1).unsqueeze(-1))
                inputs /= inputs.view(len(self.rotations), -1).max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
                inputs_images = 255 - (torch.stack([inputs] * 3).permute(1, 2, 3, 0) * 255).int()


        inputs = self.image_processor(inputs_images, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        final_features = interpolate_feature_map(outputs, self.canvas_width, self.canvas_height)
        del outputs

        feature_pcd_aggregated = aggregate_features(final_features, mappings, point_cloud, device=self.device, interpolate_missing=True)

        del final_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return point_cloud, segmentation, rendered_images, depth_maps, mappings, feature_pcd_aggregated

    def subsample(self, point_cloud, segmentation, mappings=None, early=True):
        np.random.seed(0)
        if self.subsample_point_cloud is not None and len(point_cloud) > self.subsample_point_cloud:
            subsampled_point_indices = torch.from_numpy(
                np.random.choice(np.arange(len(point_cloud)), self.subsample_point_cloud, replace=False))

            point_cloud_reduced = point_cloud[subsampled_point_indices]
            
            mappings = mappings.to(self.device)

            # Map each point to a superpoint
            if not early:
                point_to_superpoint_mapping = knn(
                    point_cloud_reduced[:, :3], point_cloud[:, :3], 1
                )[1].int()
                
                point_to_superpoint_mapping = point_to_superpoint_mapping.to(self.device)

                mappings[mappings != -1] = point_to_superpoint_mapping[mappings[mappings != -1]]
                del point_to_superpoint_mapping

            # Perform subsampling
            point_cloud = point_cloud_reduced
            segmentation = segmentation[subsampled_point_indices]

        return point_cloud, segmentation