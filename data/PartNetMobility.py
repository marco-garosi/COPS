import torch
from torch.utils.data import Dataset
import numpy as np
import json

from collections import Counter
import os
import glob

from tqdm import tqdm

class PartNetMobility_Part(Dataset):
    """
    PartNet Mobility dataset, with annotations on parts
    """
    def __init__(self, base_dir, split=None):
        # Root directory
        self.base_dir = base_dir

        # Category annotation for each object
        self.object_id2category = self.get_obj_category_annotation()
        
        # category2id
        self.category2id = {class_: i for i, class_ in enumerate(sorted(list(set(self.object_id2category.values()))))}
        # id2category
        self.id2category = {v: k for k, v in self.category2id.items()}
        
        # File with part segmentation labels
        self.label_path = 'sample-points-all-label-10000.txt'

        # Data (list of available objects, represented by their unique id)
        self.object_ids = [
            os.path.normpath(path).split(os.sep)[-3]
            for path in
            glob.glob(os.path.join(base_dir, '*/point_sample/sample-points-all-pts-nor-rgba-10000.txt'))
        ]

        # Cache for segmentation ids
        self.segmentation_part_ids = None

    def __len__(self):
        return len(self.object_ids)

    def __getitem__(self, idx):
        # Get the object id
        object_id = self.object_ids[idx]

        # Get the category
        category = self.category2id[self.object_id2category[int(object_id)]]

        # Construct the path to load the data
        path = os.path.join(self.base_dir, object_id, 'point_sample', 'sample-points-all-pts-nor-rgba-10000.txt')
        with open(path, 'r') as f:
            # Read the data
            data = f.readlines()

            # (#points, 10): x, y, z, nx, ny, nz, r, g, b, a
            # Where:
            # - (x, y, z) are the 3D coordinates
            # - (nx, ny, nz) are the normals
            # - (r, g, b, a) are the RGB and alpha channels
            point_cloud = torch.from_numpy(np.array([np.array(x.strip().split(' ')) for x in data]).astype(np.float32))
            point_cloud = point_cloud[:, [0, 1, 2, 6, 7, 8]]

            # Part annotations
            with open(os.path.join(os.path.dirname(path), self.label_path), 'r') as f_label:
                segmentation = f_label.readlines()
                segmentation = torch.from_numpy(np.array([x.strip() for x in segmentation]).astype(np.int32))

        return point_cloud, segmentation, category

    def get_obj_category_annotation(self):
        """
        For each object, get its category annotation in the form of a dictionary {id object: category}
        """

        return {
            # object id                                  : category annotation
            int(os.path.normpath(path).split(os.sep)[-3]): json.load(open(os.path.join(self.base_dir, os.path.normpath(path).split(os.sep)[-3], 'meta.json')))['model_cat']
            for path in glob.glob(os.path.join(self.base_dir, '*/point_sample/sample-points-all-pts-nor-rgba-10000.txt'))
        }

    def get_pcd_id(self, idx):
        return self.object_ids[idx]

    @property
    def class_ids(self):
        return self.id2category
    
    @property
    def part_ids(self):
        # If cache miss, first extract the segmentation part ids, construct the set
        # that represents them, and cache it before returning it
        # This dramatically speeds up subsequent accesses, thus amortizing the build cost
        if self.segmentation_part_ids is None:
            self.segmentation_part_ids = set()

            # Iterate over the whole dataset
            for path in glob.glob(os.path.join(self.base_dir, '*/point_sample/sample-points-all-pts-nor-rgba-10000.txt')):
                # Open the annotation file for segmentation and build the set of ids it contains,
                # then compute the union with part ids extracted from other point clouds
                with open(os.path.join(os.path.dirname(path), self.label_path), 'r') as f_label:
                    segmentation = f_label.readlines()
                    segmentation = np.array([x.strip() for x in segmentation])
                    self.segmentation_part_ids.update(set(segmentation))

        return self.segmentation_part_ids

    def get_number_of_parts(self, method='average'):
        """
        How many different parts each object has for each category
        :param method: How the value is computed (e.g. max number of parts).
            Default to average
        :return: Number of parts computed on a category level based on objects belonging to the
            category itself
        """

        assert method in ['average', 'max', 'min', 'custom']

        return None


class Preprocessed_PartNetMobility_Part(Dataset):
    def __init__(self, preprocessed_base_dir, base_dir, model, get_raw_data=True, split=None):
        self.dataset = PartNetMobility_Part(base_dir, split=split)
        self.preprocessed_base_dir = preprocessed_base_dir
        self.model = model
        self.get_raw_data = get_raw_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx, filter_background=True):
        preprocessed_folder = os.path.join(self.preprocessed_base_dir, self.dataset.get_pcd_id(idx))

        try:
            rendered_images = torch.load(os.path.join(preprocessed_folder, f'rendered_images.pt'))
        except:
            rendered_images = None
        try:
            mappings = torch.load(os.path.join(preprocessed_folder, f'mappings.pt'))
        except:
            mappings = None
        try:
            features = torch.load(os.path.join(preprocessed_folder, f'outputs_{self.model}.pt'))
        except:
            features = None
        try:
            feature_pcd_aggregated = torch.load(os.path.join(preprocessed_folder, f'feature_pcd_aggregated_{self.model}.pt'))
        except:
            feature_pcd_aggregated = None

        # Return the point cloud, segmentation labels, and category
        if self.get_raw_data:
            point_cloud, segmentation, category = self.dataset[idx]
            return point_cloud, segmentation, category, rendered_images, mappings, features, feature_pcd_aggregated

        return rendered_images, mappings, features, feature_pcd_aggregated

    def get_pcd_id(self, idx):
        return self.dataset.get_pcd_id(idx)

    @property
    def class_ids(self):
        return self.dataset.class_ids

    @property
    def part_ids(self):
        return self.dataset.part_ids


def preprocessed_partnet_mobility_collate_fn(batch):
    objects = torch.stack([el[0] for el in batch])
    segmentations = torch.stack([el[1] for el in batch])
    categories = torch.tensor([el[2] for el in batch])

    rendered_images = [el[3] for el in batch]
    mappings = [el[4] for el in batch]
    features = [el[5] for el in batch]
    feature_pcd_aggregated = [el[6] for el in batch]

    return objects, segmentations, categories, rendered_images, mappings, features, feature_pcd_aggregated
