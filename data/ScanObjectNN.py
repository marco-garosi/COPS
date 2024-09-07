import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from collections import Counter
import os


id2category = {
    0: 'bag',
    1: 'bin',
    2: 'box',
    3: 'cabinet',
    4: 'chair',
    5: 'desk',
    6: 'display',
    7: 'door',
    8: 'shelf',
    9: 'table',
    10: 'bed',
    11: 'pillow',
    12: 'sink',
    13: 'sofa',
    14: 'toilet',
}

part_id_remapping = {
    0: torch.arange(4),
    1: torch.tensor([0, 1, 2, -1, 3]),
    2: torch.arange(5),
    3: torch.arange(7),
    4: torch.tensor([0, -1, 1, 2, 3, 4]),
    5: torch.arange(4),
    6: torch.arange(3),
    7: torch.arange(3),
    8: torch.arange(4),
    9: torch.arange(3),
    10: torch.arange(3),
    11: torch.arange(2),
    12: torch.arange(4),
    13: torch.arange(5),
    14: torch.arange(6)
}

"""
The following part ids have been extracted automatically from code and reported
here for quick and easy access
"""
segmentation_part_ids = {0, 1, 2, 3, 4, 5, 6}

"""
Objects to skip: we are skipping some objects that are too large and do not fit into memory
"""
skip_objects = {
    'train': [],
    'val': [],
    'test': []
}



class ScanObjectNN_Part(Dataset):
    """
    ScanObjectNN dataset, with annotations on parts
    """

    def __init__(self, base_dir, filter_background=True, filter_background_by_zeros=False, remap_labels=True, split='train', filter_category=None):
        assert split in ['train', 'test'], '`split` should be either `train` or `test`'
        self.filter_background = filter_background
        self.filter_background_by_zeros = filter_background_by_zeros
        self.remap_labels = remap_labels

        self.base_dir = os.path.join(base_dir, 'object_dataset_complete_with_parts_')

        self.data = pd.read_csv(
            os.path.join(self.base_dir, 'split_new.txt'),
            sep='\t', header=None, names=['file', 'category', 'train']
        )
        self.data['train'] = self.data['train'] != 't'

        self.data = self.data[self.data['train'] if split == 'train' else ~self.data['train']].reset_index(drop=True)
        self.data.drop(skip_objects[split], inplace=True)

        if filter_category:
            self.data = self.data[self.data['category'] == filter_category]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        category = self.data.iloc[idx]['category']
        input_file = self.data.iloc[idx]['file'][:-4]

        point_cloud = np.fromfile(os.path.join(self.base_dir, id2category[category], input_file) + '.bin', dtype=np.float32)
        segmentation = np.fromfile(os.path.join(self.base_dir, id2category[category], input_file) + '_part.bin', dtype=np.float32)

        # Reshape
        point_cloud = point_cloud[1:].reshape((-1, 11))
        segmentation = segmentation[1:].reshape((-1, 2))

        # Filter background
        if self.filter_background:
            filtered_idx = np.intersect1d(np.intersect1d(np.where(point_cloud[:, -1] != 0)[0], np.where(point_cloud[:, -1] != 1)[0]), np.where(point_cloud[:, -1] != 2)[0])
            (values, counts) = np.unique(point_cloud[filtered_idx, -1], return_counts=True)
            max_ind = np.argmax(counts)
            idx = np.where(point_cloud[:, -1] == values[max_ind])[0]

            point_cloud = point_cloud[idx]
            segmentation = segmentation[idx]

        # Keep (x, y, z) and RGB information
        point_cloud = torch.tensor(point_cloud[:, [0, 1, 2, 6, 7, 8]])
        # Keep part label
        segmentation = torch.tensor(segmentation[:, -1], dtype=torch.int)

        if self.filter_background_by_zeros:
            point_cloud = point_cloud[segmentation != 0]
            segmentation = segmentation[segmentation != 0]

        if self.remap_labels:
            segmentation = part_id_remapping[category][segmentation]

        return point_cloud, segmentation, category

    def get_pcd_id(self, idx):
        return self.data.iloc[idx]['file'][:-4]

    @property
    def class_ids(self):
        return id2category

    @property
    def part_ids(self):
        return segmentation_part_ids

    def get_number_of_parts(self, method='average'):
        """
        How many different parts each object has for each category
        :param method: How the value is computed (e.g. max number of parts).
            Default to average
        :return: Number of parts computed on a category level based on objects belonging to the
            category itself
        """

        assert method in ['average', 'max', 'min', 'custom']

        if method == 'max':
            # return [4, 4, 5, 7, 5, 4, 3, 3, 4, 3, 3, 5, 4, 6, 6]
            return [4, 4, 5, 7, 5, 4, 3, 3, 4, 3, 3, 2, 4, 5, 6]
        if method == 'average':
            return [2, 2, 2, 3, 4, 4, 3, 2, 2, 3, 3, 2, 3, 5, 4]

        return None

    @property
    def index_start(self):
        return [0] * 15


class Preprocessed_ScanObjectNN_Part(Dataset):
    def __init__(self, preprocessed_base_dir, base_dir, model, get_raw_data=True, split='train'):
        self.dataset = ScanObjectNN_Part(base_dir, split=split)
        self.preprocessed_base_dir = preprocessed_base_dir
        self.model = model
        self.get_raw_data = get_raw_data

    def __len__(self):
        return len(self.dataset.data)

    def __getitem__(self, idx, filter_background=True):
        preprocessed_folder = os.path.join(self.preprocessed_base_dir, self.dataset.data.iloc[idx]['file'][:-4])

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

    @property
    def data(self):
        return self.dataset.data

    def get_pcd_id(self, idx):
        return self.data.iloc[idx]['file'][:-4]

    @property
    def class_ids(self):
        return id2category

    @property
    def part_ids(self):
        return segmentation_part_ids


def scanobjectnn_part_collate_fn(batch):
    objects = [el[0] for el in batch]
    segmentations = [el[1] for el in batch]
    categories = [el[2] for el in batch]

    return objects, segmentations, categories


def preprocessed_scanobjectnn_part_collate_fn(batch):
    objects = [el[0] for el in batch]
    segmentations = [el[1] for el in batch]
    categories = [el[2] for el in batch]

    rendered_images = [el[3] for el in batch]
    mappings = [el[4] for el in batch]
    features = [el[5] for el in batch]
    feature_pcd_aggregated = [el[6] for el in batch]

    return objects, segmentations, categories, rendered_images, mappings, features, feature_pcd_aggregated
