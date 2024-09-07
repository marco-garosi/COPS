import torch
from torch.utils.data import Dataset
import numpy as np
import h5py

from itertools import chain
import os
import glob
import json

from .PartNet import PartNet


id2category = {
    0: 'Chair',
    1: 'Dishwasher',
    2: 'Lamp',
    3: 'Table',
    4: 'Bowl',
    5: 'StorageFurniture',
    6: 'Faucet',
    7: 'Microwave',
    8: 'TrashCan',
    9: 'Laptop',
    10: 'Earphone',
    11: 'Display',
    12: 'Refrigerator',
    13: 'Bed',
    14: 'Knife',
    15: 'Bag',
    16: 'Vase',
    17: 'Bottle',
    18: 'Hat',
    19: 'Mug',
    20: 'Scissors',
    21: 'Clock',
    22: 'Door',
    23: 'Keyboard'
}

"""
The following part ids have been extracted automatically from code and reported
here for quick and easy access.
"""
segmentation_part_ids = set(range(18))


def load_partnet_sem_seg_data(file_path):
    with h5py.File(file_path, 'r') as f:
        point_cloud = f['data'][:]
        labels = f['label_seg'][:]

    # Folder structure is Category-<level>/file.h5, so extract category from it
    category = os.path.normpath(file_path).split(os.sep)[-2].split('-')[0]

    return point_cloud, labels, [category] * len(point_cloud)


class PartNetSemanticSegmentation(Dataset):
    """
    PartNet for Semantic Segmentation dataset
    """
    def __init__(self, base_dir, base_dir_all_annotations, level='1', split='train'):
        assert split in ['train', 'val', 'test'], '`split` should be either `train`, `val` or `test`'
        assert level in ['1', '2', '3'], 'level should be either `1` or `2` or `3`'
        self.split = split
        self.level = level

        self.partnet = PartNet(base_dir_all_annotations, split=split)

        self.base_dir = os.path.join(base_dir, 'sem_seg_h5')

        self.files = glob.glob(self.base_dir + '/**/{}*.h5'.format(split), recursive=True)
        # Select only the correct level of annotations
        self.files = [
            file for file in self.files
            if f'-{self.level}' in os.path.basename(os.path.dirname(file))
        ]

        self.category2id = {v: k for k, v in id2category.items()}

        # Cache files
        self.cache = {}
        self.cat_count = {}
        for file in self.files:
            point_cloud, labels, category = load_partnet_sem_seg_data(file)
            self.cache[file] = {
                'point_cloud': torch.tensor(point_cloud),
                'labels': torch.tensor(labels),
                'category': torch.tensor([self.category2id[c] for c in category])
            }

            category = self.cache[file]['category']
            self.cat_count[category[0].item()] = self.cat_count.get(category[0].item(), 0) + len(category)

        self.idx2file = list(chain.from_iterable([
            [file_idx] * len(cached['point_cloud'])
            for file_idx, cached in enumerate(self.cache.values())
        ]))

        self.point_cloud_counts = [0]
        for stored in self.cache.values():
            self.point_cloud_counts.append(len(stored['category']))
        self.point_cloud_cum_counts = np.cumsum(self.point_cloud_counts)

        self.displacement_category = np.cumsum([0] + list(self.cat_count.values()))
        cumsum = np.zeros(len(id2category), dtype=int)
        for idx, x in enumerate(self.cat_count.keys()):
            cumsum[x] = self.displacement_category[idx]
        self.displacement_category = cumsum

    def __len__(self):
        return len(self.idx2file)
        # return len(self.partnet)

    def __getitem__(self, idx):
        file = self.files[self.idx2file[idx]]
        local_idx = idx - self.point_cloud_cum_counts[self.idx2file[idx]]

        point_cloud = self.cache[file]['point_cloud'][local_idx]
        segmentation = self.cache[file]['labels'][local_idx]
        category = self.cache[file]['category'][local_idx].item()

        idx_in_category = idx - self.displacement_category[category]
        idx_partnet = idx_in_category + self.partnet.displacement_category[category]

        point_cloud = self.partnet[idx_partnet][0]

        return point_cloud, segmentation, category

    def get_pcd_id(self, idx):
        return f'{self.split}_{idx}'

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
            if self.level == '1':
                return [6, 3, 18, 11, 4, 7, 8, 3, 5, 3, 6, 3, 3, 4, 5, 4, 4, 6, 6, 4, 3, 6, 3, 3]
            elif self.level == '2':
                return [30, 5, 28, 42, 0, 19, 0, 5, 0, 0, 0, 0, 6, 10, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0]
            else:
                return [39, 7, 41, 51, 0, 24, 12, 6, 11, 0, 10, 4, 7, 15, 10, 0, 6, 9, 0, 0, 0, 11, 5, 0]

        return None

    @property
    def index_start(self):
        return [0] * 24


def preprocessed_partnet_sem_seg_collate_fn(batch):
    objects = torch.stack([el[0] for el in batch])
    segmentations = torch.stack([el[1] for el in batch])
    categories = torch.tensor([el[2] for el in batch])

    rendered_images = [el[3] for el in batch]
    mappings = [el[4] for el in batch]
    features = [el[5] for el in batch]
    feature_pcd_aggregated = [el[6] for el in batch]

    return objects, segmentations, categories, rendered_images, mappings, features, feature_pcd_aggregated
