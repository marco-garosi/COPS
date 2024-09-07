import torch
from torch.utils.data import Dataset
import numpy as np
import h5py

from itertools import chain
import os
import glob
import json


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
Segmentation part id `538` seem to be missing (in training, val and test splits).
"""
segmentation_part_ids = {i for i in range(1, 598)}.difference({538})

def load_partnet_data(file_path):
    with h5py.File(file_path, 'r') as f:
        point_cloud = f['pts'][:]
        labels = f['label'][:]
        colors = f['rgb'][:]
        nor = f['nor'][:]
        opacity = f['opacity'][:]

    # Folder structure is Category/file.h5, so extract category from it
    category = os.path.normpath(file_path).split(os.sep)[-2]

    return point_cloud, labels, colors, nor, opacity, [category] * len(point_cloud)


class PartNet(Dataset):
    """
    PartNet dataset
    """
    def __init__(self, base_dir, split='train'):
        assert split in ['train', 'val', 'test'], '`split` should be either `train`, `val` or `test`'
        self.split = split

        self.base_dir = os.path.join(base_dir, 'ins_seg_h5')

        self.files = glob.glob(self.base_dir + '/**/{}*.h5'.format(split), recursive=True)
        self.files_json = glob.glob(self.base_dir + '/**/{}*.json'.format(split), recursive=True)

        self.point_cloud_counts = []
        for file in self.files_json:
            with open(file, 'r') as f:
                self.point_cloud_counts.append(len(json.load(f)))

        # Build a mapping form an index to the corresponding file
        # This supports the `__getitem__` function
        self.idx2file = list(chain.from_iterable([
            [file_idx] * length
            for file_idx, length in enumerate(self.point_cloud_counts)
        ]))

        # Cache files
        self.cache = {}

        self.point_cloud_counts = [0] + self.point_cloud_counts
        self.point_cloud_cum_counts = np.cumsum(self.point_cloud_counts)
        self.category2id = {v: k for k, v in id2category.items()}

        if self.split == 'test':
            self.displacement_category = np.array([189, 1504, 2135, 3205, 150, 2754, 1850, 2636, 4873, 2554, 1797, 1555, 2710, 29, 2058, 0, 4936, 66, 1982, 2675, 2741, 1406, 1746, 2027])
        else:
            self.displacement_category = None

    def __len__(self):
        return len(self.idx2file)

    def __getitem__(self, idx):
        file = self.files[self.idx2file[idx]]

        # Load file into cache if it has not yet been loaded
        if file not in self.cache.keys():
            point_cloud, labels, colors, nor, opacity, category = load_partnet_data(file)
            self.cache[file] = {
                'point_cloud': torch.cat([torch.tensor(point_cloud), torch.tensor(colors)], dim=-1),
                'labels': torch.tensor(labels),
                'nor': torch.tensor(nor),
                'opacity': torch.tensor(opacity),
                'category': category,
            }

        # Get the index local to the file that contains the point cloud
        local_idx = idx - self.point_cloud_cum_counts[self.idx2file[idx]]

        point_cloud = self.cache[file]['point_cloud'][local_idx]
        segmentation = self.cache[file]['labels'][local_idx]
        category = self.category2id[self.cache[file]['category'][local_idx]]

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

        return None


class Preprocessed_PartNet(Dataset):
    def __init__(self, preprocessed_base_dir, base_dir, model, get_raw_data=True, split='train'):
        self.dataset = PartNet(base_dir, split=split)
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
        return id2category

    @property
    def part_ids(self):
        return segmentation_part_ids

def preprocessed_partnet_collate_fn(batch):
    objects = torch.stack([el[0] for el in batch])
    segmentations = torch.stack([el[1] for el in batch])
    categories = torch.tensor([el[2] for el in batch])

    rendered_images = [el[3] for el in batch]
    mappings = [el[4] for el in batch]
    features = [el[5] for el in batch]
    feature_pcd_aggregated = [el[6] for el in batch]

    return objects, segmentations, categories, rendered_images, mappings, features, feature_pcd_aggregated
