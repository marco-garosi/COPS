import torch
from torch.utils.data import Dataset
import numpy as np
import json

from collections import Counter
import os
import glob

from tqdm import tqdm

class PartNetE(Dataset):
    """
    PartNetE dataset, with annotations on parts
    """
    def __init__(self, base_dir, semantic_segmentation=True, sum_one=False, split='test'):
        assert split in ['train', 'test'], '`split` should be either `train` or `test`'

        # Root directory
        self.base_dir = os.path.join(base_dir, split)

        # Semantic or Instance segmentation
        self.semantic_segmentation = semantic_segmentation

        # If True, segmentation will be incremented by one, so that it starts from 0
        self.sum_one = sum_one

        # Data (list of available objects, represented by their unique id)
        self.object_ids = [
            os.path.normpath(path).split(os.sep)[-1]
            for path in
            glob.glob(os.path.join(self.base_dir, '*'))
        ]

        # Category annotation for each object
        self.object_id2category = self.get_obj_category_annotation()

        # category2id
        self.category2id = {class_: i for i, class_ in enumerate(sorted(list(set(self.object_id2category.values()))))}
        # id2category
        self.id2category = {v: k for k, v in self.category2id.items()}

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
        path = os.path.join(self.base_dir, object_id, 'data.npz')
        data = np.load(path)

        # Build the data
        point_cloud = torch.from_numpy(np.concatenate([data['xyz'], data['rgb'] * 255.], axis=1)).float()
        segmentation = torch.from_numpy(data['labels_sem_seg' if self.semantic_segmentation else 'labels_ins_seg'].astype(np.int32))

        segmentation += self.sum_one

        return point_cloud, segmentation, category

    def get_obj_category_annotation(self):
        """
        For each object, get its category annotation in the form of a dictionary {id object: category}
        """

        return {
            # object id   : category annotation
            int(object_id): str(np.load(os.path.join(self.base_dir, object_id, 'data.npz'))['catagory'])
            for object_id in self.object_ids
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

            for object_id in self.object_ids:
                # Adding 1 to avoid the -1 part id
                segmentation = np.load(os.path.join(self.base_dir, object_id, 'data.npz'))['labels_sem_seg' if self.semantic_segmentation else 'labels_ins_seg']
                segmentation += self.sum_one
                self.segmentation_part_ids.update(set(np.unique(segmentation)))

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

        if method == 'max':
            return [2, 2, 2, 3, 2, 6, 2, 5, 3, 3, 4, 4, 3, 3, 2, 2, 4, 3, 3, 2, 5, 6, 4, 5, 4, 3, 3, 3, 2, 2, 3, 2, 4, 4, 3, 4, 3, 2, 7, 3, 4, 4, 3, 3, 2]

        # Based on PartSLIP
        if method == 'custom':
            return [1, 1, 1, 2, 1, 5, 1, 4, 2, 2, 3, 3, 2, 2, 1, 1, 3, 2, 2, 1, 4, 5, 3, 4, 3, 2, 2, 2, 1, 1, 2, 1, 3, 3, 2, 3, 2, 1, 6, 2, 3, 3, 2, 2, 1]

        return None

    @property
    def index_start(self):
        return [0 if self.sum_one else -1] * 45


def partnete_collate_fn(batch):
    objects = [el[0] for el in batch]
    segmentations = [el[1] for el in batch]
    categories = [el[2] for el in batch]

    return objects, segmentations, categories
