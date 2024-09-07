import torch
from torch.utils.data import Dataset
from torch_geometric.io import read_obj
import os
import json
from sklearn.preprocessing import LabelEncoder


class FAUST(Dataset):
    """
    FAUST dataset, with annotations on parts
    Derived by SATR by Abdelreheem et al., it supports both fine-grained and coarse annotations
    on parts for segmentation purposes.
    """

    def __init__(self, base_dir, fine_grained_annotations=False, split='test'):
        assert split in ['test'], '`split` should be `test`'

        self.base_dir = base_dir

        # Read the list of meshes
        with open(os.path.join(base_dir, 'meshes.txt')) as fin:
            self.meshes = [el.strip() for el in fin.readlines() if len(el) > 0 and 'obj' in el]

        self.fine_grained_annotations = fine_grained_annotations

        # Load the ground truth
        if self.fine_grained_annotations:
            with open(os.path.join(base_dir, 'fine_grained_gt.json')) as fin:
                self.dataset_gt = json.load(fin)
        else:
            with open(os.path.join(base_dir, 'coarse_gt.json')) as fin:
                self.dataset_gt = json.load(fin)

        # Encode the labels
        # Compute the set of distinct (unique) labels and sort it, to ensure consistency
        # on every use of this class
        unique_annotations = sorted(list(set(
            part
            for annotations in [list(set(v)) for v in self.dataset_gt.values()]
            for part in annotations
        )))
        self.le = LabelEncoder()
        self.le.fit(unique_annotations)
        self.unique_annotations = self.le.transform(unique_annotations)

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):
        mesh_file = self.meshes[idx]
        item = read_obj(os.path.join(self.base_dir, 'scans', mesh_file))
        segmentation = self.dataset_gt[mesh_file.split('.')[0]]
        segmentation = torch.from_numpy(self.le.transform(segmentation))

        # Setting color to black (0, 0, 0) for all points, since this dataset
        # doesn't have color information, but it still has to conform
        # to our common API, which wants the point cloud to hold 6 values
        # for each point: (x, y, z, R, G, B)
        point_cloud = torch.cat([item.pos, torch.zeros_like(item.pos)], dim=1)

        return point_cloud, segmentation, 0

    def get_pcd_id(self, idx):
        return self.meshes[idx].split('.')[0]

    @property
    def class_ids(self):
        return {0: 'Human'}

    @property
    def part_ids(self):
        return self.unique_annotations

    def get_number_of_parts(self, method='average'):
        """
        How many different parts each object has for each category
        :param method: How the value is computed (e.g. max number of parts).
            Default to average
        :return: Number of parts computed on a category level based on objects belonging to the
            category itself
        """

        assert method in ['average', 'max', 'min', 'custom']

        if not self.fine_grained_annotations:
            return [4]

        return None

    @property
    def index_start(self):
        return [0]
