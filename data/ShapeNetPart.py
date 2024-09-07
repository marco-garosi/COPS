import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import ShapeNet


class ShapeNetPart(Dataset):
    """
    ShapeNetPart dataset, with annotations on parts
    Derived by Torch Geometric implementation, this is just a wrapper to conform
    to our shared API for all datasets, to ensure compatibility with all our
    code.
    """

    def __init__(self, base_dir, split='train'):
        assert split in ['train', 'val', 'trainval', 'test'], '`split` should be either `train`, `val`, `trainval`, or `test`'

        self.data = ShapeNet(root=base_dir, include_normals=False, split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Setting color to black (0, 0, 0) for all points, since this dataset
        # doesn't have color information, but it still has to conform
        # to our common API, which wants the point cloud to hold 6 values
        # for each point: (x, y, z, R, G, B)
        point_cloud = torch.cat([item.pos, torch.zeros_like(item.pos)], dim=1)

        return point_cloud, item.y, item.category.item()

    def get_pcd_id(self, idx):
        return str(idx)

    @property
    def class_ids(self):
        return {idx: v for idx, v in enumerate(self.data.categories)}

    @property
    def inverted_class_ids(self):
        return {v: idx for idx, v in enumerate(self.data.categories)}

    @property
    def part_ids(self):
        return set(part_id for part_ids in self.data.seg_classes.values() for part_id in part_ids)

    def get_number_of_parts(self, method='average'):
        """
        How many different parts each object has for each category
        :param method: How the value is computed (e.g. max number of parts).
            Default to average
        :return: Number of parts computed on a category level based on objects belonging to the
            category itself
        """

        assert method in ['average', 'max', 'min', 'custom']

        if method == 'average':
            return [4, 2, 2, 4, 3, 3, 3, 2, 3, 2, 6, 2, 3, 3, 3, 2]
        if method == 'max':
            return [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]

        return None


    @property
    def index_start(self):
        return [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
