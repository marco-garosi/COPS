import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import fps, knn


class GeometricAwareFeatureAggregation(nn.Module):
    def __init__(
            self,
            levels=torch.tensor([256, 256]),
            neighborhood=torch.tensor([10, 90]),
            aggregation_type=['xyz', 'sem'],
            weight_by_distance=torch.tensor([False, False]),
            upsample_to_original_size=True
    ):
        super().__init__()

        assert len(levels) == len(neighborhood) == len(aggregation_type) == len(
            weight_by_distance), '`levels`, `neighborhood`, `aggregation_type`, and `weight_by_distance` must be same length'

        self.levels = levels
        self.neighborhood = neighborhood
        self.aggregation_type = aggregation_type
        self.weight_by_distance = weight_by_distance
        self.upsample_to_original_size = upsample_to_original_size

    def forward(self, point_cloud, features, per_point_vectors):
        superpoint_features = features
        points = point_cloud[:, :3]
        labels = per_point_vectors.float()
        for level, neighbors, aggregation_type, weight_by_distance in zip(self.levels, self.neighborhood,
                                                                          self.aggregation_type,
                                                                          self.weight_by_distance):
            # Extract superpoints
            index = fps(points, ratio=level / len(points))
            superpoints = points[index]
            labels = labels[:, index]

            # Aggregate features based either on xyz space or on feature space
            # Also, features might be weighted according to their distance
            # from the superpoint
            if aggregation_type == 'xyz':
                sm = superpoints
                lg = points
            else:
                sm = superpoint_features[index]
                lg = superpoint_features

            if weight_by_distance:
                values, point_to_superpoint_mapping = torch.cdist(sm, lg).topk(neighbors, largest=False, dim=-1)
                weights = F.softmin(values, dim=-1)

                superpoint_features = (weights.unsqueeze(-1) * superpoint_features[point_to_superpoint_mapping]).sum(
                    dim=1)
            else:
                point_to_superpoint_mapping = knn(
                    lg, sm, neighbors
                )[1].view(len(superpoints), -1)
                superpoint_features = superpoint_features[point_to_superpoint_mapping].mean(dim=1)
            points = superpoints

            # Upsample
            if self.upsample_to_original_size:
                index = self.upsample(superpoints, point_cloud[:, :3])
                points = point_cloud[:, :3]
                superpoint_features = superpoint_features[index].mean(dim=1)
                labels = labels[:, index].mean(dim=-1)

        return points, superpoint_features, labels

    def upsample(self, superpoints, points, closest_superpoints=1):
        point_to_superpoint_mapping = knn(
            superpoints, points, closest_superpoints
        )[1].view(len(points), -1)
        return point_to_superpoint_mapping

    def to(self, device):
        self.levels = self.levels.to(device)
        self.neighborhood = self.neighborhood.to(device)
        return self
