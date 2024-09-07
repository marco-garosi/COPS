import torch

from source.point_cloud_utils.kmeans import kmeans
from source.point_cloud_utils.spectral_clustering import spectral_clustering_cluster_qr
from source.point_cloud_utils.map_cluster_ids_to_gt import map_cluster_ids_to_gt
from source.metrics.compute_metrics import compute_all_metrics
from source.metrics.compute_metrics import preprocess_ground_truth


def get_best_cluster_assignment(feature_pcd_aggregated, iterations, k_means_iterations, k_means_clusters, segmentation_for_hungarian, metric='jaccard_index', return_centroids=False, compute_sse=False, segmentation_gt=None, num_classes=None, return_before_hungarian=False, device='cpu'):
    """
    Cluster the point cloud `iterations` times using k-means and keep the best result
    based on the ground truth and the given metric.

    :param feature_pcd_aggregated: tensor of shape (#points, embedding dimensionality)
    :param iterations: how many times to repeat clustering
    :param k_means_iterations: how many iterations for k-means
    :param k_means_clusters: how many clusters for k-means
    :param segmentation: ground truth segmentation
    :param metric: metric to use to determine the "best" assignment
    :param return_centroids: whether to return the centroids of the clusters or not.
        Default to False
    :param compute_sse: whether to compute the SSE across all clusters or not.
        Default to False
    :param num_classes: how many classes the ground truth has. Set to None to automatically estimate it.
        Default to None
    :param device: where to perform computation on
    :return: the best cluster assignments, and all the metrics for the assignment
    """

    if segmentation_gt is None:
        segmentation_gt = segmentation_for_hungarian

    best_result = float('-inf')
    best_cluster_assignment = None
    best_cluster_assignment_unmapped = None
    best_metrics = None
    best_centroids = None
    best_sse = None

    for i in range(iterations):
        cluster_assignment, centroids = kmeans(k_means_clusters, feature_pcd_aggregated, iterations=k_means_iterations, return_centroids=True)[:len(segmentation_gt)]
        cluster_assignment_unmapped = torch.clone(cluster_assignment)

        # Compute SSE
        sse = 0
        if compute_sse:
            for cluster_id in range(k_means_clusters):
                distances = torch.cdist(feature_pcd_aggregated[cluster_assignment == cluster_id], centroids[cluster_id].unsqueeze(0))
                sse += distances.sum().item() ** 2

        # Map to the ground truth
        _, ids_pred_h, _ = map_cluster_ids_to_gt(cluster_assignment[:len(segmentation_for_hungarian)], segmentation_for_hungarian.to(device))
        ids_pred_h = ids_pred_h.to(device).int()
        cluster_assignment = ids_pred_h[cluster_assignment]

        metrics = compute_all_metrics(cluster_assignment[:len(segmentation_gt)], segmentation_gt, num_classes=num_classes, device=device)

        if metrics[metric] > best_result:
            best_result = metrics[metric]
            best_cluster_assignment = cluster_assignment
            best_cluster_assignment_unmapped = cluster_assignment_unmapped
            best_metrics = metrics
            best_centroids = centroids

            # Update SSE if necessary
            if compute_sse:
                best_sse = sse

    # else not necessary, but to make code more readable
    if return_centroids or compute_sse:
        return best_cluster_assignment, best_metrics, {
            'centroids': best_centroids,
            'sse': best_sse,
            'cluster_assignment_unmapped': best_cluster_assignment_unmapped,
        }
    elif return_before_hungarian:
        return best_cluster_assignment, best_metrics, {
            'cluster_assignment_unmapped': best_cluster_assignment_unmapped,
        }
    else:
        return best_cluster_assignment, best_metrics


def get_best_cluster_assignment_spectral(feature_pcd_aggregated, k_means_iterations, k_means_clusters, segmentation, device='cpu'):
    """
    Cluster the point cloud using spectral clustering and k-means and keep the best result
    based on the ground truth and the given metric.

    :param feature_pcd_aggregated: tensor of shape (#points, embedding dimensionality)
    :param iterations: how many times to repeat clustering
    :param k_means_iterations: how many iterations for k-means
    :param k_means_clusters: how many clusters for k-means
    :param segmentation: ground truth segmentation
    :param metric: metric to use to determine the "best" assignment
    :param device: where to perform computation on
    :return: the best cluster assignments, and all the metrics for the assignment
    """

    segmentation, _ = preprocess_ground_truth(segmentation)

    cluster_assignment = spectral_clustering_cluster_qr(k=k_means_clusters, features=feature_pcd_aggregated, k_means_iterations=k_means_iterations)

    # Map to the ground truth
    _, ids_pred_h, _ = map_cluster_ids_to_gt(cluster_assignment, segmentation.to(device))
    ids_pred_h = ids_pred_h.to(device).int()
    cluster_assignment = ids_pred_h[cluster_assignment]

    metrics = compute_all_metrics(cluster_assignment, segmentation, device=device)

    return cluster_assignment, metrics


def aggregate_cluster_id_topk_mode(cluster_id_kmeans, cluster_id_spectral, point_cloud, topk=20):
    
    cluster_id = torch.zeros_like(cluster_id_kmeans)
    
    for i in range(len(cluster_id_kmeans)):
        
        if cluster_id_kmeans[i] == cluster_id_spectral[i]:
            cluster_id[i] = cluster_id_kmeans[i]
        else:
            # find the k-nn of the point in the 3d domain
            # and assign the cluster id of the k-nn
            # to the point
            point = point_cloud[i, :3]
            point = point.unsqueeze(0)
            point = point.repeat(len(point_cloud), 1)
            dist = torch.norm(point - point_cloud[:, :3], dim=1)
            dist = dist.unsqueeze(1)
            dist = dist.repeat(1, 2)
            
            cluster_id_temp = torch.cat((cluster_id_kmeans.unsqueeze(1), cluster_id_spectral.unsqueeze(1)), dim=1).to(dist.device)
            cluster_id_temp = torch.cat((cluster_id_temp, dist), dim=1).to(dist.device)
            cluster_id_temp = cluster_id_temp[cluster_id_temp[:, 2].argsort()]
            
            # get the mode of the cluster ids of the k-nn
            cluster_id_temp = cluster_id_temp[:topk, :]
            
            # get the mode of the topk cluster ids
            cluster_id_temp = cluster_id_temp[:, 0]
            
            # assign the mode to the point
            cluster_id[i] = cluster_id_temp.mode()[0].item()
    
    return cluster_id


def kmeans_sim(k, vectors, iterations=30, return_centroids=False):
    """
    Fast k-means implementation which works even on GPU with cosine similarity.

    :param k: number of clusters
    :param vectors: tensor with all the vectors to be clustered,
        shape is (#vectors, dimensionality)
    :param iterations: how many iterations the algorithm should be run for.
        This is a hard constraint: there are no early-stopping conditions in
        case of premature convergence, so the algorithm will run for the
        specified number of iterations in any case
    :param return_centroids: whether to return the centroids of the clusters or not.
        Default to False
    :return: index of cluster for each vector in vectors. Output has
        shape (#vectors)
    """

    # Initialize centroids randomly
    centroids = vectors[torch.randperm(vectors.size(0))[:k]]

    # Initialize cluster_assignment
    cluster_assignment = None

    for _ in range(iterations):
        # Calculate similarities from centroids
        similarities = torch.mm(vectors, centroids.t())

        # Assign each point to the nearest centroid
        cluster_assignment = torch.argmin(similarities, dim=1)

        # Update centroids by taking the mean of assigned points
        for cluster in range(k):
            cluster_points = vectors[cluster_assignment == cluster]
            if len(cluster_points) > 0:
                centroids[cluster] = cluster_points.mean(dim=0)

    # else not necessary, but to make code more readable
    if return_centroids:
        return cluster_assignment, centroids
    else:
        return cluster_assignment
    
    
def co_clustering_3d_distances_and_features_similarities(coords_3d, features, k_mode_closest_neighbors: int = 30, iterations: int = 30, k: int = 10, device='cpu'):
    """
    Co-clustering of 3D points and features.
    
    :param coords_3d: 3D coordinates of the points
    :param features: features of the points
    :param k_mode_closest_neighbors: number of closest neighbors to consider for the mode
    :param iterations: number of iterations for k-means
    :param k: number of clusters for k-means
    :return: cluster_assignment_3d: cluster assignment of the 3D points
    """

    # run k-means and get the cluster assignments
    cluster_assignment_3d = kmeans(k, coords_3d, iterations=iterations)
    cluster_assignment_features = kmeans(k, features, iterations=iterations)
    cluster_assignment_features_only_for_points = cluster_assignment_features[:len(cluster_assignment_3d)]

    # map the cluster ids using the Hungarian algorithm
    matrix_parts_iou, ids_pred_h, ids_gt_h = map_cluster_ids_to_gt(cluster_assignment_3d, cluster_assignment_features_only_for_points)
    ids_pred_h = ids_pred_h.to(device)

    # align the cluster ids
    cluster_assignment_3d = ids_pred_h[cluster_assignment_3d]

    # mask where the cluster ids are not aligned
    mask = cluster_assignment_3d != cluster_assignment_features_only_for_points

    # get distances matrix
    distances = torch.cdist(coords_3d, coords_3d)

    # get the indices of the k closest 3d points for each misaligned point
    indices = torch.argsort(distances[mask], dim=1)[:, :k_mode_closest_neighbors]
    del distances

    # compute the mode of the cluster ids of the k closest 3d points
    cluster_assignment_features_only_for_points[mask] = torch.mode(cluster_assignment_features[indices], dim=1)[0]
    cluster_assignment_features[:len(cluster_assignment_3d)] = cluster_assignment_features_only_for_points

    return cluster_assignment_features
