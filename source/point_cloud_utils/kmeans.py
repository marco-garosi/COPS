import torch


def kmeans(k, vectors, iterations=30, return_centroids=False, centroids=None):
    """
    Fast k-means implementation which works even on GPU

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
    if centroids is None:
        centroids = vectors[torch.randperm(vectors.size(0))[:k]]

    # Initialize cluster_assignment
    cluster_assignment = None

    for _ in range(iterations):
        # Calculate distances from centroids
        distances = torch.cdist(vectors, centroids)

        # Assign each point to the nearest centroid
        cluster_assignment = torch.argmin(distances, dim=1)

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
