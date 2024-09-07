import torch
from source.point_cloud_utils.kmeans import kmeans


def spectral_clustering_cluster_qr(features, k, k_means_iterations=30, ratio=0.5, device='cuda'):
    # features: (N, C)
    # k: number of clusters
    # ratio: ratio of eigenvectors to use for clustering
    # device: cpu or cuda
    # returns: (N, 1) cluster assignments
    #
    
    # compute the impact of each dimension on the clustering
    # (N, C)
    features = features.to(device)
    features = features - features.mean(dim=0, keepdim=True)
    
    # (C, C)
    cov = features.T @ features
    
    # (C, C)
    _, S, _ = torch.svd(cov)
    
    # (C, 1)
    S = S.unsqueeze(-1)
    
    # (C, 1)
    S = torch.sqrt(S)
    
    # (C, 1)
    S = torch.reciprocal(S)
    
    # (C, C)
    S = torch.diag(S.squeeze())
    
    # (C, C)
    cov_inv = S @ cov @ S
    
    # get the best k dimensions
    # (C, k)
    _, _, V = torch.svd(cov_inv)
    
    # (C, k)
    V = V[:, :k]
    
    # (N, k)
    features = features @ V

    # get the cluster assignments
    # (N)
    cluster_assignments = kmeans(k, features, iterations=k_means_iterations)
    
    return cluster_assignments