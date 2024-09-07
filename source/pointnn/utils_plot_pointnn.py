import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def random_sampling(point_cloud, segmentation, num_samples=2048):
    """
    Randomly samples points from the point cloud.
    
    params: point_cloud: point cloud to sample from
    params: segmentation: segmentation of the point cloud
    params: num_samples: number of points to sample
    
    :return: sampled_point_cloud: sampled point cloud
    
    """
    mask = torch.randperm(point_cloud.shape[0])[:num_samples]
    sampled_point_cloud = point_cloud[mask]
    segmentation_mask = segmentation[mask]
    
    return sampled_point_cloud, segmentation_mask


def plot_point_cloud(point_cloud, title='', axis=False, show=True):
    """
    Visualizes a point cloud with matplotlib.

        param point_cloud: a numpy array of shape (#points, 6), where 6 is due to (x, y, z, R, G, B)
        param title: the title of the plot
        param axis: whether to show the axis or not
        :return: None
        
    """
    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    
    # rotate the point cloud to be able to see it from 90 degrees x axis and 30 degrees z axis and 90 degrees y axis
    ax.view_init(30, 0, 90)
    # ax.view_init(45, 90)
    
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=12.0, c=point_cloud[:, 3:]/255.0)
    if axis:
        ax.set(xlabel='X', ylabel='Y', zlabel='Z')
    ax.set_title(title)#, fontsize=23)
    if show:
        plt.show()
    
    # get the figure as numpy array dpi=300
    fig = plt.gcf()
    fig.canvas.draw()
    fig_np = np.array(fig.canvas.renderer._renderer)
    plt.close()
    
    return fig_np