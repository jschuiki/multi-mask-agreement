
import numpy as np
import os

from scipy.spatial import Voronoi, KDTree

class MaskGenerator():
    
    def __init__(self, factor=0.1092):
        self.factor = factor

    def generate_mask(self, patch_size, num_obj=None, seed=None):

        if num_obj is None:
            num_obj = round((patch_size*self.factor)**2)

        if seed is not None:
            np.random.seed(seed)

        points = patch_size*np.random.rand(num_obj, 2)
        return points, self.points_to_voronoi(points, patch_size)

    def points_to_voronoi(self, points, grid_size):
        tree = KDTree(points)
        x = np.linspace(0, grid_size, grid_size)
        y = np.linspace(0, grid_size, grid_size)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # For each grid point, find the closest seed index
        _, cell_labels = tree.query(grid_points)
        return cell_labels.reshape((grid_size, grid_size))+1    

