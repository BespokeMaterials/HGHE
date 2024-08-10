import os
import json
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import glob
import math
import torch

def matrix_plot(matrix, filename, grid1_step=1, grid2_step=13):
    """
    Generate and save a heatmap from a given matrix.

    :param matrix: 2D array of data
    :param filename: The file name to save the heatmap
    :param grid1_step: Step for the first grid (default is 1)
    :param grid2_step: Step for the second grid (default is 13)
    """
    # Determine the min and max values of the matrix
    min_val = np.min(matrix)
    if min_val >=0:
        min_val=-0.1
    max_val = np.max(matrix)
    if max_val <=0:
        max_val=+0.1

    # Create the heatmap
    plt.figure()
    norm = TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
    plt.imshow(matrix, cmap='seismic', norm=norm, interpolation='nearest')
    plt.colorbar()

    # Add grids
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], grid1_step), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], grid1_step), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.05)

    if grid2_step > 0:
        ax.set_xticks(np.arange(-0.5, matrix.shape[1], grid2_step), minor=False)
        ax.set_yticks(np.arange(-0.5, matrix.shape[0], grid2_step), minor=False)
        ax.grid(which='major', color='gray', linestyle='-', linewidth=0.25)

    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print("Saved heatmap to {}".format(filename))