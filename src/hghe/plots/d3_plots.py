import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D


def plot_points_and_lattice(elements, coordinates, lattice_vectors, sphere_radii, sphere_colors):
    """
    Plot points as spheres and lattice vectors as arrows.

    :param elements: List of atomic elements.
    :type elements: list of str
    :param coordinates: Array of atomic coordinates.
    :type coordinates: numpy.ndarray
    :param lattice_vectors: Array of lattice vectors.
    :type lattice_vectors: numpy.ndarray
    :param sphere_radii: List of radii for the spheres representing points.
    :type sphere_radii: list of float
    :param sphere_colors: List of colors for the spheres.
    :type sphere_colors: list of str
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the atoms as spheres
    for coord, radius, color in zip(coordinates, sphere_radii, sphere_colors):
        plot_sphere(ax, coord, radius, color)

    # Plot the lattice vectors
    origin = np.zeros(3)
    for vec in lattice_vectors:
        ax.quiver(*origin, *vec, color='r', arrow_length_ratio=0.1)

    # Set plot limits
    max_limit = np.max(coordinates) + max(sphere_radii)
    min_limit = np.min(coordinates) - max(sphere_radii)
    ax.set_xlim([min_limit, max_limit])
    ax.set_ylim([min_limit, max_limit])
    ax.set_zlim([min_limit, max_limit])

    plt.show()


def plot_sphere(ax, center, radius, color):
    """
    Plot a sphere at a given center.

    :param ax: Matplotlib 3D axis.
    :type ax: mpl_toolkits.mplot3d.Axes3D
    :param center: Center of the sphere.
    :type center: list or numpy.ndarray
    :param radius: Radius of the sphere.
    :type radius: float
    :param color: Color of the sphere.
    :type color: str

    # Example usage
    elements = ['B', 'B', 'B']
    coordinates = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    lattice_vectors = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    sphere_radii = [0.5, 0.7, 0.6]
    sphere_colors = ['b', 'g', 'r']
    plot_points_and_lattice(elements, coordinates, lattice_vectors, sphere_radii, sphere_colors)
    """
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    ax.plot_surface(x, y, z, color=color, alpha=0.6)
