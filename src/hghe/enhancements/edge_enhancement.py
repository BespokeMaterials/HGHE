"""
Add edge proprieties.
"""

import numpy as np
from scipy.special import sph_harm
import math
import torch
from tqdm import tqdm
import concurrent.futures

def f_cut(r, decay_rate=3, cutoff=50):
    """
    Computes the cosine decay cutoff function.

    Parameters:
        r (float or numpy array): Distance value(s).
        decay_rate (float): Decay rate parameter.
        cutoff (float): Cutoff distance.

    Returns:
        float or numpy array: Output value(s) of the cosine decay cutoff function.
    """
    # Compute the cosine term with the cutoff
    cutoffs = 0.5 * (np.cos(r * math.pi / cutoff) + 1.0)

    # Apply the exponential decay with the decay rate
    decay = np.exp(-decay_rate * r)

    # Combine both terms
    combined = cutoffs * decay

    # Remove contributions beyond the cutoff radius
    combined *= (r < cutoff)

    return combined


def bessel_distance(c1, c2, n=[1, 2, 3, 4, 5, 6], rc=3):
    """
    Computes the Bessel distance between two points in 3D space using a Bessel function
    with a cosine decay cutoff function.

    Parameters:
        c1 (tuple or list): Coordinates (x, y, z) of the first point.
        c2 (tuple or list): Coordinates (x, y, z) of the second point.
        n (list): List of integers representing the order of Bessel functions.
        rc (float): Cutoff distance.

    Returns:
        list: A list of Bessel function values modulated by a cosine decay cutoff function
              and normalized by the distance between points.
    """
    # Compute squared Euclidean distance between c1 and c2
    d = (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2
    rij = np.sqrt(d)

    # Compute normalization factor
    c = np.sqrt(2 / rc)

    # Compute cosine decay cutoff function
    fc = f_cut(rij, rc * 0.5)

    # Compute Bessel function values modulated by the cutoff function
    bes = [c * fc * (np.sin(n_ * math.pi * rij / rc)) / rij for n_ in n]

    return bes


def spherical_harmonics(c1, c2, max_l=1):
    """
    Computes the real spherical harmonics between two points in 3D space up to a specified order.

    Parameters:
        c1 (array-like): Coordinates (x, y, z) of the first point.
        c2 (array-like): Coordinates (x, y, z) of the second point.
        max_l (int): Maximum degree of the spherical harmonics.

    Returns:
        list: A list of real spherical harmonics values for each (l, m) up to the specified order.
    """
    # Move to center
    rc = c1 - c2

    # Convert cartesian coordinates to spherical coordinates
    r, theta, phi = cartesian_to_spherical(rc[0], rc[1], rc[2])

    y = []

    # Compute real spherical harmonics up to the specified order
    for l in range(max_l + 1):  # Correcting range to include max_l
        for m in range(-l, l + 1):  # Correcting range to include l
            ylm = real_spherical_harmonics(l, m, theta, phi)
            y.append(ylm)

    return y


def cartesian_to_spherical(x, y, z):
    """
    Converts Cartesian coordinates to spherical coordinates.

    Parameters:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.

    Returns:
        tuple: A tuple (r, theta, phi) where:
            - r (float): The radial distance from the origin to the point.
            - theta (float): The polar angle (inclination) in radians, measured from the positive z-axis.
            - phi (float): The azimuthal angle in radians, measured from the positive x-axis in the x-y plane.
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def real_spherical_harmonics(l, m, theta, phi):
    """
    Computes the real part of the spherical harmonics function for given parameters.

    Parameters:
        l (int): Degree of the spherical harmonics.
        m (int): Order of the spherical harmonics.
        theta (float): Polar angle in radians.
        phi (float): Azimuthal angle in radians.

    Returns:
        float: The value of the real spherical harmonics for the given l, m, theta, and phi.
    """
    # Compute the complex spherical harmonics
    Y_lm_complex = sph_harm(m, l, phi, theta)

    # Compute real spherical harmonics based on m value
    if m > 0:
        return np.sqrt(2) * np.real(Y_lm_complex)
    elif m == 0:
        return np.real(Y_lm_complex)
    else:
        return np.sqrt(2) * (-1) ** m * np.imag(Y_lm_complex)


def compute_distance(coord_a, coord_b):
    # Subtract coordinates of the second point from the first point
    diff = np.array(coord_a) - np.array(coord_b)

    # Compute the square of each difference
    squared_diff = diff ** 2

    # Sum these squared differences
    sum_squared_diff = np.sum(squared_diff)

    # Take the square root of the sum to get the Euclidean distance
    distance = np.sqrt(sum_squared_diff)

    return [distance]


class EdgeEnhanceElementGraph:
    def __init__(self):

        self.descript = ["distance",
                         "bessel_distance",
                         "spherical_harmonics",
                         ]

    def enhance_descriptor(self, element_graph):

        edge_descriptors = [None] * len(element_graph.data.edge_index.T)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for index, edge in enumerate(tqdm(element_graph.data.edge_index.T)):
                futures.append(executor.submit(compute_descriptor, index, edge, element_graph, self.descript))

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                index, descriptor, li_desc = future.result()
                edge_descriptors[index] = descriptor

        # TODO:This is a place for speed up:
        # for edge_nr, edge in enumerate(tqdm(element_graph.data.edge_index.T)):
        #
        #     descriptor,li_desc =get_descriptor(node_1=edge[0],
        #                                node_2=edge[1],
        #                                element_graph=element_graph,
        #                                descript=self.descript, )
        #
        #     distance,b_distance,shea_h=li_desc
        #     descriptor=descriptor
        #     edge_descriptors.append(descriptor)

        dsc = []
        if "distance" in self.descript:
            for i, k in enumerate(distance):
                dsc.append(f"distance_{i}")
        if "bessel_distance" in self.descript:
            for i, k in enumerate(b_distance):
                dsc.append(f"bessel_distance_{i}")
        if "spherical_harmonics" in self.descript:
            for i, k in enumerate(shea_h):
                dsc.append(f"spherical_harmonics_{i}")

        edge_descriptors = torch.tensor(edge_descriptors)

        # Insert the descriptor to the node graph
        element_graph.data.edge_attr = edge_descriptors
        element_graph.edge_descriptor = dsc

        return element_graph


def compute_descriptor(index, edge, element_graph, descript):
    descriptor, li_desc = get_descriptor(node_1=edge[0], node_2=edge[1], element_graph=element_graph, descript=descript)
    return index, descriptor, li_desc
def get_descriptor(node_1,node_2, element_graph, descript ):

    descriptor = []

    coord_a = element_graph.data.x[node_1][:3]
    coord_b = element_graph.data.x[node_2][:3]

    if "distance" in descript:
        distance = compute_distance(coord_a, coord_b)
        descriptor.extend(distance)
    if "bessel_distance" in descript:
        b_distance = bessel_distance(coord_a, coord_b, n=[i for i in range(1, 9)])
        descriptor.extend(b_distance)
    if "spherical_harmonics" in descript:
        shea_h = spherical_harmonics(coord_a, coord_b, max_l=7)
        descriptor.extend(shea_h)



    return descriptor , [distance, b_distance, shea_h]