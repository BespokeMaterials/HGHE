"""
Some utility functions built on top of ASE. don top of ase.
"""
from ase import Atoms
from ase.neighborlist import NeighborList


def find_neighbors_within_radius(elements, coordinates, lattice_vectors, radius):
    """
    Find neighbors within a given radius considering periodic boundary conditions.

    :param elements: List of atomic elements.
    :type elements: list of str
    :param coordinates: Array of atomic coordinates.
    :type coordinates: numpy.ndarray
    :param lattice_vectors: Array of lattice vectors.
    :type lattice_vectors: numpy.ndarray
    :param radius: Radius within which to search for neighbors.
    :type radius: float
    :return: Dictionary with atom indices as keys and lists of neighbor indices as values.
    :rtype: dict of int to list of int
    """
    # Create an ASE Atoms object
    atoms = Atoms(symbols=elements, positions=coordinates, cell=lattice_vectors, pbc=True)

    # Generate cutoff distances for each atom
    cutoffs = [radius] * len(atoms)

    # Create the NeighborList object
    nl = NeighborList(cutoffs=cutoffs, bothways=True, self_interaction=False)
    nl.update(atoms)

    neighbors = {i: [] for i in range(len(atoms))}

    # Find neighbors for each atom
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        neighbors[i] = indices.tolist()

    return neighbors