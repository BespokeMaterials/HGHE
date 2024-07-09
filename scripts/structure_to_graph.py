"""
Convert atoms and coordinates to a graph knowing the cutoff radius and the lattice.
It uses periodic boundary condition.
"""

from hghe.ase_utils import find_neighbors_within_radius
from hghe.plots import plot_points_and_lattice
from hghe import ElementGraph
import numpy as np


def extract_elem_xyz_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract the lattice vectors from the second line
    lattice_line = lines[1]
    lattice_str = lattice_line.split('Lattice="')[1].split('"')[0]
    lattice_vectors = list(map(float, lattice_str.split()))

    # Skip the first line (64) and the second line (Lattice information)
    atoms_info = lines[2:]

    elements = []
    coordinates = []

    for line in atoms_info:
        parts = line.split()
        element = parts[0]
        coords = list(map(float, parts[1:4]))
        elements.append(element)
        coordinates.append(coords)

    lattice_vectors = np.array(lattice_vectors).reshape(-1, 3)
    return elements, coordinates, lattice_vectors


def main(file_path, radius):
    atoms, coordinates, lattice_vectors = extract_elem_xyz_from_file(file_path)
    print(f"\nnr atoms:{len(atoms)}\nlattice_vectors:{lattice_vectors}\natoms:{atoms}\ncoordinates:{coordinates}")

    # Neighbours:
    coordinates = np.array(coordinates)
    neighbors = find_neighbors_within_radius(atoms, coordinates, lattice_vectors, radius)
    print("Neighbors within radius:")
    for atom, neighs in neighbors.items():
        print(f"Atom {atom} ({atoms[atom]}): {neighs}")

    # Now we should be able to plot it to ensure that the neighbors are assigned properly.

    # Example usage
    sphere_radii = [0.1 for _ in atoms]
    sphere_colors = ['b' for _ in atoms]

    #mark an atom and neighbours
    atom = 0
    sphere_radii[atom] = 0.2
    sphere_colors[atom] = 'r'
    for a in neighbors[atom]:
        sphere_colors[a] = "g"

    plot_points_and_lattice(elements=atoms,
                            coordinates=coordinates,
                            lattice_vectors=lattice_vectors,
                            sphere_radii=sphere_radii,
                            sphere_colors=sphere_colors)

    graph = ElementGraph(atoms, coordinates, lattice_vectors, radius)
    graph.display_graph()

    # Save the graph
    graph.save('element_graph.pth')

    # Load the graph
    loaded_graph = ElementGraph.load('element_graph.pth')

    print(loaded_graph)
    loaded_graph.display_graph()


if __name__ == "__main__":
    file_path = "/Users/voicutomut/Documents/GitHub/HGHE/Data/test_structures/aBN.txt"
    radius = 1
    main(file_path, radius)
