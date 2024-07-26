"""
Convert to orbital representation.
"""
from hghe import ElementGraph, OrbitalGraph
from structure_to_graph import extract_elem_xyz_from_file


def main(file_path, radius):
    # Build the atom graph
    atoms, coordinates, lattice_vectors = extract_elem_xyz_from_file(file_path)
    print(f"\nnr atoms:{len(atoms)}\nlattice_vectors:{lattice_vectors}\natoms:{atoms}\ncoordinates:{coordinates}")

    atomic_graph = ElementGraph(atoms, coordinates, lattice_vectors, radius)
    atomic_graph.display_graph()
    print("atomic_graph:", atomic_graph)
    print("atomic_graph.data:", atomic_graph.data)

    # Extend to orbitals
    orbital_map = {"B": ["s", "p"], "N": ["s", "q"]}
    orbital_encode = {"s": [0, 0, 0, 0, 0, 1],
                      "p": [0, 0, 0, 0, 1, 0],
                      "q": [0, 0, 0, 0, 1, 1]}
    orbital_graph = OrbitalGraph(atomic_graph, orbital_map, orbital_encode)
    orbital_graph.display_graph()

    print("atomic_graph:", atomic_graph)
    print("atomic_graph.data:", atomic_graph.data)
    print("orbital_graph:", orbital_graph)
    print("orbital_graph.data:", orbital_graph.data)
    print("orbitals:", orbital_graph.orbitals)


if __name__ == "__main__":
    example_file_path = "../Data/test_structures/aBN.txt"
    example_radius = 2
    main(example_file_path, example_radius)
