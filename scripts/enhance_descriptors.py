"""
Given a graph that contains the atomic vectors, atomic times, positions, and possibly the orbitals,
how much additional information can we extract to build a descriptor? And what should we eliminate?
"""

from hghe import ElementGraph, OrbitalGraph
from hghe.enhancements import ChemEnhanceElementGraph, EdgeEnhanceElementGraph
from structure_to_graph import extract_elem_xyz_from_file


def main(file_path, radius):
    # Build the atom graph
    atoms, coordinates, lattice_vectors = extract_elem_xyz_from_file(file_path)
    print(f"\nnr atoms:{len(atoms)}\nlattice_vectors:{lattice_vectors}\natoms:{atoms}\ncoordinates:{coordinates}")

    atomic_graph = ElementGraph(atoms, coordinates, lattice_vectors, radius)
    atomic_graph.display_graph()
    print("atomic_graph:", atomic_graph)
    print("atomic_graph.data:", atomic_graph.data)
    print("atomic_graph.node_descriptor:", atomic_graph.node_descriptor)

    # Enhance the atomic graph
    # Node enhancement with chemical data
    chem_enh = ChemEnhanceElementGraph()
    chem_enh_atomic_graph = chem_enh.enhance_descriptor(atomic_graph)

    print("chem_enh_atomic_graph:", chem_enh_atomic_graph)
    print("chem_enh_atomic_graph.data:", chem_enh_atomic_graph.data)
    print("chem_enh_atomic_graph.node_descriptor", chem_enh_atomic_graph.node_descriptor)

    # Node edge enhancement:
    edge_enh = EdgeEnhanceElementGraph()
    enh_graph = edge_enh.enhance_descriptor(chem_enh_atomic_graph)

    print("enh_atomic_graph:", enh_graph)
    print("enh_atomic_graph.data:", enh_graph.data)
    print("enh_atomic_graph.node_descriptor", enh_graph.node_descriptor)
    print("enh_atomic_graph.edge_descriptor", enh_graph.edge_descriptor)

    # Let´s make it equivariant and remouve the x,y, z info from nodes
    enh_graph.edge_descriptor = enh_graph.edge_descriptor[3:]
    enh_graph.data.x = enh_graph.data.x[:, 3:]
    print("enh_atomic_graph.data:", enh_graph.data)
    print("enh_atomic_graph.node_descriptor", enh_graph.node_descriptor)

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
