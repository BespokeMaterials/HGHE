from hghe import ElementGraph, OrbitalGraph
from hghe.enhancements import ChemEnhanceElementGraph, EdgeEnhanceElementGraph
from datasets import QH9Stable
import torch
import periodictable


def main(dataset, radius, save_path):
    graphs = []

    for index, g in enumerate(dataset):
        print("g:", g)

        # Build the atom graph
        atoms = g.atoms
        atoms = [periodictable.elements[int(atom[0])].symbol for atom in atoms]
        coordinates = g.pos
        lattice_vectors = torch.tensor([1, 1, 1])

        print(
            f"\nnr atoms:{len(atoms)}\nlattice_vectors:{lattice_vectors}\natoms:{atoms}\ncoordinates:{coordinates}")

        atomic_graph = ElementGraph(atoms, coordinates, lattice_vectors, radius)
        # TODO : coment
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

        atomic_graph.data.hmat_on = g.diagonal_hamiltonian
        atomic_graph.data.hmat_hop = g.non_diagonal_hamiltonian

        # prune the edges:
        print("atomic_graph.data:", atomic_graph.data)
        print("atomic_graph.data.hmat_on:", atomic_graph.data.hmat_on.shape)
        print("atomic_graph.data.hmat_hop:", atomic_graph.data.hmat_hop.shape)
        print("atomic_graph.data.edge_index:", atomic_graph.data.edge_index)
        print("atomic_graph.data.edge_index:", g.edge_index_full)

        ordered_edges_atr, ordered_edge_index = order_function(hop_edge_index=g.edge_index_full,
                                                               edge_atr_index=atomic_graph.data.edge_index,
                                                               edge_atr=atomic_graph.data.edge_attr, )
        atomic_graph.data.edge_index = ordered_edge_index
        atomic_graph.data.edge_attr = ordered_edges_atr
        print("GRAPH:", atomic_graph)
        graphs.append(atomic_graph)

    # Combine the graphs and target properties into a single dictionary
    data_to_save = {
        'graphs': graphs,

    }

    # Save the data
    torch.save(data_to_save, save_path)
    print(f"Total of {len(graphs)} graphs for training")


def order_function(hop_edge_index, edge_atr_index, edge_atr):
    map = {}
    for i in range(len(hop_edge_index[0])):
        for j in range(len(edge_atr_index[0])):
            if hop_edge_index[0][i] == edge_atr_index[0][j] and hop_edge_index[1][i] == edge_atr_index[1][j]:
                map[i] = j

                break

    ordered_edges_atr = torch.stack([edge_atr[map[i]] for i in range(len(hop_edge_index[0]))])
    ordered_edge_index = hop_edge_index

    return ordered_edges_atr, ordered_edge_index


if __name__ == "__main__":
    import pickle

    # dataset = QH9Stable(split='random')
    # train_dataset = dataset[dataset.train_mask][:10]
    # with open('train_dataset.pkl', 'wb') as f:
    #     pickle.dump(train_dataset, f)
    with open('train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    save_path = "DATA/QH9/DFT_graphs_QH9_train_atomic.pt"
    main(train_dataset, radius=10, save_path=save_path)
