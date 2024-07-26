import torch
from torch_geometric.data import Data
from hghe.utils import extract_neighbors


class OrbitalGraph:
    def __init__(self, atomic_graph, orbital_map):
        """
        Initialize the OrbitalGraph.

        :param atomic_graph: The original atomic graph.
        :type atomic_graph: ElementGraph
        :param orbital_map: Mapping from elements to their orbitals.
        :type orbital_map: dict
        """
        self.atomic_graph = atomic_graph
        self.orbital_map = orbital_map
        self.orbitals = None
        self.data = self._build_orbital_graph()

    def _build_orbital_graph(self):
        """
        Build PyTorch Geometric data object with orbitals.

        :return: PyTorch Geometric data object.
        :rtype: torch_geometric.data.Data
        """
        node_info = self.atomic_graph.data.x
        edge_info = self.atomic_graph.data.edge_attr
        global_info = self.atomic_graph.data.u
        neighbors = extract_neighbors(self.atomic_graph.data)

        atoms_and_orbitals = {}

        # Create orbital nodes and store their indices
        idx = 0
        for atom, element in enumerate(self.atomic_graph.elements):
            orbitals = self.orbital_map[element]
            atoms_and_orbitals[atom] = {}
            for orbital in orbitals:
                atoms_and_orbitals[atom][idx] = orbital
                idx += 1

        self.orbitals = atoms_and_orbitals

        # Create edges between orbital nodes and their neighbors' orbitals
        edge_index = [[], []]
        x = []
        edge_attr = []

        # go thru each atom
        for atom, nbrs in neighbors.items():
            # go through etch orbital of the atom
            orbitals = atoms_and_orbitals[atom]
            for orbital in orbitals:
                # orbital node:
                extended_orbital_node=node_info[atom]

                extended_orbital_node=torch.cat((extended_orbital_node, torch.tensor([orbital])), dim=0)
                x.append(extended_orbital_node)
                # orbital end edge:
                e1 = orbital

                # go to all the neighbours and build edges
                for nb in nbrs:
                    # go through the orbitals of the neighbour:
                    n_orbitals = atoms_and_orbitals[nb[0]]
                    for n_orbital in n_orbitals:
                        if n_orbital != orbital:
                            e2 = n_orbital
                            # add edge:
                            edge_index[0].append(e1)
                            edge_index[1].append(e2)
                            edge_attr.append(edge_info[nb[1]])

                # Connection between orbitals of the same atoms.
                for n_orbital in orbitals:
                    if n_orbital != orbital:
                        e2 = n_orbital
                        # add edge:
                        edge_index[0].append(e1)
                        edge_index[1].append(e2)
                        edge_attr.append(torch.zeros_like(edge_info[0]))

        # pack everything to a graph:
        edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
        edge_attr = torch.stack(edge_attr)
        x = torch.stack(x)
        u = global_info
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u)

    def __repr__(self):
        return f"OrbitalGraph(nodes={self.data.num_nodes}, edges={self.data.num_edges})"

    def display_graph(self):
        """
        Display the graph nodes and edges.
        """
        print("Nodes:")
        for i, element in enumerate(self.data.x):
            print(f"Node {i}: {element}")

        print("\nEdges:")
        edge_index = self.data.edge_index.t().tolist()
        edge_attr = self.data.edge_attr.tolist()
        for edge, attr in zip(edge_index, edge_attr):
            print(f"Edge between Node {edge[0]} and Node {edge[1]} with attributes {attr}")
