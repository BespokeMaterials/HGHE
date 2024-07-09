"""
Put xyz data into a graph format with no extra information ito it.
"""

import torch
from torch_geometric.data import Data
from .ase_utils import find_neighbors_within_radius


class ElementGraph:
    def __init__(self, elements, coordinates, lattice_vectors, radius):
        """
        Initialize the ElementGraph.

        :param elements: List of atomic elements.
        :type elements: list of str
        :param coordinates: Array of atomic coordinates.
        :type coordinates: numpy.ndarray
        :param lattice_vectors: Array of lattice vectors.
        :type lattice_vectors: numpy.ndarray
        :param radius: Radius within which to search for neighbors.
        :type radius: float
        """
        self.elements = elements
        self.coordinates = coordinates
        self.lattice_vectors = lattice_vectors
        self.radius = radius
        self.data = self._build_pyg_data()

    def _build_pyg_data(self):
        """
        Build PyTorch Geometric data object.

        :return: PyTorch Geometric data object.
        :rtype: torch_geometric.data.Data
        """
        neighbors = find_neighbors_within_radius(self.elements, self.coordinates, self.lattice_vectors, self.radius)
        edge_index = []

        for i, nbrs in neighbors.items():
            for j in nbrs:
                edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        x = torch.tensor(self.coordinates, dtype=torch.float)
        lattice = torch.tensor(self.lattice_vectors, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, lattice=lattice)

    def __repr__(self):
        return f"ElementGraph(nodes={self.data.num_nodes}, edges={self.data.num_edges})"

    def display_graph(self):
        """
        Display the graph nodes and edges.
        """
        print("Nodes:")
        for i, element in enumerate(self.elements):
            print(f"Node {i}: {element} at {self.coordinates[i]}")

        print("\nEdges:")
        edge_index = self.data.edge_index.t().tolist()
        for edge in edge_index:
            print(f"Edge between Node {edge[0]} and Node {edge[1]}")

    def save(self, filepath):
        """
        Save the ElementGraph to a file.

        :param filepath: Path to the file where the graph will be saved.
        :type filepath: str
        """
        torch.save({
            'elements': self.elements,
            'coordinates': self.coordinates,
            'lattice_vectors': self.lattice_vectors,
            'radius': self.radius,
            'data': self.data
        }, filepath)

    @classmethod
    def load(cls, filepath):
        """
        Load the ElementGraph from a file.

        :param filepath: Path to the file where the graph is saved.
        :type filepath: str
        :return: Loaded ElementGraph object.
        :rtype: ElementGraph
        """
        checkpoint = torch.load(filepath)
        element_graph = cls.__new__(cls)
        element_graph.elements = checkpoint['elements']
        element_graph.coordinates = checkpoint['coordinates']
        element_graph.lattice_vectors = checkpoint['lattice_vectors']
        element_graph.radius = checkpoint['radius']
        element_graph.data = checkpoint['data']
        return element_graph
