import torch
from collections import defaultdict


def extract_neighbors(data):
    """
        Extracts neighbors and the edge numbers connecting them as a dictionary from a graph represented by a Data object.

        :param data: A Data object containing graph information with edge_index attribute.
        :type data: torch_geometric.data.Data
        :return: A dictionary where the keys are node indices and the values are lists of tuples (neighboring node index, edge number).
        :rtype: dict
        """
    edge_index = data.edge_index

    # Initialize a defaultdict of lists for storing neighbors and edge numbers
    neighbors_with_edges = defaultdict(list)

    # Convert edge_index to a list of tuples
    edges = edge_index.t().tolist()

    # Populate the neighbors_with_edges dictionary
    for edge_nr, (src, dst) in enumerate(edges):
        neighbors_with_edges[src].append((dst, edge_nr))
        neighbors_with_edges[dst].append((src, edge_nr))

    # Convert defaultdict to a regular dict
    neighbors_with_edges = dict(neighbors_with_edges)

    return neighbors_with_edges