"""
PyTorch Geometric contains a lot of layers already implemented; however, the majority of them focus on computing
the node properties, not the edges. They also provide a function that converts a graph to a line graph.
This tool is created to merge these together. The line layer takes a graph, converts it to a line graph,
then applies the original PyTorch layer, and then converts it back to a normal graph.
"""

import torch


def average_reciprocal_edges(edge_index, edge_attr):
    # Convert edge_index to a list of tuples
    edge_tuples = [(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.size(1))]

    # Create a dictionary to store unique edges and their attributes
    edge_dict = {}

    for i, (u, v) in enumerate(edge_tuples):
        if (v, u) in edge_dict:
            # If the reciprocal edge exists, average the attributes
            edge_dict[(v, u)] = (edge_dict[(v, u)] + edge_attr[i]) / 2
        elif (u, v) in edge_dict:
            # If the edge itself already exists, average the attributes
            edge_dict[(u, v)] = (edge_dict[(u, v)] + edge_attr[i]) / 2
        else:
            # Otherwise, add the edge and its attributes to the dictionary
            edge_dict[(u, v)] = edge_attr[i]

    # Extract the unique edges and their attributes
    unique_edges = list(edge_dict.keys())
    unique_edge_attr = list(edge_dict.values())

    # Convert the list of unique edges back to a tensor
    new_edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
    new_edge_attr = torch.stack(unique_edge_attr)
    return new_edge_index, new_edge_attr


def line_graph(x, edge_index, edge_attr):
    # Convert to undirected graph to ensure bidirectional edges


    # average (01 + 10)/2=01
    edge_index, edge_attr = average_reciprocal_edges(edge_index, edge_attr)



    new_x = edge_attr
    node_neighbours = {}
    old_node_new_edge = {}
    for i in range(len(x)):
        node_neighbours[i] = {}
        old_node_new_edge[i] = []


    for i in range(edge_index.shape[1]):
        na = edge_index[0][i].item()
        nb = edge_index[1][i].item()
        node_neighbours[na][nb] = i
        node_neighbours[nb][na] = i



    new_edge_index = [[], []]
    new_edge_atr = []
    edge_nr = 0

    for old_edge_i in range(edge_index.shape[1]):
        na = edge_index[0][old_edge_i].item()
        nb = edge_index[1][old_edge_i].item()

        # edges with na neighbour:
        for node in [na, nb]:
            for key in node_neighbours[node].keys():
                old_edge_j = node_neighbours[node][key]
                new_edge_index[0].append(old_edge_i)
                new_edge_index[1].append(old_edge_j)
                new_edge_atr.append(x[node])
                old_node_new_edge[node].append(edge_nr)
                edge_nr += 1

    edge_attr = torch.stack(new_edge_atr)
    x = new_x

    edge_index = torch.tensor(new_edge_index).to(x.device)

    return x, edge_index, edge_attr, (node_neighbours, old_node_new_edge)


def reverse_line_graph(x, edge_attr, old_info):
    old_node_neighbours, old_node_new_edge = old_info


    new_edge_index = [[], []]
    new_edge_attr = []

    for na in old_node_neighbours.keys():
        for nb in old_node_neighbours[na].keys():
            new_edge_index[0].append(na)
            new_edge_index[1].append(nb)
            new_edge_attr.append(x[old_node_neighbours[na][nb]])

    new_x = []
    dummy = torch.zeros_like(edge_attr[0])
    for node in old_node_new_edge.keys():
        nx = dummy
        for edge in old_node_new_edge[node]:
            nx += edge_attr[edge]
        if len(old_node_new_edge[node])!=0:
            nx = nx / len(old_node_new_edge[node])
        new_x.append(nx)

    edge_index = torch.tensor(new_edge_index)
    edge_attr = torch.stack(new_edge_attr)
    x = torch.stack(new_x)


    return x, edge_index, edge_attr


# Tests
def test_line_graph():
    x = torch.randn((4, 3))  # Node features
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)  # Edge indices
    edge_attr = torch.randn((4, 2))  # Edge features (optional)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # Batch info (optional)
    bond_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # Bond batch info (optional)

    new_x, new_edge_index, edge_attr, old_node_neighbours = line_graph(x, edge_index, edge_attr)

    print("new_edge_index:", new_edge_index)
    assert new_x.size(0) == edge_index.size(1) or new_x.size(0) == int(
        edge_index.size(1) / 2), "Number of line graph nodes should equal number of original edges or half"
    assert new_edge_index.size(0) == 2, "Edge index should have two rows"
    assert new_edge_index.size(1) % 2 == 0, "Each edge should be bidirectional"

    original_x, original_edge_index, edge_attr = reverse_line_graph(
        new_x, edge_attr, old_node_neighbours
    )

    print("original_edge_index:", original_edge_index)
    assert original_edge_index.size(1) == edge_index.size(1), "Original and reconstructed edge indices should match"
    assert original_x.size(0) == x.size(0), "Original and reconstructed node features should match"

    print("All tests passed!")


class LineWrapper(torch.nn.Module):
    def __init__(self, line_module):
        super().__init__()
        self.line_module = line_module

    def forward(self, x, edge_index, edge_attr=None, state=None, batch=None, bond_batch=None):
        # x: Node feature matrix
        # edge_index: Graph connectivity matrix
        # edge_attr: Edge feature matrix (optional)
        # state: Additional state information (optional)
        # batch: Batch information for batched input (optional)
        # bond_batch: Batch information specific to bonds (optional)

        # convert to line_graph


        new_x, new_edge_index, edge_attr, old_description = line_graph(x, edge_index, edge_attr)




        # literally forward step
        new_x, edge_attr, u = self.line_module(new_x, new_edge_index, edge_attr, state)

        # recover old shape
        x, edge_index, edge_attr = reverse_line_graph(
            new_x, edge_attr, old_description
        )

        return x, edge_attr, state


if __name__ == "__main__":
    # Run tests
    test_line_graph()
