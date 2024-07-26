"""
PyTorch Geometric contains a lot of layers already implemented; however, the majority of them focus on computing
the node properties, not the edges. They also provide a function that converts a graph to a line graph.
This tool is created to merge these together. The line layer takes a graph, converts it to a line graph,
then applies the original PyTorch layer, and then converts it back to a normal graph.
"""

from hghe import ElementGraph, OrbitalGraph, LineWrapper
from structure_to_graph import extract_elem_xyz_from_file
import torch
from torch_geometric.nn import ResGatedGraphConv


class AdaptToLine(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(AdaptToLine, self).__init__()

        self.conv1 = ResGatedGraphConv(in_channels=in_channels, out_channels=out_channels, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr, u):
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)

        return x, edge_attr, u


class DemoModel(torch.nn.Module):
    def __init__(self, in_channels_line, out_channels_line, edge_dim_line):
        super().__init__()
        self.line_g_layer1 = LineWrapper(line_module=AdaptToLine(in_channels_line, out_channels_line, edge_dim_line))

    def forward(self, x, edge_index, edge_attr, u, batch):
        x, edge_attr, u = self.line_g_layer1(x, edge_index, edge_attr, u, batch)
        return x, edge_attr, u, edge_index


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
    orbital_encode = {"s": [ 1],
                      "p": [2],
                      "q": [3]}
    orbital_graph = OrbitalGraph(atomic_graph, orbital_map,orbital_encode)
    orbital_graph.display_graph()

    print("atomic_graph:", atomic_graph)
    print("atomic_graph.data:", atomic_graph.data)
    print("orbital_graph:", orbital_graph)
    print("orbital_graph.data:", orbital_graph.data)
    print("orbitals:", orbital_graph.orbitals)

    # build  model:
    model = DemoModel(in_channels_line=1,
                      out_channels_line=4,
                      edge_dim_line=4)

    # pass through the model:
    x, edge_attr, u, edge_index = model(orbital_graph.data.x,
                                        orbital_graph.data.edge_index,
                                        orbital_graph.data.edge_attr,
                                        orbital_graph.data.u, batch=None)

    print(f"new_graph:x:{x.shape}, edge_attr:{edge_attr.shape}, u:{u.shape}, edge_index:{edge_index.shape}")


if __name__ == "__main__":
    example_file_path = "../Data/test_structures/aBN.txt"
    example_radius = 2
    main(example_file_path, example_radius)
