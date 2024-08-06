"""
Model for Hamiltonian prediction.

This model contains 4 blocks
a) Orbital encoding blok  block
b) Edge encoding blok
c) Onsite prediction blok
d) Interaction prediction block
"""

import torch
from torch_geometric.nn import GINEConv, TransformerConv, GATv2Conv
from hghe.graphs import LineWrapper


class OrbitalBlock(torch.nn.Module):
    def __init__(self, n_shape_in, e_shape_in, u_shape_in,
                 n_shape_out, e_shape_out, u_shape_out):
        super().__init__()
        self.input_dimensions = [n_shape_in, e_shape_in, u_shape_in]
        self.output_dimensions = [n_shape_out, e_shape_out, u_shape_out]

        # MLP layers for GINEConv
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(n_shape_in, int((n_shape_in + n_shape_out) / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int((n_shape_in + n_shape_out) / 2), n_shape_out)
        )

        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(n_shape_out, int(n_shape_out / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(n_shape_out / 2), n_shape_out)
        )

        # Convolutional layers
        self.conv1 = GINEConv(self.mlp1, train_eps=True, edge_dim=e_shape_in)
        self.conv2 = GINEConv(self.mlp2, train_eps=True, edge_dim=e_shape_in)
        self.conv3 = TransformerConv(n_shape_out, n_shape_out, heads=4, edge_dim=e_shape_in)
        self.conv4 = GATv2Conv(n_shape_out * 4, n_shape_out, edge_dim=e_shape_in)

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None, bond_batch=None):
        # Apply the first GINEConv layer
        x = self.conv1(x, edge_index, edge_attr, batch=batch, bond_batch=bond_batch)

        # Apply the second GINEConv layer
        x = self.conv2(x, edge_index, edge_attr, batch=batch, bond_batch=bond_batch)

        # Apply the TransformerConv layer
        x = self.conv3(x, edge_index, edge_attr, batch=batch, bond_batch=bond_batch)

        # Apply the GATv2Conv layer
        x = self.conv4(x, edge_index, edge_attr, batch=batch, bond_batch=bond_batch)

        return x


class InteractionBlock(torch.nn.Module):
    def __init__(self, n_shape_in, e_shape_in, u_shape_in,
                 n_shape_out, e_shape_out, u_shape_out,
                 ):
        super().__init__()
        self.input_dimensions = [n_shape_in, e_shape_in, u_shape_in]
        self.output_dimensions = [n_shape_out, e_shape_out, u_shape_out]



        class Seq(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n_shape_in=e_shape_in
                self.n_shape_out=e_shape_out
                self.e_shape_in=n_shape_in


                # MLP layers for GINEConv
                self.mlp1 = torch.nn.Sequential(
                    torch.nn.Linear(self.n_shape_in, int((self.n_shape_in + self.n_shape_out) / 2)),
                    torch.nn.ReLU(),
                    torch.nn.Linear(int((self.n_shape_in + self.n_shape_out) / 2), self.n_shape_out)
                )


            # Convolutional layers
            self.conv1 = GINEConv(self.mlp1, train_eps=True, edge_dim=self.e_shape_in)
            self.conv2 = TransformerConv(self.n_shape_out, self.n_shape_out, heads=4, edge_dim=self.e_shape_in)
            self.conv3 = TransformerConv(self.n_shape_out, self.n_shape_out, heads=4, edge_dim=self.e_shape_in)
            self.conv4 = GATv2Conv(self.n_shape_out * 4, self.n_shape_out, edge_dim=self.e_shape_in)

            def forward(self, x, edge_index, edge_attr=None, u=None, batch=None, bond_batch=None):
                # Apply the first GINEConv layer
                x = self.conv1(x, edge_index, edge_attr)

                # Apply the first TransformerConv layer
                x = self.conv2(x, edge_index, edge_attr)

                # Apply the second TransformerConv layer
                x = self.conv3(x, edge_index, edge_attr)

                # Apply the GATv2Conv layer
                x = self.conv4(x, edge_index, edge_attr)

                return x, edge_attr, u

        self.LineSeq1 = LineWrapper(line_module=LineWrapper(Seq()))
        self.LineSeq2 = LineWrapper(line_module=LineWrapper(line_module=Seq()))
        self.LineSeq3 = LineWrapper(line_module=LineWrapper(line_module=LineWrapper(line_module=Seq())))

        self.weights = torch.nn.Parameter(torch.tensor([0.5, 0.5, 0.5])),

        pass

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None, bond_batch=None):
        x1, edge_attr1, u1, edge_index = self.LineSeq1(x, edge_index, edge_attr, u, batch)
        x2, edge_attr2, u2, edge_index = self.LineSeq1(x, edge_index, edge_attr, u, batch)
        x3, edge_attr3, u3, edge_index = self.LineSeq1(x, edge_index, edge_attr, u, batch)

        edge_attr = (edge_attr1 * self.weights[0] + edge_attr2 * self.weights[1] + edge_attr3 * self.weights[2]) / 3
        x = (x1 * self.weights[0] + x2 * self.weights[1] + x3 * self.weights[2]) / 3

        return x, edge_attr, edge_index


class OnsiteBlock(torch.nn.Module):
    def __init__(self, n_shape_in, e_shape_in, u_shape_in,
                 n_shape_out, e_shape_out, u_shape_out,
                 depth=2):
        super().__init__()
        self.input_dimensions = [n_shape_in, e_shape_in, u_shape_in]
        self.output_dimensions = [n_shape_out, e_shape_out, u_shape_out]
        self.depth = depth

    def forward(self, x, e, u, edge_index, edge_attr=None, batch=None, bond_batch=None):
        pass


class HoppingBlock(torch.nn.Module):
    def __init__(self, n_shape_in, e_shape_in, u_shape_in,
                 n_shape_out, e_shape_out, u_shape_out,
                 depth=2):
        super().__init__()
        self.input_dimensions = [n_shape_in, e_shape_in, u_shape_in]
        self.output_dimensions = [n_shape_out, e_shape_out, u_shape_out]
        self.depth = depth

    def forward(self, x, e, u, edge_index, edge_attr=None, batch=None, bond_batch=None):
        pass


class HModel(torch.nn.Module):
    """
    Model for Hamiltonian prediction.

    This model contains 4 blocks
    a) Orbital encoding blok  block
    b) Edge encoding blok
    c) Onsite prediction blok
    d) Interaction prediction block
    """

    def __init__(self, n_shape, e_shape, u_shape):
        super().__init__()
        self.input_dimensions = [n_shape, e_shape, u_shape]
        self.orbital_encoding_block = OrbitalBlock(n_shape_in, e_shape_in, u_shape_in,
                                                   n_shape_out, e_shape_out, u_shape_out)
        self.edge_block = InteractionBlock()
        self.onsite_blok = OnsiteBlock()
        self.interaction_block = HoppingBlock()

    def forward(self, x, e, u, edge_index, edge_attr=None, batch=None, bond_batch=None):
        x = self.orbital_encoding_block(x, e, u, edge_index, edge_attr, batch, bond_batch)
        x2, e = self.edge_block(x, e, u, edge_index, edge_attr, batch, bond_batch)

        x = x + x2

        x = self.onsite_blok(x, e, edge_index, edge_attr, batch, bond_batch)
        e = self.interaction_block(e, u, edge_index, edge_attr, batch, bond_batch)

        return x, e
