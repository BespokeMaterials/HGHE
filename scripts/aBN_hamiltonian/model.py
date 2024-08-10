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


class LocalBlock(torch.nn.Module):
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
        x = self.conv1(x, edge_index, edge_attr,  )

        # Apply the second GINEConv layer
        x = self.conv2(x, edge_index, edge_attr, )

        # Apply the TransformerConv layer
        x = self.conv3(x, edge_index, edge_attr,)

        # Apply the GATv2Conv layer
        x = self.conv4(x, edge_index, edge_attr,)

        return x


class InteractionBlock(torch.nn.Module):
    def __init__(self, n_shape_in, e_shape_in, u_shape_in,
                 n_shape_out, e_shape_out, u_shape_out):
        super().__init__()
        self.input_dimensions = [n_shape_in, e_shape_in, u_shape_in]
        self.output_dimensions = [n_shape_out, e_shape_out, u_shape_out]

        class Seq(torch.nn.Module):
            def __init__(self, n_shape_in, e_shape_in, n_shape_out):
                super().__init__()
                self.n_shape_in = n_shape_in
                self.n_shape_out = n_shape_out
                self.e_shape_in = e_shape_in

                # MLP layers for GINEConv
                self.mlp1 = torch.nn.Sequential(
                    torch.nn.Linear(self.n_shape_in, int((self.n_shape_in + self.n_shape_out) / 2)),
                    torch.nn.ReLU(),
                    torch.nn.Linear(int((self.n_shape_in + self.n_shape_out) / 2), self.n_shape_out)
                )

                # Convolutional layers
                self.conv1 = GINEConv(self.mlp1, train_eps=True, edge_dim=self.e_shape_in)
                self.conv2 = TransformerConv(self.n_shape_out, self.n_shape_out, heads=4, edge_dim=self.e_shape_in)
                self.conv3 = TransformerConv(self.n_shape_out*4, self.n_shape_out, heads=4, edge_dim=self.e_shape_in)
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

                print("x-shape:", x.shape)
                return x, edge_attr, u

        # Instantiate the Seq class
        self.seq1 = LineWrapper(Seq(n_shape_in=e_shape_in,
                                    e_shape_in=n_shape_in,
                                    n_shape_out=n_shape_out))
        self.seq2 = LineWrapper(LineWrapper(Seq(n_shape_in=n_shape_in,
                                                e_shape_in=e_shape_in,
                                                n_shape_out=n_shape_out)))
        self.seq3 = LineWrapper(LineWrapper(LineWrapper(Seq(n_shape_in=e_shape_in,
                                                            e_shape_in=n_shape_in,
                                                            n_shape_out=n_shape_out))))

        # Define weights as trainable parameters
        self.weights = torch.nn.Parameter(torch.tensor([0.5, 0.5, 0.5]))

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None, bond_batch=None):
        x1, edge_attr1,u = self.seq1(x, edge_index, edge_attr)

        x2, edge_attr2,u = self.seq2(x, edge_index, edge_attr)
        x3, edge_attr3,u = self.seq3(x, edge_index, edge_attr)

        # Weighted average of outputs
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

    def __init__(self, orbital_in, orbital_out,

                 interaction_in, interaction_out,
                 onsite_in, onsite_out,
                 hopping_in, hopping_out,

                 ):
        super().__init__()

        self.orbital_encoding_block = LocalBlock(orbital_in[0], orbital_in[1], orbital_in[2],
                                                 orbital_out[0], orbital_out[1], orbital_out[2])
        self.interaction_block = InteractionBlock(interaction_in[0], interaction_in[1], interaction_in[2],
                                                  interaction_out[0], interaction_out[1], interaction_out[2])
        self.onsite_blok = OnsiteBlock(onsite_in[0], onsite_in[1], onsite_in[2],
                                       onsite_out[0], onsite_out[1], onsite_out[2])
        self.hopping_block = HoppingBlock(hopping_in[0], hopping_in[1], hopping_in[2],
                                          hopping_out[0], hopping_out[1], hopping_out[2])

    def forward(self, x, u, edge_index, edge_attr=None, batch=None, bond_batch=None):

        x = self.orbital_encoding_block(x, edge_index, edge_attr, u, batch, bond_batch)

        x2, e,edge_index = self.interaction_block(x, edge_index, edge_attr, u, batch, bond_batch)

        x = x + x2

        x = self.onsite_blok(x, edge_index, edge_attr, u, batch, bond_batch)
        e = self.hopping_block(x, edge_index, edge_attr, u, batch, bond_batch)

        return x, e, edge_index
