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
import torch_geometric.nn as pyg_nn
from hghe.graphs import LineWrapper
from hghe.plots import matrix_plot
import math

ALLCLOSE = 0.1


class LocalBlock(torch.nn.Module):
    def __init__(self, n_shape_in, e_shape_in, u_shape_in,
                 n_shape_out, e_shape_out, u_shape_out):
        super().__init__()
        self.input_dimensions = [n_shape_in, e_shape_in, u_shape_in]
        self.output_dimensions = [n_shape_out, e_shape_out, u_shape_out]

        # MLP layers for GINEConv
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(n_shape_in, int((n_shape_in + n_shape_out) / 2)),
            torch.nn.Sigmoid(),
            torch.nn.Linear(int((n_shape_in + n_shape_out) / 2), n_shape_out)
        )

        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(n_shape_out, int(n_shape_out / 2)),
            torch.nn.Sigmoid(),
            torch.nn.Linear(int(n_shape_out / 2), n_shape_out)
        )

        self.mlp3 = torch.nn.Sequential(
            torch.nn.Linear(4*n_shape_out, 2*n_shape_out),
            torch.nn.Sigmoid(),
            torch.nn.Linear(2*n_shape_out, n_shape_out)
        )


        # Convolutional layers
        self.conv1 = GINEConv(self.mlp1, train_eps=True, edge_dim=e_shape_in)
        self.conv2 = GINEConv(self.mlp2, train_eps=True, edge_dim=e_shape_in)
        self.conv3 = GATv2Conv(n_shape_out, 2 * n_shape_out, edge_dim=e_shape_in)
        self.conv4 = TransformerConv(2 * n_shape_out, n_shape_out, heads=4, edge_dim=e_shape_in)
        self.conv5 = TransformerConv(4*n_shape_out, n_shape_out, heads=4, edge_dim=e_shape_in)
        self.conv6 = GINEConv(self.mlp3, train_eps=True, edge_dim=e_shape_in)

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None, bond_batch=None):
        x = self.conv1(x, edge_index, edge_attr, )

        x = self.conv2(x, edge_index, edge_attr, )

        x = self.conv3(x, edge_index, edge_attr, )

        x = self.conv4(x, edge_index, edge_attr, )

        x = self.conv5(x, edge_index, edge_attr, )

        x = self.conv6(x, edge_index, edge_attr, )


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
                    torch.nn.Linear(n_shape_in, int((n_shape_in + n_shape_out) / 2)),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(int((n_shape_in + n_shape_out) / 2), n_shape_out)
                )

                self.mlp2 = torch.nn.Sequential(
                    torch.nn.Linear(n_shape_out, int(n_shape_out * 3)),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(int(n_shape_out * 3), n_shape_out)
                )

                self.mlp3 = torch.nn.Sequential(
                    torch.nn.Linear(4*n_shape_out, n_shape_out * 2),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(2 * n_shape_out, n_shape_out)
                )

                # Convolutional layers
                self.conv1 = GINEConv(self.mlp1, train_eps=True, edge_dim=e_shape_in)
                self.conv2 = GINEConv(self.mlp2, train_eps=True, edge_dim=e_shape_in)
                self.conv3 = GATv2Conv(n_shape_out, 2 * n_shape_out, edge_dim=e_shape_in)
                self.conv4 = TransformerConv(2 * n_shape_out, n_shape_out, heads=4, edge_dim=e_shape_in)
                self.conv5 = TransformerConv(4*n_shape_out, n_shape_out, heads=4,edge_dim=e_shape_in )
                self.conv6 = GINEConv(self.mlp3, train_eps=True, edge_dim=e_shape_in)# edge_dim=e_shape_in)

            def forward(self, x, edge_index, edge_attr=None, u=None, batch=None, bond_batch=None):
                #compare(x, v=2, atol=0.1, text_="sequence  x0:")

                x = self.conv1(x, edge_index,edge_attr  )
                #compare(x, v=2, atol=0.1, text_="sequence  x1:")
                x = self.conv2(x, edge_index,edge_attr )
               # compare(x, v=2, atol=0.1, text_="sequence  x2:")
                x = self.conv3(x, edge_index, edge_attr)
                #compare(x, v=2, atol=0.1, text_="sequence  x3:")
                x = self.conv4(x, edge_index, edge_attr)
                #compare(x, v=2, atol=0.1, text_="sequence  x4:")
                #x = self.conv5(x, edge_index,edge_attr )
                #compare(x, v=2, atol=0.1, text_="sequence  x5:")
                x = self.conv6(x, edge_index, edge_attr)

                # compare(x, v=2, atol=0.1, text_="sequence  x:")
               # compare(edge_attr, v=2, atol=0.1, text_="sequence  edge_attr:")

                return x, edge_attr, u

        class NodeUP(torch.nn.Module):
            def __init__(self, n_shape_in, e_shape_in, n_shape_out):
                super().__init__()
                self.n_shape_in = n_shape_in
                self.n_shape_out = n_shape_out
                self.e_shape_in = e_shape_in

                self.mlp1 = torch.nn.Sequential(
                    torch.nn.Linear(n_shape_in, int((n_shape_in + n_shape_out) / 2)),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(int((n_shape_in + n_shape_out) / 2), n_shape_out)
                )

                # Convolutional layers
                self.conv1 = GINEConv(self.mlp1, train_eps=True, edge_dim=e_shape_in)

            def forward(self, x, edge_index, edge_attr=None, u=None, batch=None, bond_batch=None):

                x = self.conv1(x, edge_index, edge_attr, )
                return x, edge_attr, u

        # Instantiate the Seq class
        self.seq1 = LineWrapper(Seq(n_shape_in=e_shape_in,
                                    e_shape_in=n_shape_in,
                                    n_shape_out=e_shape_out),
                                NodeUP(n_shape_in=n_shape_in,
                                       e_shape_in=e_shape_out,
                                       n_shape_out=n_shape_out)
                                )
        self.seq2 = LineWrapper(Seq(n_shape_in=e_shape_in,
                                    e_shape_in=n_shape_in,
                                    n_shape_out=e_shape_out),
                                NodeUP(n_shape_in=n_shape_in,
                                       e_shape_in=e_shape_out,
                                       n_shape_out=n_shape_out)
                                )
        self.seq3 = LineWrapper(Seq(n_shape_in=e_shape_in,
                                    e_shape_in=n_shape_in,
                                    n_shape_out=e_shape_out),
                                NodeUP(n_shape_in=n_shape_in,
                                       e_shape_in=e_shape_out,
                                       n_shape_out=n_shape_out)
                                )

        # Define weights as trainable parameters
        self.weights = torch.nn.Parameter(torch.tensor([10.0, 10.0, 10.0]))

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None, bond_batch=None):
        x1, edge_attr1, u = self.seq1(x, edge_index, edge_attr)
        x2, edge_attr2, u = self.seq2(x, edge_index, edge_attr)
        x3, edge_attr3, u = self.seq3(x, edge_index, edge_attr)



        # Weighted average of outputs
        edge_attr = (edge_attr1 * self.weights[0] + edge_attr2 * self.weights[1] + edge_attr3 * self.weights[2]) / 3
        x = x1#(x1 * self.weights[0] + x2 * self.weights[1] + x3 * self.weights[2]) / 3

        #compare(x, v=2, atol=0.1, text_="interact  x:")
        # compare(edge_attr, v=2, atol=0.1, text_="edg  x:")
        return x, edge_attr, edge_index


class OnsiteBlock(torch.nn.Module):
    def __init__(self, n_shape_in, e_shape_in, u_shape_in,
                 n_shape_out, e_shape_out, u_shape_out):
        super().__init__()
        self.input_dimensions = [n_shape_in, e_shape_in, u_shape_in]
        self.output_dimensions = [n_shape_out, e_shape_out, u_shape_out]

        # MLP layers for GINEConv
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(n_shape_in, int((n_shape_in + n_shape_out) / 2)),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(int((n_shape_in + n_shape_out) / 2), n_shape_out)
        )

        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(n_shape_out, int(n_shape_out * 2)),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(int(n_shape_out * 2), n_shape_out)
        )
        self.mlp3 = torch.nn.Sequential(
            torch.nn.Linear(4*n_shape_out, n_shape_out * 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2 * n_shape_out, n_shape_out)
        )

        # Convolutional layers
        self.conv1 = GINEConv(self.mlp1, train_eps=True, edge_dim=e_shape_in)
        self.conv2 = GINEConv(self.mlp2, train_eps=True, edge_dim=e_shape_in)
        self.conv3 = GATv2Conv(n_shape_out,  n_shape_out, edge_dim=e_shape_in)
        self.conv4 = TransformerConv( n_shape_out, n_shape_out, heads=4, edge_dim=e_shape_in)
        #self.conv5 = TransformerConv(4*n_shape_out, n_shape_out, heads=1, edge_dim=e_shape_in)
        self.conv6 = GINEConv(self.mlp3, train_eps=True, edge_dim=e_shape_in)

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None, bond_batch=None):
        x = self.conv1(x, edge_index, edge_attr, )

        x = self.conv2(x, edge_index, edge_attr, )

        # x = self.conv3(x, edge_index, edge_attr, )

        x = self.conv4(x, edge_index, edge_attr, )

        # x = self.conv5(x, edge_index, edge_attr, )

        x = self.conv6(x, edge_index, edge_attr, )

        return x


class HoppingBlock(torch.nn.Module):
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
                    torch.nn.Linear(self.n_shape_in, self.n_shape_in + self.n_shape_out),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.n_shape_in + self.n_shape_out,int((self.n_shape_in + self.n_shape_out) / 2)),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(int((self.n_shape_in + self.n_shape_out) / 2), self.n_shape_out*4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.n_shape_out * 4, self.n_shape_out * 2),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.n_shape_out * 2, self.n_shape_out )
                )


                self.mlp2 = torch.nn.Sequential(
                    torch.nn.Linear(self.n_shape_out , self.n_shape_out *2),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.n_shape_out * 2, self.n_shape_out )
                )

                self.mlp3 = torch.nn.Sequential(
                    torch.nn.Linear(self.n_shape_out, 2*self.n_shape_out),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(2*self.n_shape_out, self.n_shape_out),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.n_shape_out, self.n_shape_out)
                )

                # Convolutional layers
                self.conv1 = pyg_nn.GATConv(n_shape_in, n_shape_out, heads=4, concat=False,
                                            edge_dim=e_shape_in)  # Replacing GINEConv with GATConv

                self.conv3 = pyg_nn.GATv2Conv(n_shape_out, n_shape_out, edge_dim=e_shape_in)  # Keeping GATv2Conv
                self.conv4 = pyg_nn.ARMAConv(n_shape_out, n_shape_out,
                                             )  # Replacing TransformerConv with ARMAConv
                self.conv5 = pyg_nn.ARMAConv(n_shape_out, n_shape_out,
                                             )  # Replacing TransformerConv with ChebConv
                self.conv6 = GINEConv(self.mlp3, train_eps=True, edge_dim=self.e_shape_in)

            def forward(self, x, edge_index, edge_attr=None, u=None, batch=None, bond_batch=None):
                # compare(x, v=2, atol=0.1, text_="hopping in  x:")
                x = self.conv1(x, edge_index, edge_attr)

                # x = self.conv2(x, edge_index, edge_attr)
                # compare(x, v=2, atol=0.1, text_="hopping in  x1:")
                x = self.conv3(x, edge_index, edge_attr)
                # compare(x, v=2, atol=0.1, text_="hopping in  x3:")
                x = self.conv4(x, edge_index)
                # compare(x, v=2, atol=0.1, text_="hopping in  x4:")
                x = self.conv5(x, edge_index)
                # compare(x, v=2, atol=0.1, text_="hopping  x5:")

                x = self.conv6(x, edge_index, edge_attr)
                # compare(x, v=2, atol=0.1, text_="hopping  x:")

                return x, edge_attr, u

        class NodeUP(torch.nn.Module):
            def __init__(self, n_shape_in, e_shape_in, n_shape_out):
                super().__init__()
                self.n_shape_in = n_shape_in
                self.n_shape_out = n_shape_out
                self.e_shape_in = e_shape_in

                self.mlp1 = torch.nn.Sequential(
                    torch.nn.Linear(n_shape_in, int((n_shape_in + n_shape_out) / 2)),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(int((n_shape_in + n_shape_out) / 2), n_shape_out)
                )

                # Convolutional layers
                self.conv1 = GINEConv(self.mlp1, train_eps=True, edge_dim=e_shape_in)

            def forward(self, x, edge_index, edge_attr=None, u=None, batch=None, bond_batch=None):
                x = self.conv1(x, edge_index, edge_attr, )
                return x, edge_attr, u

        # Instantiate the Seq class
        self.seq1 = LineWrapper(Seq(n_shape_in=e_shape_in,
                                    e_shape_in=n_shape_in,
                                    n_shape_out=e_shape_out),
                                NodeUP(n_shape_in=n_shape_in,
                                       e_shape_in=e_shape_out,
                                       n_shape_out=n_shape_out)
                                )

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None, bond_batch=None):
        x, edge_attr, u = self.seq1(x, edge_index, edge_attr)

        return edge_attr


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

        x1 = self.orbital_encoding_block(x, edge_index, edge_attr, u, batch, bond_batch)
        # compare(x1, v=2, atol=0.1, text_="orbital_encoding_block comparation x:")
        # compare(edge_attr, v=2, atol=0.1, text_="orbital_encoding_block edge_attr:")

        x2, e, edge_index_0 = self.interaction_block(x1, edge_index, edge_attr, u, batch, bond_batch)
        #compare(x2, v=2, atol=0.1, text_="interaction_block comparation x:")
        #compare(e, v=2, atol=0.1, text_="interaction_block edge_attr e:")
        x = x1 + x2



        xo = self.onsite_blok(x1, edge_index, e, u, batch, bond_batch)

        e = self.hopping_block(x, edge_index, e, u, batch, bond_batch)

        return xo, e, edge_index


# helping finction
def compare(vector, v=2, atol=0.1, text_="Compare"):
    for i, xi in enumerate(vector):
        print(f"{text_} {i}-{v}:{torch.allclose(xi, vector[v])}|{torch.norm(xi - vector[v])}", )
