import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from model import HModel
from utils import create_directory_if_not_exists
from torch_geometric.data import InMemoryDataset, Data
from torch.utils.data import random_split
from hghe.plots import matrix_plot


def contains_nan(tensor):
    """
    Check if there are any NaN values in a PyTorch tensor.

    Parameters:
    tensor (torch.Tensor): The input tensor to check.

    Returns:
    bool: True if there is at least one NaN value, False otherwise.
    """
    return torch.isnan(tensor).any().item()


class CustomGraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(CustomGraphDataset, self).__init__()
        self.data_list = data_list
        self.data, self.slices = self.collate(data_list)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def view_eval(pred, targets, edge_index, exp_path, img_id):
    """
    Plotting function
    :param pred:
    :type pred:
    :param targets:
    :type targets:
    :param edge_index:
    :type edge_index:
    :param exp_path:
    :type exp_path:
    :param img_id:
    :type img_id:
    :return:
    :rtype:
    """
    nr_orb = targets[0].shape[1]
    mat_shape = len(pred[0]) * nr_orb

    h_mat = torch.zeros(mat_shape, mat_shape)
    # set onsites:
    for i, p in enumerate(pred[0]):
       h_mat[i * nr_orb:(i + 1) * nr_orb, i * nr_orb:(i + 1) * nr_orb] = p.reshape(nr_orb, nr_orb)


    # set hop:
    for k in range(len(edge_index[0])):
        i = edge_index[0][k]
        j = edge_index[1][k]

        #print(pred[1][k].reshape(nr_orb, nr_orb))
        h_mat[i * nr_orb:(i + 1) * nr_orb, j * nr_orb:(j + 1) * nr_orb] = pred[1][k].reshape(nr_orb, nr_orb)

    h_tar = torch.zeros(mat_shape, mat_shape)
    # set onsites:
    for i, p in enumerate(targets[0]):
        h_tar[i * nr_orb:(i + 1) * nr_orb, i * nr_orb:(i + 1) * nr_orb] = p
    # set hop:
    for k in range(len(edge_index[0])):
        i = edge_index[0][k]
        j = edge_index[1][k]

        h_tar[i * nr_orb:(i + 1) * nr_orb, j * nr_orb:(j + 1) * nr_orb] = targets[1][k]

    dif = h_mat - h_tar

    matrix_plot(dif.detach().numpy(), filename=f"{exp_path}/{img_id}_dif.png", grid1_step=1, grid2_step=nr_orb)
    matrix_plot(h_mat.detach().numpy(), filename=f"{exp_path}/{img_id}_hmat_pred.png", grid1_step=1, grid2_step=nr_orb)
    matrix_plot(h_tar.detach().numpy(), filename=f"{exp_path}/{img_id}_hmat_tar.png", grid1_step=1, grid2_step=nr_orb)


# Training routine
class Trainer:
    def __init__(self, model,
                 train_loader,
                 val_loader,
                 loss_fn,
                 optimizer,
                 device='cpu',
                 eval_storage=""):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.path_exp = eval_storage

    def train(self, num_epochs):
        self.model=self.model.to(self.device)
        for epoch in range(num_epochs):
            # self.evaluate(epoch, num_epochs, 0)
            self.model.train()  # Set the model to training mode
            running_loss = 0.0

            for inputs in self.train_loader:

                self.optimizer.zero_grad()
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"NaNs found in gradients of {name}")
                        if torch.isinf(param.grad).any():
                            print(f"Infs found in gradients of {name}")
                        if torch.max(param.grad).item() > 1e5:
                            print(f"Large gradient values in {name}")

                inputs = inputs.to(self.device)
                # getting the onsite and hoppings
                targets = (inputs.hmat_on, inputs.hmat_hop, inputs.smat_on, inputs.smat_hop)
                # print("Target shapes:", targets[0].shape, targets[1].shape, targets[2].shape, targets[3].shape)
                # # Prediction:
                # print("x contaians NAN:", contains_nan(inputs.x.to(torch.float32)))
                # print("u contains nan:", contains_nan(inputs.u.to(torch.float32)))
                # print("edge_attr contians nan:", contains_nan(inputs.edge_attr.to(torch.float32)))

                pred = self.model(x=inputs.x.to(torch.float32).to(self.device),
                                  u=inputs.u.to(torch.float32).to(self.device),
                                  edge_index=inputs.edge_index.to(torch.int64).to(self.device),
                                  edge_attr=inputs.edge_attr.to(torch.float32).to(self.device),
                                  batch=inputs.batch.to(self.device),
                                  bond_batch=None)
                # compute loss
                loss = self.loss_fn(pred, targets, inputs.edge_index.to(torch.int64))
                loss.backward()
                #optimize
                self.optimizer.step()

                running_loss += loss.item()

                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         if torch.isnan(param.grad).any():
                #             print(f"2:NaNs found in gradients of {name}")
                #         if torch.isinf(param.grad).any():
                #             print(f"2:Infs found in gradients of {name}")
                #         if torch.max(param.grad).item() > 1e5:
                #             print(f"2:Large gradient values in {name}")

            # Validation loop
            if epoch !=0:
                print(f"Epoch {epoch} loss: {running_loss/len(self.train_loader)}")
                self.evaluate(epoch, num_epochs, running_loss)
        return self.model

    def evaluate(self, epoch=0, num_epochs=None, running_loss=None):
        self.model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            img_id = 0
            if self.val_loader is not None and epoch % 5 == 0 or epoch == 1:
                for inputs in self.val_loader:
                    inputs = inputs.to(self.device)
                    targets = (inputs.hmat_on, inputs.hmat_hop, inputs.smat_on, inputs.smat_hop)

                    pred = self.model(x=inputs.x.to(torch.float32).to(self.device),
                                      u=inputs.u.to(torch.float32).to(self.device),
                                      edge_index=inputs.edge_index.to(torch.int64).to(self.device),
                                      edge_attr=inputs.edge_attr.to(torch.float32).to(self.device),
                                      batch=inputs.batch,
                                      bond_batch=None
                                      )

                    loss = self.loss_fn(pred, targets, inputs.edge_index.to(torch.int64))
                    view_eval(
                        pred,
                        targets,
                        inputs.edge_index.to(torch.int64),
                        self.path_exp,
                        f"epoch{epoch}_{img_id}")

                    img_id += 1

                    val_loss += loss.item()

                # Print statistics
                print(f"Epoch {epoch + 1}/{num_epochs}, "
                      f"Train Loss: {running_loss / len(self.train_loader):.4f}, "
                      f"Val Loss: {val_loss / len(self.val_loader):.4f}")

            else:
                # Print statistics
                print(f"Epoch {epoch + 1}/{num_epochs}, "
                      f"Train Loss: {running_loss / len(self.train_loader):.4f}, ")


def ham_difference(pred, target, edge_index):




    #return force_pred(pred, target, edge_index)
    onsite_p = pred[0]
    hop_p = pred[1]


    #------------simetry -----------------
    # Extract indices
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]

    # Perform batched addition
    # Create a zero tensor for accumulation
    accum = torch.zeros_like(hop_p)

    # Sum up the values
    accum[src_nodes] += hop_p[dst_nodes]
    accum[dst_nodes] += hop_p[src_nodes]

    # Update hop_p with accumulated values
    hop_p =(hop_p+ accum)/2
    # ------------simetry -----------------


    onsite_t = target[0].view(onsite_p.shape[0], -1)
    hop_t = target[1].view(hop_p.shape[0], -1)

    onsite_dif = torch.norm(onsite_t - onsite_p) ** 2

    hop_d = hop_t - hop_p
    scale = torch.count_nonzero(hop_t, dim=1)
    zero_count = torch.count_nonzero(hop_p)

    hop_dif = torch.norm(hop_d)**2

    nop_e_norm=torch.norm(hop_d, dim=1)
    hop_d_scale = torch.norm(nop_e_norm* scale)

    variance_penalty_h = (torch.var(hop_p)-torch.var(hop_t))**2*10000
    variance_penalty_o = (torch.var(onsite_p) - torch.var(onsite_t)) ** 2

    node_distances = torch.cdist(hop_p, hop_p, p=2)
    contrastive_penalty = torch.mean(torch.relu(1 - node_distances))

    dif =onsite_dif + hop_d_scale+hop_dif+10*contrastive_penalty
    print(f"onsite dif: {onsite_dif} |\nhop dif:{hop_dif}|\nhop_d_scale: {hop_d_scale}|\nvariance_penalty:{variance_penalty_h}|\ncontrastive_penalty:{contrastive_penalty}|\ntotal dif in module :{dif}|\n\n")
    return dif



def main(device, data_path, save_exp_path):
    # Construct the directories for saving the experiment:
    create_directory_if_not_exists(save_exp_path)
    create_directory_if_not_exists(save_exp_path + "/evaluation")
    create_directory_if_not_exists(save_exp_path + "/model")

    # Load the data
    loaded_data = torch.load(data_path)

    # Now, you can use the loaded data as needed
    print("Data graph:", loaded_data['graphs'][0].data)
    dataset = CustomGraphDataset([g.data for g in loaded_data['graphs']])
    total_size = len(dataset)
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print("len(train_dataset)", len(train_dataset))
    print("len(val_dataset)", len(val_dataset))

    # Construct the model
    model = HModel(orbital_in=[84, 73, 3], orbital_out=[168, 101, 10],
                   interaction_in=[168, 73, 3], interaction_out=[168, 103, 10],
                   onsite_in=[168, 103, 10], onsite_out=[169, 1, 1],
                   hopping_in=[168, 103, 10], hopping_out=[1, 169, 3])

    # Define the optimizer
    train_loader = (DataLoader
                    (train_dataset, batch_size=2, shuffle=True))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    trainer = Trainer(model,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      loss_fn=ham_difference,
                      optimizer=optimizer,
                      device=device,
                      eval_storage=save_exp_path + "/evaluation"
                      )

    model = trainer.train(num_epochs=53)
    torch.save(model.state_dict(), save_exp_path + "/model/" + 'model.pth')

    optimizer = Adam(model.parameters(), lr=1e-5)

    create_directory_if_not_exists(save_exp_path + "/evaluation_ls2")
    trainer = Trainer(model,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      loss_fn=ham_difference,
                      optimizer=optimizer,
                      device=device,
                      eval_storage=save_exp_path + "/evaluation_ls2"
                      )

    model = trainer.train(num_epochs=100)
    torch.save(model.state_dict(), save_exp_path + "/model/" + 'model.pth')

    print("Done !")


main(device="cpu",
     data_path='/Users/voicutomut/Documents/GitHub/HGHE/scripts/aBN_hamiltonian/DATA/DFT/aBN_DFT_CSV/DFT_graphs_8atoms_atomic.pt',
     save_exp_path="test_exp_8atom_2"

     )
