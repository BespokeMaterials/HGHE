import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from model import HModel
from utils import create_directory_if_not_exists
from torch_geometric.data import InMemoryDataset, Data
from torch.utils.data import random_split
class CustomGraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(CustomGraphDataset, self).__init__()
        self.data_list = data_list
        self.data, self.slices = self.collate(data_list)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# Training routine
class Trainer:
    def __init__(self, model,
                 train_loader,
                 val_loader,
                 loss_fn,
                 optimizer,
                 device='cpu'):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0

            # Training loop
            for inputs in self.train_loader:
                self.optimizer.zero_grad()

                inputs = inputs.to(self.device)
                # getting the onsite and hoppings
                targets = (inputs.hmat_on, inputs.hmat_hop, inputs.smat_on, inputs.smat_hop)
                print("Target shapes:",targets[0].shape, targets[1].shape, targets[2].shape,targets[3].shape)
                # Prediction:

                pred = self.model(x=inputs.x.to(torch.float32),
                                  u=inputs.u.to(torch.float32),
                                  edge_index=inputs.edge_index.to(torch.int64),
                                  edge_attr=inputs.edge_attr.to(torch.float32),
                                  batch=inputs.batch,
                                  bond_batch=None)
                # compute loss
                loss = self.loss_fn(pred, targets)
                loss.backward()
                #optimize
                self.optimizer.step()
                running_loss += loss.item()

            # Validation loop
            self.evaluate(epoch, num_epochs, running_loss)
            return self.model

    def evaluate(self, epoch=0, num_epochs=None, running_loss=None):
        self.model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            if self.val_loader is not None and epoch % 5 == 0:
                for inputs in self.val_loader:
                    inputs = inputs.to(self.device)
                    targets = (inputs.onsite, inputs.hop)

                    pred = self.model(x=inputs.x.to(torch.float32),
                                      u=inputs.u.to(torch.float32),
                                      edge_index=inputs.edge_index.to(torch.int64),
                                      edge_attr=inputs.edge_attr.to(torch.float32),
                                      batch=inputs.batch,
                                      bond_batch=inputs.bond_batch)

                    loss = self.loss_fn(pred, targets)
                    val_loss += loss.item()

                # Print statistics
                print(f"Epoch {epoch + 1}/{num_epochs}, "
                      f"Train Loss: {running_loss / len(self.train_loader):.4f}, "
                      f"Val Loss: {val_loss / len(self.val_loader):.4f}")

            else:
                # Print statistics
                print(f"Epoch {epoch + 1}/{num_epochs}, "
                      f"Train Loss: {running_loss / len(self.train_loader):.4f}, ")



def ham_difference():
    pass

def main(device,data_path, save_exp_path):

    # Construct the directories for saving the experiment:
    create_directory_if_not_exists(save_exp_path)
    create_directory_if_not_exists(save_exp_path+"/evaluation")
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



    # Construct the model
    model = HModel(orbital_in=[87, 73, 3], orbital_out=[100, 100, 10],
                   interaction_in=[100, 73, 3], interaction_out=[100, 100, 10],
                   onsite_in=[100, 100, 10], onsite_out=[169, 1, 1],
                   hopping_in=[100, 100, 10], hopping_out=[1, 169, 1])

    # Define the optimizer
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    optimizer = Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      loss_fn=ham_difference,
                      optimizer=optimizer,
                      device=device)

    model = trainer.train(num_epochs=100)
    torch.save(model.state_dict(), save_exp_path + "/model/"+'model.pth')


    print("Done !")


main(device="cpu",
    data_path='/Users/voicutomut/Documents/GitHub/HGHE/scripts/aBN_hamiltonian/DATA/DFT/aBN_DFT_CSV/DFT_graphs_64atoms_atomic.pt',
    save_exp_path="test_exp"

     )