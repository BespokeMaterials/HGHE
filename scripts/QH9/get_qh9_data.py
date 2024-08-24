from torch_geometric.loader import DataLoader
from datasets import QH9Stable, QH9Dynamic

### Use one of the following lines to Load the specific dataset
dataset = QH9Stable(split='random')  # QH9-stable-id
# dataset = QH9Stable(split='size_ood')  # QH9-stable-ood
# dataset = QH9Dynamic(split='geometry', version='300k')  # QH9-dynamic-geo
# dataset = QH9Dynamic(split='mol', version='300k')  # QH9-dynamic-mol

### Get the training/validation/testing subsets
train_dataset = dataset[dataset.train_mask]
valid_dataset = dataset[dataset.val_mask]
test_dataset = dataset[dataset.test_mask]

### Get the dataloders
train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_data_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Iterating over the DataLoader:")
for batch_idx, batch in enumerate(train_data_loader):
    print(f"Batch {batch_idx + 1}:")
    print(batch[0])

    print(batch[0].pos)
    print("atoms:",batch[0].atoms)
    print("diag_ham:", batch[0].diagonal_hamiltonian)
    print("non_diagonal_hamiltonian:", batch[0].non_diagonal_hamiltonian)
    print("non_diagonal_hamiltonian_mask:", batch[0].diagonal_hamiltonian_mask)
    print("edge_index_full:", batch[0].edge_index_full )
    break
