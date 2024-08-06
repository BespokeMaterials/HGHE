import torch


# Data

import torch

# Specify the path where your data was saved
save_path = './DATA/DFT/aBN_DFT_CSV/DFT_graphs_64atoms.pt'

# Load the data
loaded_data = torch.load(save_path)

# Now, you can use the loaded data as needed
print(loaded_data['graphs'])
print(len(loaded_data['graphs']))


print(loaded_data['graphs'][0].data)