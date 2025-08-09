# A file dedicated to helpful utility functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import TensorDataset, DataLoader

def plot_loss_autoencoder(train_losses):
    """Plots the training and test loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training  Over Epochs For AutoEncoder')
    plt.legend()
    plt.show()

# Data Loader For AutoEncoder
# meth_tensor = torch.tensor(meth.values, dtype=torch.float32)
def create_dataloader_autoencoder(X, y, batch_size = 32, shuffle = True):
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    tensor_data = TensorDataset(X_tensor, y_tensor)
    return DataLoader(tensor_data, batch_size=batch_size, shuffle=shuffle)
