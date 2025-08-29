import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Global Variables
BATCH_SIZE = 32


# meth_tensor = torch.tensor(meth.values, dtype=torch.float32)
def create_dataloader(X, y, batch_size=32, shuffle=True):
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    tensor_data = TensorDataset(X_tensor, y_tensor)
    return DataLoader(tensor_data, batch_size=batch_size, shuffle=shuffle)


def return_dataloader(BATCH_SIZE=32):
    train_df = pd.read_csv(
        "./drive/MyDrive/BRCA/brca_data/Training_Early_Fused_data.csv"
    )
    test_df = pd.read_csv("./drive/MyDrive/BRCA/brca_data/Testing_Early_Fused_data.csv")

    X_train, y_train = train_df.drop("Sample_type", axis=1), train_df["Sample_type"]
    X_test, y_test = test_df.drop("Sample_type", axis=1), test_df["Sample_type"]

    train_dataloader = create_dataloader(
        X_train, y_train, batch_size=BATCH_SIZE, shuffle=True
    )
    test_dataloader = create_dataloader(
        X_test, y_test, batch_size=BATCH_SIZE, shuffle=False
    )

    return train_dataloader, test_dataloader
