import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Union

def create_omics_dataloader(
    X: Union[pd.DataFrame, torch.Tensor],
    y: Union[pd.Series, torch.Tensor],
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a PyTorch DataLoader for omics data.

    Parameters:
    ----------
    X : pd.DataFrame or torch.Tensor
        Features (e.g., omics input data). If DataFrame, it will be converted to torch.FloatTensor.
    y : pd.Series or torch.Tensor
        Labels. If Series, it will be converted to torch.LongTensor.
    batch_size : int, optional (default=32)
        Number of samples per batch.
    shuffle : bool, optional (default=True)
        Whether to shuffle the data each epoch.

    Returns:
    -------
    torch.utils.data.DataLoader
        DataLoader containing the dataset.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


# Train, Test Data
Train_data = pd.read_csv("../data/smoteenn_resampled_train_data.csv")
Test_data = pd.read_csv("../data/smoteenn_resampled_test_data.csv")

BATCH_SIZE = 16
def pass_dataloader():
    X_train = Train_data.drop("Sample_type", axis = 1)
    y_train = Train_data["Sample_type"]

    X_test = Test_data.drop("Sample_type", axis = 1)
    y_test = Test_data["Sample_type"]

    train_dataloader = create_omics_dataloader(X = X_train,
                                               y = y_train,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    test_dataloader = create_omics_dataloader(X = X_test,
                                              y = y_test,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)
    return train_dataloader, test_dataloader