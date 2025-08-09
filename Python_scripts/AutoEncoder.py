import torch
import torch.nn as nn

# Autoencoder Architecture 1
class Autoencoder(nn.Module):
    """
    Autoencoder for dimensionality reduction and feature extraction.

    Args:
        input_dim (int): Number of features in the input data.
        latent_dim (int): Size of the compressed latent representation.
        hidden_dim (int, optional): Size of the intermediate hidden layer. Default is 128.
        dropout (float, optional): Dropout probability for regularization. Default is 0.4.

    Forward:
        x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
    
    Returns:
        latent (torch.Tensor): Compressed latent representation.
        reconstructed (torch.Tensor): Reconstructed output tensor.
    """
    def __init__(self, input_dim, latent_dim, hidden_dim = 128, dropout = 0.4):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),

            nn.Linear(256, hidden_dim),
            nn.ReLU(0.2),
            nn.BatchNorm1d(hidden_dim),

            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(0.2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(0.2),
            nn.BatchNorm1d(hidden_dim),

            nn.Linear(hidden_dim, 256),
            nn.ReLU(0.2),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.ReLU(0.2),
            nn.BatchNorm1d(512),

            nn.Linear(512, input_dim),
            nn.Sigmoid()  # Sigmoid is used if input is normalized (0-1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed