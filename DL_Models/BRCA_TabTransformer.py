import torch
import torch.nn as nn

class TabTransformer(nn.Module):
    def __init__(self, input_dim,
                 d_model=128,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=256,
                 dropout=0.5):
        super(TabTransformer, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model)  # Project input to d_model size
        self.norm = nn.LayerNorm(d_model)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier architecture for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),   # Projection layer to larger dimension
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),        # Intermediate hidden layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)           # Output: single logit for binary classification
        )

    def forward(self, x):
        x = x.unsqueeze(1)             # Shape: (batch_size, seq_len=1, input_dim)
        x = self.embedding(x)          # Linear projection to d_model
        x = self.norm(x)               # Normalization
        x = self.transformer(x)        # Transformer encoding
        x = x.mean(dim=1)              # Global average pooling
        x = self.classifier(x)         # Output: (batch_size, 1)
        return x
