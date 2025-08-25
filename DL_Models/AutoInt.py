import torch
import torch.nn as nn
import torch.nn.functional as F


# ===============================
# Feature embedding + AutoInt Block
# ===============================
class AutoIntBlock(nn.Module):
    """
    A single AutoInt self-attention block.

    Args:
        embed_dim (int): Dimension of input embeddings.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability for attention and feed-forward layers.

    Forward:
        x: Tensor of shape [batch, num_features, embed_dim]
        returns: Tensor of same shape [batch, num_features, embed_dim]
    """

    def __init__(self, embed_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, num_features, embed_dim]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ===============================
# AutoInt Model (tunable hyperparameters)
# ===============================
class AutoInt(nn.Module):
    def __init__(
        self,
        num_numeric_features,
        num_categories=None,
        embedding_dims=None,
        attn_blocks=2,  # number of AutoInt blocks
        embed_dim=32,  # embedding dimension
        num_heads=2,  # attention heads
        dropout=0.1,  # dropout in attention & FFN
        output_dim=2,
    ):
        super().__init__()

        # ---- Optional categorical embeddings ----
        if num_categories is not None and embedding_dims is not None:
            self.embeddings = nn.ModuleList(
                [
                    nn.Embedding(num_cat, emb_dim)
                    for num_cat, emb_dim in zip(num_categories, embedding_dims)
                ]
            )
            embedding_total_dim = sum(embedding_dims)
        else:
            self.embeddings = None
            embedding_total_dim = 0

        self.num_features = num_numeric_features + embedding_total_dim

        # Project numeric features to embed_dim
        self.input_proj = nn.Linear(self.num_features, embed_dim)

        # Attention blocks
        self.blocks = nn.ModuleList(
            [AutoIntBlock(embed_dim, num_heads, dropout) for _ in range(attn_blocks)]
        )

        # Output layer
        self.output_layer = nn.Linear(embed_dim, output_dim)

        # Save hyperparameters for reference
        self.hyperparams = {
            "attn_blocks": attn_blocks,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "dropout": dropout,
        }

    def forward(self, x_numeric, x_categ=None):
        # Combine numeric + categorical
        if self.embeddings is not None and x_categ is not None:
            embedded = [emb(x) for emb, x in zip(self.embeddings, x_categ)]
            embedded = torch.cat(embedded, dim=1)
            x = torch.cat([x_numeric, embedded], dim=1)
        else:
            x = x_numeric

        # Project to embedding dimension
        x = self.input_proj(x)  # [batch, embed_dim]

        # Add "feature dimension" for MultiheadAttention
        x = x.unsqueeze(1)  # [batch, num_features=1, embed_dim]

        # Pass through attention blocks
        for block in self.blocks:
            x = block(x)

        # Aggregate features (mean pooling)
        x = x.mean(dim=1)  # [batch, embed_dim]

        # Output
        out = self.output_layer(x)
        return out
