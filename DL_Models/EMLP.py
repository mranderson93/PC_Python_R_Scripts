import torch
import torch.nn as nn
import torch.nn.functional as F

class OmicsMLP(nn.Module):
    def __init__(
        self,
        num_numeric_features,
        num_categories=None,
        embedding_dims=None,
        hidden_units=[512, 256, 128],
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

        # Total input = numeric omics features + optional embeddings
        input_dim = num_numeric_features + embedding_total_dim

        # ---- Trainable "hyperparameters" ----
        # dropout rate (bounded in [0,1] using sigmoid)
        self.logit_dropout = nn.Parameter(torch.tensor(0.0))

        # scaling factor for hidden layer sizes
        self.scale = nn.Parameter(torch.tensor(1.0))

        # store base hidden_units (will be scaled dynamically)
        self.base_hidden_units = hidden_units

        # Define layers dynamically (use max sizes)
        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        prev_dim = input_dim

        for hidden_dim in hidden_units:
            max_dim = hidden_dim  # maximum width
            self.linears.append(nn.Linear(prev_dim, max_dim))
            self.bns.append(nn.BatchNorm1d(max_dim))
            prev_dim = max_dim

        self.final = nn.Linear(prev_dim, 1)

    def forward(self, x_numeric, x_categ=None):
        # Combine numeric + categorical
        if self.embeddings is not None and x_categ is not None:
            embedded = [emb(x_c) for emb, x_c in zip(self.embeddings, x_categ)]
            embedded = torch.cat(embedded, dim=1)
            x = torch.cat([x_numeric, embedded], dim=1)
        else:
            x = x_numeric

        # compute dropout probability from learnable parameter
        dropout_p = torch.sigmoid(self.logit_dropout).item()  # convert tensor → float

        # forward through dynamically scaled layers
        for linear, bn, base_dim in zip(self.linears, self.bns, self.base_hidden_units):
            # determine actual active width (scale factor applied)
            active_dim = int(base_dim * torch.clamp(self.scale, 0.25, 2.0).item())
            # slice weight & bias to active size
            weight = linear.weight[:active_dim, :]
            bias = linear.bias[:active_dim]
            x = F.linear(x, weight, bias)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=dropout_p, training=self.training)

        return self.final(x)
