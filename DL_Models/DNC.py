import torch
import torch.nn as nn
import torch.nn.functional as F


class DNC(nn.Module):
    """
    Differentiable Neural Computer (DNC) for tabular/omics data.

    Args:
        input_dim (int): Number of input features (numeric + embeddings).
        hidden_dim (int): Hidden dimension of controller RNN.
        output_dim (int): Output dimension (e.g., 1 for binary classification).
        memory_units (int): Number of memory slots.
        memory_unit_size (int): Size of each memory slot.
        read_heads (int): Number of read heads.
        num_layers (int): Number of RNN layers in controller.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        output_dim=2,
        memory_units=32,
        memory_unit_size=64,
        read_heads=4,
        num_layers=1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.memory_units = memory_units
        self.memory_unit_size = memory_unit_size
        self.read_heads = read_heads
        self.num_layers = num_layers

        # Controller RNN
        self.controller = nn.LSTM(
            input_dim + read_heads * memory_unit_size,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Memory parameters
        self.memory = nn.Parameter(torch.zeros(memory_units, memory_unit_size))
        self.read_weights = nn.Parameter(torch.zeros(read_heads, memory_units))
        self.write_weights = nn.Parameter(torch.zeros(memory_units))
        self.read_vectors = nn.Parameter(torch.zeros(read_heads, memory_unit_size))

        # Output layer
        self.fc_out = nn.Linear(hidden_dim + read_heads * memory_unit_size, output_dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): [batch, seq_len, input_dim] input tensor
        Returns:
            logits (torch.Tensor): [batch, output_dim]
        """
        batch_size, seq_len, _ = x.size()
        read_vectors = self.read_vectors.unsqueeze(0).repeat(batch_size, 1, 1)
        outputs = []

        # Initialize controller hidden states
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        for t in range(seq_len):
            xt = x[:, t, :]
            rcat = torch.cat([xt, read_vectors.view(batch_size, -1)], dim=-1).unsqueeze(
                1
            )
            out, (h, c) = self.controller(rcat, (h, c))

            # Simple memory update: write and read
            # For simplicity, using softmax over memory for reading
            mem = F.softmax(self.memory, dim=0)
            read_vectors = (
                torch.einsum("ij,kj->ki", mem, out.squeeze(1))
                .unsqueeze(0)
                .repeat(self.read_heads, 1, 1)
            )

            out_combined = torch.cat(
                [out.squeeze(1), read_vectors.view(batch_size, -1)], dim=-1
            )
            outputs.append(self.fc_out(out_combined))

        return torch.stack(outputs, dim=1).mean(dim=1)  # aggregate sequence
