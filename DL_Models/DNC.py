import torch
import torch.nn as nn
import torch.nn.functional as F

class DNC(nn.Module):
    """
    Differentiable Neural Computer (DNC) for tabular/omics data.

    Args:
        input_dim (int): Number of input features (numeric + embeddings).
        hidden_dim (int): Hidden dimension of controller RNN.
        output_dim (int): Output dimension (1 for binary classification).
        memory_units (int): Number of memory slots.
        memory_unit_size (int): Size of each memory slot.
        read_heads (int): Number of read heads.
        num_layers (int): Number of RNN layers in controller.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        output_dim=1,           # single output for BCE
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

        # Controller LSTM
        self.controller = nn.LSTM(
            input_dim + read_heads * memory_unit_size,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Memory parameters
        self.memory = nn.Parameter(torch.zeros(memory_units, memory_unit_size))

        # Output layer
        self.fc_out = nn.Linear(hidden_dim + memory_unit_size, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): [batch, input_dim]
        Returns:
            logits (torch.Tensor): [batch]
        """
        batch_size = x.size(0)

        # Initialize LSTM hidden states
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        # Expand input to sequence length 1
        x_seq = x.unsqueeze(1)  # [batch, seq_len=1, input_dim]

        # Initial read vectors
        read_vectors = torch.zeros(batch_size, self.read_heads * self.memory_unit_size).to(x.device)
        lstm_input = torch.cat([x_seq, read_vectors.unsqueeze(1)], dim=-1)  # [batch, seq_len=1, input+read]

        # Pass through LSTM controller
        out, (h, c) = self.controller(lstm_input, (h, c))  # [batch, 1, hidden_dim]

        # Memory read: mean of memory slots
        mem = F.softmax(self.memory, dim=0)  # [memory_units, memory_unit_size]
        read_vectors = mem.unsqueeze(0).repeat(batch_size, 1, 1).mean(dim=1)  # [batch, memory_unit_size]

        # Concatenate LSTM output with read vectors
        combined = torch.cat([out.squeeze(1), read_vectors], dim=-1)  # [batch, hidden_dim + memory_unit_size]

        # Output layer
        logits = self.fc_out(combined)  # [batch, 1]
        return logits.squeeze(-1)       # [batch] for BCEWithLogitsLoss
