import torch
import torch.nn as nn
import torch.nn.functional as F


class ObliviousDecisionTree(nn.Module):
    def __init__(self, input_dim, depth):
        super().__init__()
        self.depth = depth
        self.feature_selectors = nn.Parameter(torch.randn(depth, input_dim))
        self.thresholds = nn.Parameter(torch.zeros(depth))

    def forward(self, x):
        # routing probabilities
        logits = (x @ self.feature_selectors.T) - self.thresholds
        probs = torch.sigmoid(logits)

        # All possible leaf paths = 2^depths
        leaf_probs = torch.prod(
            torch.stack([probs, 1 - probs], dim=-1), dim=1
        )  # Shape: (batch, 2^depth)
        return leaf_probs


class NODE(nn.Module):
    def __init__(self, input_dim, num_trees=64, depth=6, output_dim=2):
        super().__init__()
        self.trees = nn.ModuleList(
            [ObliviousDecisionTree(input_dim, depth) for _ in range(num_trees)]
        )
        self.leaf_weights = nn.Parameter(torch.randn(num_trees, 2**depth, output_dim))

    def forward(self, x):
        outputs = []
        for tree, weights in zip(self.trees, self.leaf_weights):
            leaf_probs = tree(x)  # (batch, 2^depth)
            outputs.append(leaf_probs @ weights)
        return torch.sum(torch.stack(outputs, dim=0), dim=0)
