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
        batch_size = x.size(0)
        logits = (x @ self.feature_selectors.T) - self.thresholds  # (batch, depth)
        probs = torch.sigmoid(logits)                               # (batch, depth)

        # generate all possible leaf paths
        leaf_probs = probs.unsqueeze(-1)                            # (batch, depth, 1)
        complement = 1 - probs.unsqueeze(-1)                        # (batch, depth, 1)
        probs_stack = torch.cat([leaf_probs, complement], dim=-1)   # (batch, depth, 2)

        # recursively compute product over all leaf combinations
        leaf_probs = probs_stack[:, 0]                               # start with first depth
        for d in range(1, self.depth):
            leaf_probs = torch.einsum('bi,bj->bij', leaf_probs, probs_stack[:, d])
            leaf_probs = leaf_probs.reshape(batch_size, -1)

        return leaf_probs  # (batch, 2**depth)

# ===============================
# NODE model for binary classification
# ===============================
class NODE(nn.Module):
    def __init__(self, input_dim, num_trees=64, depth=6):
        super().__init__()
        self.num_trees = num_trees
        self.depth = depth
        self.trees = nn.ModuleList([ObliviousDecisionTree(input_dim, depth) for _ in range(num_trees)])
        self.leaf_weights = nn.Parameter(torch.randn(num_trees, 2**depth, 1))  # binary output (1 logit)

    def forward(self, x):
        outputs = []
        for tree, weights in zip(self.trees, self.leaf_weights):
            leaf_probs = tree(x)                  # (batch, 2^depth)
            outputs.append(leaf_probs @ weights)  # (batch, 1)
        return torch.sum(torch.stack(outputs, dim=0), dim=0)  # (batch, 1)
