"""Twenty one-hot inputs, 32 ReLU hidden units and nineteen sum logits."""

import torch
from torch import nn
from torch.nn import functional as F


def addition_suite():
    """Return all 100 ordered pairs, their 20 input indicators, and 19 labels.

    One-hot encoding represents identity, not magnitude: digit 3 has one 1 in
    position 3 and nine zeros. Concatenating two such vectors gives 20 inputs.
    Labels remain integer sums; CrossEntropyLoss expects these class indices.
    """
    pairs = torch.cartesian_prod(torch.arange(10), torch.arange(10))
    inputs = F.one_hot(pairs, num_classes=10).float().reshape(100, 20)
    return pairs, inputs, pairs.sum(dim=1)


class Adder(nn.Module):
    """20 -> hidden ReLU units -> 19 scores, one for every possible sum."""

    def __init__(self, hidden=32):
        super().__init__()
        self.hidden = nn.Linear(20, hidden)
        self.output = nn.Linear(hidden, 19)

    def forward(self, inputs):
        # Raw scores are called logits. Cross entropy includes log-softmax, so
        # applying softmax here would duplicate that operation and change loss.
        return self.output(F.relu(self.hidden(inputs)))
