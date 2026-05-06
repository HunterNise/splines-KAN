"""
[task1/NN2] Model architecture definition — factored out from train.py for reproducibility.

Defines the NN class used by task1/NN2/train.py. Kept in a separate file so that a verbatim
copy is saved to outputs/ at training time, ensuring eval scripts can always reconstruct the
exact architecture that was trained.

Changes vs task1/NN1 architecture:
- Dropout layers (configurable probability) inserted after each hidden ReLU.
- Accepts num_points, dim, and dropout as constructor arguments (were hardcoded in NN1).
"""


import torch
from torch import nn

from source.functions import *



class NN(nn.Module):
    # class contructor to initialize the neural network architecture and parameters
    def __init__(self, num_points=100, dim=2, num_knots=5, num_neurons=128, degree=3, dropout=0.1):
        super().__init__()              # call parent constructor
        
        # store as object variables for later use
        self.num_points = num_points
        self.dim        = dim
        self.num_knots  = num_knots
        self.degree     = degree
        self.dropout    = dropout
        
        # the knot vector must be non-decreasing, so we predict intervals between knots and then convert back to knots
        num_intervals = num_knots - 1
        
        self.stack = nn.Sequential(
            nn.Linear(num_points * dim, num_neurons),   # input layer: takes flattened data points as input
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_neurons, num_neurons),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_neurons, num_neurons),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_neurons, num_intervals),
            nn.Softmax(dim=1)       # ensure intervals are positive and sum to 1; dim=1 applies softmax across the correct dimension for batch processing
        )

    def forward(self, x):
        x = self.stack(x)          # pass through the stack of layers
        # intervals_to_knots only handles 1D; prepend zero column and cumsum manually (batch version)
        zero = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        x = torch.cumsum(torch.cat((zero, x), dim=1), dim=1)  # (batch, num_knots)
        return x
