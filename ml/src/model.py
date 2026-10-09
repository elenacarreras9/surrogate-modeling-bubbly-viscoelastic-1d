"""MLP surrogate for (p0, G) -> (p2, p3).

A single network with a shared trunk and two outputs: with only a few
hundred points and two quantities that come from the same nonlinear
physics, sharing the hidden layers makes better use of the data than two
independent networks.
"""

import torch
import torch.nn as nn


class SurrogateMLP(nn.Module):
    """Input:   [p0_norm, G_norm]                 (2 normalized features)
    Output:     [log_p2_norm, log_p3_norm]         (2 standardized outputs)

    The input/output normalization lives outside the network (see
    dataset.py, class Scaler) so that it can be saved next to the checkpoint
    and exactly the same transformation can be applied at inference time.
    """

    def __init__(self, n_hidden: int = 64, n_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(2, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(n_hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))
