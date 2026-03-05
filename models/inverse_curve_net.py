from typing import Tuple

import torch
from torch import nn


class InverseCurveNet(nn.Module):
    """
    Sequence-to-vector regressor:
    input: (B, T, F) where F ~ [mouse_dx, dt, view_dyaw]
    output: (B, P) curve params
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, num_layers: int = 2, num_params: int = 5):
        super().__init__()
        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_params),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.fc(last)