import torch
from torch import nn


class AimAnomalyNet(nn.Module):
    """
    Sequence classifier: normal vs assisted.
    Input: (B, T, F)
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 128, num_layers: int = 1):
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
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits.squeeze(-1)