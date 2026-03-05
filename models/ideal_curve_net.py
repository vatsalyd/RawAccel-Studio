import torch
from torch import nn


class IdealCurveNet(nn.Module):
    """
    Map player/session features to accel curve params.
    Input features could be:
      [avg_accuracy, avg_reaction_time, tracking_score, flick_score, ...]
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 128, num_params: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_params),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)