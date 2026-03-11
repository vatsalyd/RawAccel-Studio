"""
MLP that maps player/session behavioural features to recommended
accel curve parameters.

Enhanced with batch normalisation, dropout, and output activation
that constrains predictions to valid parameter ranges.
"""
import torch
from torch import nn

from models.curve_param_config import PARAM_BOUNDS, TUNE_KEYS


class IdealCurveNet(nn.Module):
    """
    Input:  (B, D)  — player/session feature vector
    Output: (B, P)  — predicted accel curve parameters (in valid ranges)
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 128,
        num_params: int = 5,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        constrain_output: bool = True,
    ):
        super().__init__()
        self.constrain_output = constrain_output
        self.num_params = num_params

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, num_params))
        self.net = nn.Sequential(*layers)

        # Store param bounds for output clamping
        keys = TUNE_KEYS[:num_params]
        self.register_buffer(
            "param_lo",
            torch.tensor([PARAM_BOUNDS[k][0] for k in keys], dtype=torch.float32),
        )
        self.register_buffer(
            "param_hi",
            torch.tensor([PARAM_BOUNDS[k][1] for k in keys], dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        if self.constrain_output:
            # Sigmoid → scale to [lo, hi]
            return self.param_lo + torch.sigmoid(raw) * (self.param_hi - self.param_lo)
        return raw