"""
Sequence-to-vector regressor for predicting accel curve parameters from
observed mouse + view time-series data.

Supports two pooling strategies:
- 'last': use final LSTM hidden state (default)
- 'attention': learned attention pooling over all time steps
"""
import torch
from torch import nn


class AttentionPool(nn.Module):
    """Learnable attention pooling over time dimension."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, H)
        returns : (B, H)
        """
        weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return (weights * x).sum(dim=1)                # (B, H)


class InverseCurveNet(nn.Module):
    """
    Input:  (B, T, F) — time-series of [mouse_dx, dt, view_delta, ...]
    Output: (B, P)   — predicted curve parameters
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_params: int = 5,
        dropout: float = 0.1,
        use_attention: bool = False,
    ):
        super().__init__()
        self.use_attention = use_attention

        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        pooled_dim = hidden_dim * 2  # bidirectional

        if use_attention:
            self.pool = AttentionPool(pooled_dim)
        else:
            self.pool = None

        self.fc = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_params),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)  # (B, T, 2*hidden)

        if self.pool is not None:
            pooled = self.pool(out)        # attention pooling
        else:
            pooled = out[:, -1, :]         # last hidden state

        return self.fc(pooled)