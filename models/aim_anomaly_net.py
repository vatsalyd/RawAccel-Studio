"""
Sequence classifier for normal vs assisted aim detection.

Architecture: optional 1D CNN feature extractor → BiLSTM → classifier head.
"""
import torch
from torch import nn


class AimAnomalyNet(nn.Module):
    """
    Input:  (B, T, F) — time-series of [mouse_dx, dt, view_delta, error]
    Output: (B,)      — logit (positive = assisted)
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        use_cnn: bool = False,
        cnn_channels: int = 64,
        cnn_kernel: int = 5,
    ):
        super().__init__()
        self.use_cnn = use_cnn

        # Optional 1D CNN front-end
        if use_cnn:
            self.cnn = nn.Sequential(
                # (B, F, T) after transpose
                nn.Conv1d(input_dim, cnn_channels, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
                nn.BatchNorm1d(cnn_channels),
                nn.ReLU(),
                nn.Conv1d(cnn_channels, cnn_channels, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
                nn.BatchNorm1d(cnn_channels),
                nn.ReLU(),
            )
            rnn_input = cnn_channels
        else:
            self.cnn = None
            rnn_input = input_dim

        self.rnn = nn.LSTM(
            rnn_input,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cnn is not None:
            # (B, T, F) → (B, F, T) → CNN → (B, C, T) → (B, T, C)
            x = x.transpose(1, 2)
            x = self.cnn(x)
            x = x.transpose(1, 2)

        out, _ = self.rnn(x)          # (B, T, 2*H)
        last = out[:, -1, :]          # last time step
        logits = self.fc(last)        # (B, 1)
        return logits.squeeze(-1)     # (B,)