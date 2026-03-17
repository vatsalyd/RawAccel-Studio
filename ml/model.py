"""
Neural network for predicting RawAccel curve parameters from mouse features.

Architecture:
    Input: feature vector (100-dim)
    → Shared MLP backbone with residual connections
    → Two heads:
        1. Curve style classifier (6-way: Linear/Classic/Natural/Power/Synchronous/Jump)
        2. Continuous parameter regressor (12 normalized params)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.dataset import NUM_STYLES, NUM_CONTINUOUS
from ml.features import get_feature_dim


class ResBlock(nn.Module):
    """Residual MLP block with LayerNorm and dropout."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class AccelPredictor(nn.Module):
    """
    Predicts RawAccel curve type + parameters from mouse movement features.

    Two-head architecture:
    - Style head: classifies which acceleration curve type (Linear, Classic, etc.)
    - Params head: regresses the continuous parameters (acceleration, cap, offset, etc.)
    """

    def __init__(self, feature_dim: int = None, hidden_dim: int = 256,
                 n_blocks: int = 4, dropout: float = 0.15):
        super().__init__()

        if feature_dim is None:
            feature_dim = get_feature_dim()

        # ── Shared backbone ──
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.backbone = nn.Sequential(
            *[ResBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )

        self.norm = nn.LayerNorm(hidden_dim)

        # ── Style classification head ──
        self.style_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, NUM_STYLES),
        )

        # ── Continuous parameter regression head ──
        self.params_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, NUM_CONTINUOUS),
            nn.Sigmoid(),  # output in [0, 1] (normalized params)
        )

    def forward(self, features: torch.Tensor):
        """
        Args:
            features: (batch, feature_dim)

        Returns:
            style_logits: (batch, NUM_STYLES) — raw logits for curve type
            params_pred: (batch, NUM_CONTINUOUS) — normalized [0,1] params
        """
        x = self.input_proj(features)
        x = self.backbone(x)
        x = self.norm(x)

        style_logits = self.style_head(x)
        params_pred = self.params_head(x)

        return style_logits, params_pred
