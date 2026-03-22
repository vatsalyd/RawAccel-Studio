"""
High-end model for predicting RawAccel curves from mouse movement.

Architecture: 1D-CNN + Multi-Head Self-Attention + MLP

Three-stage design:
  Stage 1 — 1D CNN: Extract local motion patterns from raw time-series
            (speed bursts, micro-corrections, flick signatures)
  Stage 2 — Transformer Encoder: Capture global temporal dependencies
            (playstyle patterns, acceleration behavior over time)
  Stage 3 — Dual-head MLP: Predict curve type + continuous parameters

Compared to the basic model:
  - Works on BOTH raw time-series AND hand-crafted features
  - ~4M parameters (vs 1.15M)
  - Multi-head attention for temporal reasoning
  - Gradient highway via residual connections throughout
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.dataset import NUM_STYLES, NUM_CONTINUOUS
from ml.features import get_feature_dim


# ─── Building Blocks ────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """1D convolution block with GroupNorm, GELU, and residual connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=kernel // 2),
            nn.GroupNorm(min(8, out_ch), out_ch),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1, stride=stride) if in_ch != out_ch or stride != 1 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the transformer."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # handle odd d_model
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Standard transformer encoder block with pre-norm."""

    def __init__(self, d_model: int, n_heads: int = 8, ff_dim: int = None,
                 dropout: float = 0.1):
        super().__init__()
        ff_dim = ff_dim or d_model * 4

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Pre-norm feedforward
        x = x + self.ff(self.norm2(x))
        return x


class MLPHead(nn.Module):
    """Classification/Regression head with LayerNorm and dropout."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.15, final_activation=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )
        self.final_activation = final_activation

    def forward(self, x):
        x = self.net(x)
        if self.final_activation:
            x = self.final_activation(x)
        return x


# ─── Main Models ────────────────────────────────────────────────────────────

class AccelPredictorCNN(nn.Module):
    """
    1D-CNN + Transformer + MLP — operates on raw time-series sequences.

    Input: (batch, 3, seq_len) where 3 channels = [dx, dy, speed]
    """

    def __init__(self, seq_len: int = 1024, d_model: int = 256,
                 n_heads: int = 8, n_transformer_layers: int = 4,
                 dropout: float = 0.15):
        super().__init__()

        # ── Stage 1: 1D CNN backbone ──
        self.cnn = nn.Sequential(
            ConvBlock(3, 64, kernel=7),
            ConvBlock(64, 128, kernel=5, stride=2),    # downsample: seq_len/2
            ConvBlock(128, 256, kernel=5, stride=2),   # downsample: seq_len/4
            ConvBlock(256, d_model, kernel=3, stride=2),  # downsample: seq_len/8
        )

        # ── Stage 2: Transformer encoder ──
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len // 8 + 1, dropout=dropout)
        self.transformer = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, dropout=dropout)
              for _ in range(n_transformer_layers)]
        )
        self.pool_norm = nn.LayerNorm(d_model)

        # ── Stage 3: Dual-head prediction ──
        self.style_head = MLPHead(d_model, 256, NUM_STYLES, dropout=dropout)
        self.params_head = MLPHead(d_model, 256, NUM_CONTINUOUS, dropout=dropout,
                                   final_activation=nn.Sigmoid())

    def forward(self, x):
        """
        Args:
            x: (batch, 3, seq_len) — channels are [dx, dy, speed]
        Returns:
            style_logits: (batch, NUM_STYLES)
            params_pred: (batch, NUM_CONTINUOUS)
        """
        # CNN: (batch, 3, seq_len) → (batch, d_model, seq_len//8)
        features = self.cnn(x)

        # Reshape for transformer: (batch, d_model, T) → (batch, T, d_model)
        features = features.permute(0, 2, 1)
        features = self.pos_enc(features)

        # Transformer
        features = self.transformer(features)

        # Global average pooling → (batch, d_model)
        pooled = features.mean(dim=1)
        pooled = self.pool_norm(pooled)

        # Dual heads
        style_logits = self.style_head(pooled)
        params_pred = self.params_head(pooled)

        return style_logits, params_pred


class AccelPredictorHybrid(nn.Module):
    """
    Hybrid model: 1D-CNN path for time-series + MLP path for hand-crafted features.

    Best of both worlds:
    - CNN+Transformer captures temporal patterns from raw data
    - MLP processes statistical features (speed distribution, flick ratio, etc.)
    - Fusion layer combines both representations

    Input:
        features: (batch, feature_dim) — hand-crafted features (100-dim)
        sequence: (batch, 3, seq_len) — raw time-series (optional)
    """

    def __init__(self, feature_dim: int = None, seq_len: int = 1024,
                 d_model: int = 256, n_heads: int = 8,
                 n_transformer_layers: int = 4, dropout: float = 0.15):
        super().__init__()

        if feature_dim is None:
            feature_dim = get_feature_dim()

        # ── Path A: CNN + Transformer (time-series) ──
        self.cnn = nn.Sequential(
            ConvBlock(3, 64, kernel=7),
            ConvBlock(64, 128, kernel=5, stride=2),
            ConvBlock(128, 256, kernel=5, stride=2),
            ConvBlock(256, d_model, kernel=3, stride=2),
        )
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len // 8 + 1, dropout=dropout)
        self.transformer = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, dropout=dropout)
              for _ in range(n_transformer_layers)]
        )

        # ── Path B: MLP (hand-crafted features) ──
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Fusion ──
        fusion_dim = d_model * 2  # concat both paths
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Dual heads ──
        self.style_head = MLPHead(d_model, 256, NUM_STYLES, dropout=dropout)
        self.params_head = MLPHead(d_model, 256, NUM_CONTINUOUS, dropout=dropout,
                                   final_activation=nn.Sigmoid())

        # ── Feature-only mode fallback (when no sequence provided) ──
        self.feature_only_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

    def forward(self, features, sequence=None):
        """
        Args:
            features: (batch, feature_dim) — always provided
            sequence: (batch, 3, seq_len) — optional raw time-series

        Returns:
            style_logits, params_pred
        """
        # Path B: hand-crafted features
        feat_repr = self.feature_encoder(features)

        if sequence is not None:
            # Path A: CNN + Transformer
            cnn_out = self.cnn(sequence)
            cnn_out = cnn_out.permute(0, 2, 1)
            cnn_out = self.pos_enc(cnn_out)
            cnn_out = self.transformer(cnn_out)
            seq_repr = cnn_out.mean(dim=1)  # global avg pool

            # Fusion
            combined = torch.cat([feat_repr, seq_repr], dim=1)
            fused = self.fusion(combined)
        else:
            # Feature-only mode
            fused = self.feature_only_proj(feat_repr)

        style_logits = self.style_head(fused)
        params_pred = self.params_head(fused)

        return style_logits, params_pred


# ─── Backward Compatible Wrapper ────────────────────────────────────────────

class AccelPredictorV2(nn.Module):
    """
    Drop-in replacement for AccelPredictor v1.

    Uses the feature-only path of AccelPredictorHybrid, making it
    backward compatible with the existing training pipeline.
    Can be upgraded to use sequences later.

    ~4M parameters (vs 1.15M in v1).
    """

    def __init__(self, feature_dim: int = None, hidden_dim: int = 512,
                 n_blocks: int = 8, n_heads: int = 8, dropout: float = 0.15):
        super().__init__()

        if feature_dim is None:
            feature_dim = get_feature_dim()

        # ── Deep feature encoder ──
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Self-attention over feature subdivisions ──
        # Reshape features into "tokens" for attention
        self.n_tokens = 10  # split 100-dim features into 10 tokens of 10-dim each
        self.token_proj = nn.Linear(feature_dim // self.n_tokens, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=self.n_tokens + 1, dropout=dropout)

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Transformer blocks
        self.attention_layers = nn.Sequential(
            *[TransformerBlock(hidden_dim, n_heads, dropout=dropout) for _ in range(4)]
        )

        # ── Deep MLP backbone ──
        self.backbone = nn.Sequential(
            *[nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout),
            ) for _ in range(n_blocks)]
        )
        # Residual connections handled manually
        self.backbone_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_blocks)])

        self.final_norm = nn.LayerNorm(hidden_dim)

        # ── Style head (classification) ──
        self.style_head = MLPHead(hidden_dim, hidden_dim // 2, NUM_STYLES, dropout=dropout)

        # ── Params head (regression) ──
        self.params_head = MLPHead(hidden_dim, hidden_dim // 2, NUM_CONTINUOUS,
                                   dropout=dropout, final_activation=nn.Sigmoid())

    def forward(self, features):
        """
        Args:
            features: (batch, feature_dim)
        Returns:
            style_logits: (batch, NUM_STYLES)
            params_pred: (batch, NUM_CONTINUOUS)
        """
        B = features.size(0)

        # ── Attention path: split features into tokens ──
        tokens = features.view(B, self.n_tokens, -1)  # (B, 10, 10)
        tokens = self.token_proj(tokens)                # (B, 10, hidden_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)        # (B, 11, hidden_dim)
        tokens = self.pos_enc(tokens)

        # Self-attention
        tokens = self.attention_layers(tokens)
        attn_repr = tokens[:, 0]  # CLS token output: (B, hidden_dim)

        # ── MLP path ──
        x = self.input_proj(features)

        # Combine attention + MLP
        x = x + attn_repr

        # Deep backbone with residual connections
        for block, norm in zip(self.backbone, self.backbone_norms):
            x = x + block(norm(x))

        x = self.final_norm(x)

        # Dual heads
        style_logits = self.style_head(x)
        params_pred = self.params_head(x)

        return style_logits, params_pred
