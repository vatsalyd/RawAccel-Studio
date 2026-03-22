"""
Training loop for the accel curve predictor.

Usage:
    python -m ml.train                    # v2 model (default, ~4M params)
    python -m ml.train --model v1         # v1 model (1.15M params)
    python -m ml.train --epochs 50 --batch-size 64
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml.model import AccelPredictor
from ml.model_v2 import AccelPredictorV2
from ml.dataset import SyntheticAccelDataset, NUM_STYLES, NUM_CONTINUOUS


def train(epochs: int = 30, batch_size: int = 64, lr: float = 1e-3,
          train_size: int = 20000, val_size: int = 2000,
          save_dir: str = "checkpoints", model_version: str = "v2"):
    """Train the accel predictor on synthetic data."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # ── Data ──
    print("📊 Creating synthetic datasets...")
    train_ds = SyntheticAccelDataset(size=train_size)
    val_ds = SyntheticAccelDataset(size=val_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # ── Model ──
    if model_version == "v2":
        model = AccelPredictorV2().to(device)
        print("🚀 Using AccelPredictorV2 (CNN+Attention+Deep MLP)")
    else:
        model = AccelPredictor().to(device)
        print("📦 Using AccelPredictor v1 (basic MLP)")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model: {n_params:,} parameters")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Loss ──
    style_criterion = nn.CrossEntropyLoss()
    params_criterion = nn.MSELoss()

    # ── Training ──
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_style_correct = 0
        train_total = 0
        t0 = time.time()

        for features, style_labels, param_labels in train_loader:
            features = features.to(device)
            style_labels = style_labels.to(device)
            param_labels = param_labels.to(device)

            style_logits, params_pred = model(features)

            loss_style = style_criterion(style_logits, style_labels)
            loss_params = params_criterion(params_pred, param_labels)
            loss = loss_style + 2.0 * loss_params  # weight params more

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_style_correct += (style_logits.argmax(1) == style_labels).sum().item()
            train_total += features.size(0)

        scheduler.step()

        train_loss /= train_total
        train_acc = train_style_correct / train_total

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        val_style_correct = 0
        val_total = 0
        val_param_mae = 0.0

        with torch.no_grad():
            for features, style_labels, param_labels in val_loader:
                features = features.to(device)
                style_labels = style_labels.to(device)
                param_labels = param_labels.to(device)

                style_logits, params_pred = model(features)

                loss_style = style_criterion(style_logits, style_labels)
                loss_params = params_criterion(params_pred, param_labels)
                loss = loss_style + 2.0 * loss_params

                val_loss += loss.item() * features.size(0)
                val_style_correct += (style_logits.argmax(1) == style_labels).sum().item()
                val_total += features.size(0)
                val_param_mae += F.l1_loss(params_pred, param_labels, reduction="sum").item()

        val_loss /= val_total
        val_acc = val_style_correct / val_total
        val_param_mae /= (val_total * NUM_CONTINUOUS)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs} │ "
            f"train_loss={train_loss:.4f} style_acc={train_acc:.1%} │ "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.1%} "
            f"param_mae={val_param_mae:.4f} │ {elapsed:.1f}s"
        )

        # ── Save best ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "model_version": model_version,
            }, path)
            print(f"   💾 Saved best model → {path}")

    print(f"\n✅ Training complete. Best val_loss={best_val_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="🧠 Train RawAccel curve predictor")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--model", type=str, default="v2", choices=["v1", "v2"])
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_size=args.train_size,
        val_size=args.val_size,
        model_version=args.model,
    )


if __name__ == "__main__":
    main()
