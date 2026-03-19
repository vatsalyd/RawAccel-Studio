"""
Fine-tune the pretrained model on real mouse data.

Strategy (self-training / pseudo-labeling):
1. Load the synthetic-pretrained model
2. Load real mouse data features (from data/convert.py)
3. Use the pretrained model to generate pseudo-labels for real data
4. Mix real (pseudo-labeled) + fresh synthetic data
5. Fine-tune with lower learning rate + regularization

This iterative process helps the model adapt to real mouse distributions
while preserving what it learned from synthetic data.

Usage:
    python -m ml.finetune
    python -m ml.finetune --epochs 20 --lr 1e-4
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from ml.model import AccelPredictor
from ml.dataset import (
    SyntheticAccelDataset, NUM_STYLES, NUM_CONTINUOUS,
    sample_random_params, generate_realistic_speeds,
    apply_accel_curve, params_to_labels,
)
from ml.features import extract_features, get_feature_dim


class RealDataDataset(Dataset):
    """
    Dataset for real mouse data with pseudo-labels.

    Uses a pretrained model to generate labels for real data,
    then trains on those labels.
    """

    def __init__(self, features: np.ndarray, model: AccelPredictor,
                 device: torch.device):
        self.features = features
        self.style_labels = []
        self.param_labels = []

        # Generate pseudo-labels using the pretrained model
        model.eval()
        with torch.no_grad():
            batch_size = 64
            for i in range(0, len(features), batch_size):
                batch = torch.from_numpy(features[i:i+batch_size]).to(device)
                style_logits, params_pred = model(batch)

                styles = style_logits.argmax(dim=1).cpu()
                params = params_pred.cpu()

                self.style_labels.append(styles)
                self.param_labels.append(params)

        self.style_labels = torch.cat(self.style_labels, dim=0)
        self.param_labels = torch.cat(self.param_labels, dim=0)

        print(f"  📊 Generated pseudo-labels for {len(features)} samples")

        # Log style distribution
        for i in range(NUM_STYLES):
            count = (self.style_labels == i).sum().item()
            if count > 0:
                from ml.dataset import STYLES_LIST
                print(f"     {STYLES_LIST[i].value}: {count}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]),
            self.style_labels[idx],
            self.param_labels[idx],
        )


class AugmentedSyntheticDataset(Dataset):
    """
    Synthetic dataset with augmentation to better match real data.

    Adds noise and jitter to make synthetic data more realistic.
    """

    def __init__(self, size: int = 5000):
        self.size = size
        self.feature_dim = get_feature_dim()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        params = sample_random_params()
        dx, dy, timestamps = generate_realistic_speeds(n_samples=2000, session_duration=30.0)

        dt = np.diff(timestamps)
        out_dx, out_dy = apply_accel_curve(dx[1:], dy[1:], dt, params)

        # ── Augmentation: add realistic noise ──
        # Sensor jitter
        out_dx += np.random.normal(0, 0.5, len(out_dx))
        out_dy += np.random.normal(0, 0.5, len(out_dy))

        # Polling rate variation (stretch/compress some intervals)
        if np.random.random() > 0.5:
            noise = np.random.normal(1.0, 0.05, len(timestamps) - 1)
            timestamps_aug = np.cumsum(np.diff(timestamps) * noise)
            timestamps_aug = np.concatenate([[0], timestamps_aug])
        else:
            timestamps_aug = timestamps

        # Random dropout (simulate missed samples)
        if np.random.random() > 0.7:
            keep = np.random.random(len(out_dx)) > 0.05
            out_dx = out_dx[keep]
            out_dy = out_dy[keep]
            timestamps_aug = timestamps_aug[1:][keep]

        features = extract_features(
            out_dx.astype(np.int32), out_dy.astype(np.int32), timestamps_aug
        )

        style_idx, continuous = params_to_labels(params)

        return (
            torch.from_numpy(features),
            torch.tensor(style_idx, dtype=torch.long),
            torch.from_numpy(continuous),
        )


def finetune(epochs: int = 20, batch_size: int = 64, lr: float = 1e-4,
             synthetic_mix: int = 5000, checkpoint: str = "checkpoints/best_model.pt"):
    """Fine-tune the pretrained model on real + augmented synthetic data."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # ── Load pretrained model ──
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        print("❌ No pretrained model found. Run: python -m ml.train")
        return

    model = AccelPredictor().to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"✅ Loaded pretrained model (val_loss={ckpt.get('val_loss', '?')})")

    # ── Load real data ──
    real_path = Path("data/processed/real_features.npz")
    if not real_path.exists():
        print("⚠️  No processed real data. Run: python -m data.convert")
        print("   Falling back to augmented synthetic data only.")
        real_features = None
    else:
        data = np.load(str(real_path))
        real_features = data["features"]
        print(f"📊 Loaded {len(real_features)} real feature vectors")

    # ── Build datasets ──
    datasets = []

    # Real data with pseudo-labels
    if real_features is not None and len(real_features) > 0:
        real_ds = RealDataDataset(real_features, model, device)
        datasets.append(real_ds)

    # Augmented synthetic data (with noise for domain adaptation)
    aug_ds = AugmentedSyntheticDataset(size=synthetic_mix)
    datasets.append(aug_ds)

    # Fresh synthetic for regularization (prevent forgetting)
    reg_ds = SyntheticAccelDataset(size=synthetic_mix // 2)
    datasets.append(reg_ds)

    combined = ConcatDataset(datasets)
    loader = DataLoader(combined, batch_size=batch_size, shuffle=True, num_workers=0)

    total = len(combined)
    print(f"📦 Training on {total} samples (real + augmented + synthetic)")

    # ── Optimizer with lower LR for fine-tuning ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    style_criterion = nn.CrossEntropyLoss()
    params_criterion = nn.MSELoss()

    # ── Training loop ──
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        n = 0
        t0 = time.time()

        for features, style_labels, param_labels in loader:
            features = features.to(device)
            style_labels = style_labels.to(device)
            param_labels = param_labels.to(device)

            style_logits, params_pred = model(features)

            loss_style = style_criterion(style_logits, style_labels)
            loss_params = params_criterion(params_pred, param_labels)
            loss = loss_style + 2.0 * loss_params

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item() * features.size(0)
            correct += (style_logits.argmax(1) == style_labels).sum().item()
            n += features.size(0)

        scheduler.step()
        avg_loss = total_loss / n
        acc = correct / n
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{epochs} │ "
            f"loss={avg_loss:.4f} acc={acc:.1%} │ {elapsed:.1f}s"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_loss,
                "finetuned": True,
            }, path)
            print(f"   💾 Saved → {path}")

    print(f"\n✅ Fine-tuning complete. Best loss={best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="🔧 Fine-tune on real data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--synthetic-mix", type=int, default=5000)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()

    finetune(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        synthetic_mix=args.synthetic_mix,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
