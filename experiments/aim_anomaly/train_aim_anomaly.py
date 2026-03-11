"""
P4: Anomaly / assisted-aim detection.

Features:
- Multiple assisted-aim patterns (snap, smooth aimbot, trigger bot)
- Train / val split
- ROC-AUC, precision, recall, F1 metrics
- TensorBoard logging

Usage:
    python -m experiments.aim_anomaly.train_aim_anomaly
"""
import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from env.aim_sim.env_core import SimpleAimTask, HumanLikeControllerConfig
from models.curve_param_config import DEFAULT_ACCEL_PARAMS, apply_accel
from models.aim_anomaly_net import AimAnomalyNet

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


# ---------------------------------------------------------------------------
# Sequence simulation helpers
# ---------------------------------------------------------------------------

def _run_controller(
    gain: float,
    noise_std: float,
    steps: int,
    rng: np.random.Generator,
    snap_delay: int = 0,
) -> np.ndarray:
    """Generic controller runner. Returns (T, 4) array."""
    task = SimpleAimTask(rng=rng)
    dt = 0.01
    seq = []
    for i in range(steps):
        error = task.target_yaw - task.view_yaw
        if i < snap_delay:
            mouse_dx_counts = 0.0
        else:
            mouse_dx_counts = gain * error + rng.standard_normal() * noise_std * abs(error)
            mouse_dx_counts *= 100.0

        mouse_dx = torch.tensor(mouse_dx_counts, dtype=torch.float32)
        dt_t = torch.tensor(dt, dtype=torch.float32)
        view_delta = apply_accel(mouse_dx, dt_t, DEFAULT_ACCEL_PARAMS)
        task.view_yaw += float(view_delta.item())
        seq.append([mouse_dx_counts, dt, float(view_delta.item()), float(error)])

        done, _ = task.step_with_params(DEFAULT_ACCEL_PARAMS, HumanLikeControllerConfig(), dt=dt)
        if done:
            break
    return np.array(seq, dtype=np.float32)


def simulate_normal(steps: int = 60, rng: np.random.Generator = None) -> np.ndarray:
    """Human-like: moderate gain, realistic noise."""
    rng = rng or np.random.default_rng()
    return _run_controller(gain=0.8, noise_std=0.05, steps=steps, rng=rng)


def simulate_snap(steps: int = 60, rng: np.random.Generator = None) -> np.ndarray:
    """Snap aimbot: waits briefly then snaps with extremely high gain, no noise."""
    rng = rng or np.random.default_rng()
    return _run_controller(gain=5.0, noise_std=0.001, steps=steps, rng=rng, snap_delay=3)


def simulate_smooth(steps: int = 60, rng: np.random.Generator = None) -> np.ndarray:
    """Smooth aimbot: high gain, very low noise — unnaturally precise tracking."""
    rng = rng or np.random.default_rng()
    return _run_controller(gain=2.0, noise_std=0.005, steps=steps, rng=rng)


def simulate_trigger(steps: int = 60, rng: np.random.Generator = None) -> np.ndarray:
    """Trigger bot: normal-ish aim but perfect correction when close to target."""
    rng = rng or np.random.default_rng()
    task = SimpleAimTask(rng=rng)
    ctrl = HumanLikeControllerConfig()
    dt = 0.01
    seq = []
    for _ in range(steps):
        error = task.target_yaw - task.view_yaw
        if abs(error) < 10.0:
            # Perfect correction
            mouse_dx_counts = error * 300.0
        else:
            # Normal-ish aim
            mouse_dx_counts = (ctrl.base_gain * error +
                               rng.standard_normal() * ctrl.noise_std * abs(error)) * 100.0
        mouse_dx = torch.tensor(mouse_dx_counts, dtype=torch.float32)
        dt_t = torch.tensor(dt, dtype=torch.float32)
        view_delta = apply_accel(mouse_dx, dt_t, DEFAULT_ACCEL_PARAMS)
        task.view_yaw += float(view_delta.item())
        seq.append([mouse_dx_counts, dt, float(view_delta.item()), float(error)])
        done, _ = task.step_with_params(DEFAULT_ACCEL_PARAMS, ctrl, dt=dt)
        if done:
            break
    return np.array(seq, dtype=np.float32)


ASSISTED_GENERATORS = {
    "snap": simulate_snap,
    "smooth": simulate_smooth,
    "trigger": simulate_trigger,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AnomalyDataset(Dataset):
    def __init__(self, n_per_class: int = 500, max_len: int = 80, seed: int = 99):
        rng = np.random.default_rng(seed)

        normal_seqs = [simulate_normal(rng=rng) for _ in range(n_per_class)]
        assisted_seqs = []
        gens = list(ASSISTED_GENERATORS.values())
        for i in range(n_per_class):
            gen_fn = gens[i % len(gens)]
            assisted_seqs.append(gen_fn(rng=rng))

        self.seqs = normal_seqs + assisted_seqs
        self.labels = [0] * n_per_class + [1] * n_per_class
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        T = len(seq)
        if T < self.max_len:
            pad = np.zeros((self.max_len - T, seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
        else:
            seq = seq[:self.max_len]
        return torch.from_numpy(seq), torch.tensor(self.labels[idx], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(config_path: str = None):
    n_per_class = 500
    epochs = 25
    batch_size = 32
    lr = 1e-3
    save_dir = "runs/aim_anomaly"
    tb_dir = os.path.join(save_dir, "tb")
    os.makedirs(save_dir, exist_ok=True)

    dataset = AnomalyDataset(n_per_class=n_per_class)
    n = len(dataset)
    n_val = int(n * 0.2)
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n - n_val, n_val], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AimAnomalyNet(input_dim=4, use_cnn=True, dropout=0.2).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(tb_dir) if SummaryWriter else None

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        all_probs, all_labels, all_preds = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += loss_fn(logits, y).item() * x.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(float)
                all_probs.extend(probs.tolist())
                all_labels.extend(y.cpu().numpy().tolist())
                all_preds.extend(preds.tolist())
        val_loss /= len(val_ds)

        # Metrics
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)
        all_probs_np = np.array(all_probs)

        accuracy = float(np.mean(all_preds_np == all_labels_np))
        try:
            auc = roc_auc_score(all_labels_np, all_probs_np)
        except ValueError:
            auc = 0.0
        precision = precision_score(all_labels_np, all_preds_np, zero_division=0)
        recall = recall_score(all_labels_np, all_preds_np, zero_division=0)
        f1 = f1_score(all_labels_np, all_preds_np, zero_division=0)

        print(
            f"Epoch {epoch:3d}: loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"acc={accuracy:.3f}  AUC={auc:.3f}  P={precision:.3f}  "
            f"R={recall:.3f}  F1={f1:.3f}"
        )

        if writer:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("metrics/accuracy", accuracy, epoch)
            writer.add_scalar("metrics/roc_auc", auc, epoch)
            writer.add_scalar("metrics/precision", precision, epoch)
            writer.add_scalar("metrics/recall", recall, epoch)
            writer.add_scalar("metrics/f1", f1, epoch)

    torch.save(model.state_dict(), os.path.join(save_dir, "aim_anomaly_net.pt"))
    print(f"Model saved to {save_dir}")
    if writer:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P4: Anomaly / assisted-aim detection")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(config_path=args.config)