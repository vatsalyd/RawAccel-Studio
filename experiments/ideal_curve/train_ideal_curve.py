"""
P3: Personalized ideal-curve learning from synthetic player profiles.

Features:
- Realistic correlated synthetic profiles
- Simulator-in-the-loop evaluation (predict params → run in AimEnv)
- Validation loop with early stopping
- TensorBoard logging

Usage:
    python -m experiments.ideal_curve.train_ideal_curve
"""
import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split

from models.ideal_curve_net import IdealCurveNet
from models.curve_param_config import (
    AccelParams,
    DEFAULT_ACCEL_PARAMS,
    TUNE_KEYS,
    PARAM_BOUNDS,
    tensor_to_params,
    clamp_params,
)
from env.aim_sim.env_core import SimpleAimTask, HumanLikeControllerConfig

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


# ---------------------------------------------------------------------------
# Correlated synthetic profile generator
# ---------------------------------------------------------------------------

class SyntheticProfileDataset(Dataset):
    """
    Generate synthetic player profiles with correlated features.

    Features (8):
        avg_accuracy, avg_reaction_time, tracking_score, flick_score,
        hours_played, preferred_sens, preferred_dpi_norm, mouse_weight_norm
    """

    def __init__(self, n: int = 5000, input_dim: int = 8, num_params: int = 5, seed: int = 123):
        rng = np.random.default_rng(seed)

        # Generate correlated features
        accuracy = rng.beta(5, 2, size=n)                        # [0, 1]
        reaction_time = 0.1 + 0.4 * rng.beta(2, 5, size=n)      # [0.1, 0.5]s
        tracking = rng.beta(4, 3, size=n) * accuracy             # correlated with accuracy
        flicking = rng.beta(3, 4, size=n) * (1 - reaction_time)  # fast reactions → better flicks
        hours = rng.lognormal(6, 1.5, size=n)
        hours = np.clip(hours / hours.max(), 0, 1)               # normalise
        pref_sens = rng.gamma(2, 0.3, size=n)
        pref_sens = np.clip(pref_sens / pref_sens.max(), 0, 1)
        pref_dpi = rng.uniform(0.2, 1.0, size=n)
        mouse_weight = rng.uniform(0.3, 1.0, size=n)

        self.features = np.stack(
            [accuracy, reaction_time, tracking, flicking,
             hours, pref_sens, pref_dpi, mouse_weight],
            axis=-1,
        ).astype(np.float32)

        # Ground-truth params: non-linear mapping from features
        # Idea: good players (high accuracy, fast reactions) → lower gain, sharper breakpoint
        k1 = 0.001 + 0.005 * (1 - accuracy) + 0.002 * rng.standard_normal(n)
        a = 0.8 + 0.4 * flicking + 0.1 * rng.standard_normal(n)
        k2 = 0.0005 + 0.003 * pref_sens + 0.001 * rng.standard_normal(n)
        b = 1.0 + 0.3 * tracking + 0.1 * rng.standard_normal(n)
        v0 = 200 + 800 * (1 - reaction_time) + 100 * rng.standard_normal(n)

        params = np.stack([k1, a, k2, b, v0], axis=-1).astype(np.float32)

        # Clamp to valid ranges
        for i, key in enumerate(TUNE_KEYS):
            lo, hi = PARAM_BOUNDS[key]
            params[:, i] = np.clip(params[:, i], lo, hi)

        self.params = params

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.params[idx])


# ---------------------------------------------------------------------------
# Simulator-in-the-loop evaluation
# ---------------------------------------------------------------------------

def sim_eval_params(params: AccelParams, n_trials: int = 20, seed: int = 0) -> dict:
    """Run aim tasks and return metrics for the given params."""
    rng = np.random.default_rng(seed)
    ctrl = HumanLikeControllerConfig()
    hits, times, errors = [], [], []
    for _ in range(n_trials):
        task = SimpleAimTask(rng=rng)
        done = False
        while not done:
            done, m = task.step_with_params(params, ctrl)
        hits.append(m["hit"])
        times.append(m["time"])
        errors.append(m["error"])
    return {
        "hit_rate": float(np.mean(hits)),
        "avg_time": float(np.mean(times)),
        "avg_error": float(np.mean(errors)),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(config_path: str = None):
    epochs = 40
    batch_size = 64
    lr = 1e-3
    lr_patience = 5
    save_dir = "runs/ideal_curve"
    tb_dir = os.path.join(save_dir, "tb")
    os.makedirs(save_dir, exist_ok=True)

    dataset = SyntheticProfileDataset()
    n = len(dataset)
    n_val = int(n * 0.2)
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n - n_val, n_val], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IdealCurveNet(input_dim=8, num_params=5, use_batch_norm=True, dropout=0.1).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optim, patience=lr_patience, factor=0.5)
    loss_fn = nn.MSELoss()

    writer = SummaryWriter(tb_dir) if SummaryWriter else None
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += loss_fn(model(x), y).item() * x.size(0)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d}: train={train_loss:.5f}  val={val_loss:.5f}")

        if writer:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "ideal_curve_net_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch}")
                break

    torch.save(model.state_dict(), os.path.join(save_dir, "ideal_curve_net.pt"))

    # Simulator-in-the-loop eval on a few samples
    print("\n--- Simulator-in-the-loop evaluation ---")
    model.eval()
    rng_eval = np.random.default_rng(99)
    with torch.no_grad():
        for i in range(5):
            x, y_true = dataset[i]
            pred = model(x.unsqueeze(0).to(device)).cpu().squeeze(0)
            p_pred = tensor_to_params(pred, keys=TUNE_KEYS)
            p_true = tensor_to_params(y_true, keys=TUNE_KEYS)
            m_pred = sim_eval_params(p_pred, seed=i)
            m_true = sim_eval_params(p_true, seed=i)
            print(f"  Sample {i}: predicted hit={m_pred['hit_rate']:.2f}  "
                  f"true hit={m_true['hit_rate']:.2f}  "
                  f"pred_time={m_pred['avg_time']:.3f}  "
                  f"true_time={m_true['avg_time']:.3f}")

    if writer:
        writer.close()
    print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P3: Ideal curve learning")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(config_path=args.config)