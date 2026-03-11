"""
Generate synthetic training data for the inverse model.

For each random accel parameter set, run the aim simulator and record
sequences of (mouse_dx, dt, view_delta). Store sequences + ground-truth
params as compressed .npz files with train/val split.

Usage:
    python -m experiments.inverse_model.gen_synthetic_data
    python -m experiments.inverse_model.gen_synthetic_data --num-curves 5000
"""
import argparse
import os
from typing import List

import numpy as np
import torch

from env.aim_sim.env_core import SimpleAimTask, HumanLikeControllerConfig
from models.curve_param_config import (
    AccelParams,
    apply_accel,
    sample_random_params,
    TUNE_KEYS,
)


def simulate_sequence(
    params: AccelParams,
    steps: int = 60,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Run one aim task and return (T, 3) array: [mouse_dx, dt, view_delta]."""
    rng = rng or np.random.default_rng()
    task = SimpleAimTask(rng=rng)
    ctrl = HumanLikeControllerConfig()
    dt = 0.01

    seq: List[List[float]] = []
    for _ in range(steps):
        error = task.target_yaw - task.view_yaw
        controller_output = (
            ctrl.base_gain * error
            + rng.standard_normal() * ctrl.noise_std * abs(error)
        )
        mouse_dx_counts = controller_output * 100.0

        mouse_dx_t = torch.tensor(mouse_dx_counts, dtype=torch.float32)
        dt_t = torch.tensor(dt, dtype=torch.float32)
        view_delta = apply_accel(mouse_dx_t, dt_t, params)
        task.view_yaw += float(view_delta.item())

        seq.append([mouse_dx_counts, dt, float(view_delta.item())])

        done, _ = task.step_with_params(params, ctrl, dt=dt)
        if done:
            break

    return np.array(seq, dtype=np.float32)


def main(config_path: str = None, num_curves: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    out_dir = os.path.join("data", "sim_inverse")
    os.makedirs(out_dir, exist_ok=True)

    sequences: List[np.ndarray] = []
    params_list: List[List[float]] = []

    for i in range(num_curves):
        p = sample_random_params(rng)
        seq = simulate_sequence(p, rng=rng)
        sequences.append(seq)
        d = p.as_dict()
        params_list.append([d[k] for k in TUNE_KEYS])

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_curves} curves")

    params_arr = np.array(params_list, dtype=np.float32)

    # Train / val split (80 / 20)
    n = len(sequences)
    indices = rng.permutation(n)
    split = int(n * 0.8)
    train_idx, val_idx = indices[:split], indices[split:]

    train_seqs = [sequences[i] for i in train_idx]
    val_seqs = [sequences[i] for i in val_idx]
    train_params = params_arr[train_idx]
    val_params = params_arr[val_idx]

    np.savez_compressed(
        os.path.join(out_dir, "train.npz"),
        sequences=np.array(train_seqs, dtype=object),
        params=train_params,
    )
    np.savez_compressed(
        os.path.join(out_dir, "val.npz"),
        sequences=np.array(val_seqs, dtype=object),
        params=val_params,
    )
    print(f"Saved {len(train_seqs)} train / {len(val_seqs)} val samples to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic inverse-model data")
    parser.add_argument("--num-curves", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(num_curves=args.num_curves, seed=args.seed)