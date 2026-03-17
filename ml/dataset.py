"""
Synthetic dataset generation for training the accel curve predictor.

Strategy:
1. Pick a random RawAccel curve type + parameters
2. Generate realistic mouse speed sequences
3. Apply the curve to get (input_speed → output_sensitivity) pairs
4. Extract features from the simulated movement
5. Label = the curve parameters that generated it

This gives us unlimited supervised training data with perfect labels.
"""
from __future__ import annotations

import random
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from rawaccel.curves import (
    AccelStyle, CurveParams, STYLE_PARAM_BOUNDS,
    evaluate, evaluate_curve,
)
from ml.features import extract_features, get_feature_dim


# ─── Realistic mouse speed distributions ────────────────────────────────────

def generate_realistic_speeds(n_samples: int = 2000,
                              session_duration: float = 30.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate realistic-looking mouse movement data.

    Simulates a mix of:
    - Idle periods (very low speed)
    - Tracking movements (medium sustained speed)
    - Flick movements (high speed bursts)
    - Micro-adjustments (tiny corrections)

    Returns:
        dx, dy, timestamps arrays
    """
    timestamps = np.linspace(0, session_duration, n_samples)
    dt_avg = session_duration / n_samples

    dx = np.zeros(n_samples)
    dy = np.zeros(n_samples)

    i = 0
    while i < n_samples:
        # Pick a behavior segment
        behavior = random.choices(
            ["idle", "tracking", "flick", "micro"],
            weights=[0.2, 0.35, 0.25, 0.2],
        )[0]

        if behavior == "idle":
            seg_len = random.randint(20, 100)
            # Near-zero movement with tiny jitter
            end = min(i + seg_len, n_samples)
            dx[i:end] = np.random.normal(0, 0.3, end - i)
            dy[i:end] = np.random.normal(0, 0.3, end - i)

        elif behavior == "tracking":
            seg_len = random.randint(50, 200)
            end = min(i + seg_len, n_samples)
            # Sustained movement in one direction
            angle = random.uniform(0, 2 * np.pi)
            base_speed = random.uniform(3, 15)  # counts/ms
            speed_var = np.random.normal(base_speed, base_speed * 0.2, end - i)
            speed_var = np.clip(speed_var, 0.5, 30)
            dx[i:end] = speed_var * np.cos(angle) * dt_avg
            dy[i:end] = speed_var * np.sin(angle) * dt_avg

        elif behavior == "flick":
            seg_len = random.randint(5, 20)
            end = min(i + seg_len, n_samples)
            # High-speed burst
            angle = random.uniform(0, 2 * np.pi)
            peak_speed = random.uniform(20, 60)
            # Ramp up then down
            t_seg = np.linspace(0, np.pi, end - i)
            speed_profile = peak_speed * np.sin(t_seg) * dt_avg
            dx[i:end] = speed_profile * np.cos(angle)
            dy[i:end] = speed_profile * np.sin(angle)

        elif behavior == "micro":
            seg_len = random.randint(10, 40)
            end = min(i + seg_len, n_samples)
            # Tiny precise corrections
            dx[i:end] = np.random.normal(0, 1.5, end - i) * dt_avg
            dy[i:end] = np.random.normal(0, 1.5, end - i) * dt_avg

        i = end if 'end' in dir() else i + 1

    return dx.astype(np.float64), dy.astype(np.float64), timestamps


def apply_accel_curve(dx: np.ndarray, dy: np.ndarray,
                      dt: np.ndarray, params: CurveParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a RawAccel curve to mouse deltas.

    For each sample, compute the input speed, evaluate the curve
    to get the sensitivity multiplier, and scale the output.

    Returns:
        (output_dx, output_dy) — the accelerated mouse output
    """
    dist = np.sqrt(dx ** 2 + dy ** 2)
    speed = dist / np.clip(dt, 1e-6, None)  # counts/ms

    # Evaluate sensitivity at each speed
    sens = np.array([evaluate(s, params) for s in speed])

    # Apply sensitivity to output
    out_dx = dx * sens
    out_dy = dy * sens

    return out_dx, out_dy


# ─── Random parameter sampling ─────────────────────────────────────────────

def sample_random_params(style: Optional[AccelStyle] = None) -> CurveParams:
    """
    Sample a random set of RawAccel parameters.

    If style is None, picks a random style.
    """
    if style is None:
        style = random.choice(list(AccelStyle))

    params = CurveParams(style=style)

    # Sample style-specific parameters within bounds
    bounds = STYLE_PARAM_BOUNDS[style]
    for param_name, (lo, hi) in bounds.items():
        val = random.uniform(lo, hi)
        setattr(params, param_name, val)

    # Global parameters
    params.sens_multiplier = random.uniform(0.5, 2.0)
    params.yx_ratio = random.choice([1.0, 1.0, 1.0, random.uniform(0.8, 1.2)])

    return params


# ─── Label encoding ────────────────────────────────────────────────────────

STYLES_LIST = list(AccelStyle)
NUM_STYLES = len(STYLES_LIST)

# Continuous parameters we predict (in this order)
CONTINUOUS_PARAMS = [
    "acceleration", "exponent", "scale", "motivity", "gamma",
    "smooth", "decay_rate", "sync_speed", "offset",
    "cap_output", "output_offset", "sens_multiplier",
]
NUM_CONTINUOUS = len(CONTINUOUS_PARAMS)

# Normalization ranges for continuous parameters
PARAM_RANGES = {
    "acceleration": (0.0, 2.0),
    "exponent": (0.0, 4.0),
    "scale": (0.0, 5.0),
    "motivity": (1.0, 4.0),
    "gamma": (0.0, 3.0),
    "smooth": (0.0, 1.0),
    "decay_rate": (0.0, 1.0),
    "sync_speed": (0.0, 20.0),
    "offset": (0.0, 20.0),
    "cap_output": (0.0, 5.0),
    "output_offset": (0.0, 1.5),
    "sens_multiplier": (0.0, 3.0),
}


def params_to_labels(params: CurveParams) -> Tuple[int, np.ndarray]:
    """
    Convert CurveParams to training labels.

    Returns:
        (style_index, normalized_continuous_params)
    """
    style_idx = STYLES_LIST.index(params.style)

    continuous = np.zeros(NUM_CONTINUOUS, dtype=np.float32)
    for i, name in enumerate(CONTINUOUS_PARAMS):
        val = getattr(params, name)
        lo, hi = PARAM_RANGES[name]
        # Normalize to [0, 1]
        continuous[i] = (val - lo) / (hi - lo + 1e-8)
        continuous[i] = np.clip(continuous[i], 0.0, 1.0)

    return style_idx, continuous


def labels_to_params(style_idx: int, continuous: np.ndarray) -> CurveParams:
    """Inverse of params_to_labels — reconstruct CurveParams from labels."""
    style = STYLES_LIST[style_idx]
    params = CurveParams(style=style)

    for i, name in enumerate(CONTINUOUS_PARAMS):
        lo, hi = PARAM_RANGES[name]
        val = continuous[i] * (hi - lo) + lo
        setattr(params, name, float(val))

    return params


# ─── PyTorch Dataset ────────────────────────────────────────────────────────

class SyntheticAccelDataset(Dataset):
    """
    On-the-fly synthetic dataset for training.

    Each __getitem__ call:
    1. Samples random RawAccel params
    2. Generates realistic mouse data
    3. Applies the accel curve
    4. Extracts features
    5. Returns (features, style_label, continuous_labels)
    """

    def __init__(self, size: int = 10000, session_samples: int = 2000,
                 session_duration: float = 30.0):
        self.size = size
        self.session_samples = session_samples
        self.session_duration = session_duration
        self.feature_dim = get_feature_dim()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        # 1. Random params
        params = sample_random_params()

        # 2. Generate raw mouse data
        dx, dy, timestamps = generate_realistic_speeds(
            n_samples=self.session_samples,
            session_duration=self.session_duration,
        )

        # 3. Apply accel curve
        dt = np.diff(timestamps)
        out_dx, out_dy = apply_accel_curve(dx[1:], dy[1:], dt, params)

        # 4. Extract features from the *output* (what the user actually sees)
        # Use timestamps[1:] to align with the diff'd delta arrays
        features = extract_features(
            out_dx.astype(np.int32), out_dy.astype(np.int32), timestamps[1:]
        )

        # 5. Labels
        style_idx, continuous = params_to_labels(params)

        return (
            torch.from_numpy(features),
            torch.tensor(style_idx, dtype=torch.long),
            torch.from_numpy(continuous),
        )
