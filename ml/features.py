"""
Feature extraction from raw mouse movement data.

Takes timestamped (dx, dy) sequences and computes features
that characterize a player's mouse behavior + the acceleration
curve they're likely using.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def compute_speeds(dx: np.ndarray, dy: np.ndarray,
                   dt: np.ndarray) -> np.ndarray:
    """Compute mouse speed in counts/ms from deltas and time intervals."""
    dist = np.sqrt(dx.astype(float) ** 2 + dy.astype(float) ** 2)
    dt_safe = np.clip(dt, a_min=1e-6, a_max=None)
    return dist / dt_safe


def extract_features(dx: np.ndarray, dy: np.ndarray,
                     timestamps: np.ndarray,
                     n_speed_bins: int = 50,
                     n_accel_bins: int = 30) -> np.ndarray:
    """
    Extract a fixed-length feature vector from a mouse recording session.

    Args:
        dx: x-deltas (counts), shape (N,)
        dy: y-deltas (counts), shape (N,)
        timestamps: time of each sample (seconds), shape (N,)
        n_speed_bins: number of bins for speed histogram
        n_accel_bins: number of bins for acceleration histogram

    Returns:
        Feature vector, shape (n_features,)
    """
    N = len(dx)
    if N < 10:
        # Too few samples — return zeros
        total_features = n_speed_bins + n_accel_bins + 20
        return np.zeros(total_features, dtype=np.float32)

    # ── Time deltas ──
    dt = np.diff(timestamps)
    dt = np.clip(dt, 1e-6, 1.0)  # clamp outliers

    dx_f = dx[1:].astype(np.float64)
    dy_f = dy[1:].astype(np.float64)

    # ── Speed (counts/ms) ──
    dist = np.sqrt(dx_f ** 2 + dy_f ** 2)
    speed = dist / dt

    # ── Acceleration (change in speed) ──
    accel = np.diff(speed) / np.clip(dt[1:], 1e-6, None)

    # ── 1. Speed histogram (normalized) ──
    max_speed = np.percentile(speed, 99) if len(speed) > 0 else 1.0
    speed_hist, _ = np.histogram(
        speed, bins=n_speed_bins, range=(0, max(max_speed, 0.1))
    )
    speed_hist = speed_hist.astype(np.float32)
    speed_hist /= (speed_hist.sum() + 1e-8)

    # ── 2. Acceleration histogram (normalized) ──
    if len(accel) > 0:
        accel_abs = np.abs(accel)
        max_accel = np.percentile(accel_abs, 99)
        accel_hist, _ = np.histogram(
            accel_abs, bins=n_accel_bins, range=(0, max(max_accel, 0.1))
        )
        accel_hist = accel_hist.astype(np.float32)
        accel_hist /= (accel_hist.sum() + 1e-8)
    else:
        accel_hist = np.zeros(n_accel_bins, dtype=np.float32)

    # ── 3. Summary statistics (20 features) ──
    stats = np.zeros(20, dtype=np.float32)

    # Speed stats
    stats[0] = np.mean(speed)
    stats[1] = np.std(speed)
    stats[2] = np.median(speed)
    stats[3] = np.percentile(speed, 10)
    stats[4] = np.percentile(speed, 25)
    stats[5] = np.percentile(speed, 75)
    stats[6] = np.percentile(speed, 90)
    stats[7] = np.percentile(speed, 99)

    # Acceleration stats
    if len(accel) > 0:
        stats[8] = np.mean(np.abs(accel))
        stats[9] = np.std(accel)
        stats[10] = np.percentile(np.abs(accel), 90)

    # Direction change rate (flick detection)
    if len(dx_f) > 1:
        angles = np.arctan2(dy_f, dx_f + 1e-8)
        angle_diff = np.abs(np.diff(angles))
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        stats[11] = np.mean(angle_diff)                   # avg direction change
        stats[12] = np.sum(angle_diff > np.pi / 2) / len(angle_diff)  # sharp turn rate

    # Flick vs tracking ratio
    flick_threshold = np.percentile(speed, 85) if len(speed) > 10 else 10.0
    flick_mask = speed > flick_threshold
    stats[13] = np.sum(flick_mask) / len(speed)           # flick fraction
    stats[14] = np.mean(speed[flick_mask]) if flick_mask.any() else 0  # avg flick speed
    stats[15] = np.mean(speed[~flick_mask]) if (~flick_mask).any() else 0  # avg track speed

    # Micro-correction rate (very small movements after large ones)
    if len(speed) > 2:
        slow_after_fast = (speed[1:] < np.percentile(speed, 20)) & \
                          (speed[:-1] > np.percentile(speed, 80))
        stats[16] = np.sum(slow_after_fast) / len(speed)

    # Movement duration stats
    stats[17] = len(speed) / (timestamps[-1] - timestamps[0] + 1e-6)  # sample rate
    stats[18] = timestamps[-1] - timestamps[0]             # total duration

    # Click rate (if clicks encoded as zero-distance events)
    zero_dist = dist < 0.5
    stats[19] = np.sum(zero_dist) / (stats[18] + 1e-6)    # clicks per second

    # ── Concatenate all features ──
    features = np.concatenate([speed_hist, accel_hist, stats])
    return features


def get_feature_dim(n_speed_bins: int = 50, n_accel_bins: int = 30) -> int:
    """Return the total feature vector dimensionality."""
    return n_speed_bins + n_accel_bins + 20


def load_session_arrays(session_data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dx, dy, timestamps from a session JSON dict.

    Handles both desktop logger format (samples with t, dx, dy)
    and browser format (events with time, dx, dy).
    """
    if "samples" in session_data:
        samples = session_data["samples"]
        dx = np.array([s["dx"] for s in samples if s.get("event_type", "move") == "move"])
        dy = np.array([s["dy"] for s in samples if s.get("event_type", "move") == "move"])
        t = np.array([s["t"] for s in samples if s.get("event_type", "move") == "move"])
    elif "events" in session_data:
        events = [e for e in session_data["events"] if e.get("type") == "move"]
        dx = np.array([e.get("dx", 0) for e in events])
        dy = np.array([e.get("dy", 0) for e in events])
        t = np.array([e.get("time", 0) for e in events]) / 1000.0  # ms → seconds
    else:
        raise ValueError("Unknown session format")

    return dx, dy, t
