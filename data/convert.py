"""
Convert external FPS datasets into our internal training format.

Takes raw downloaded data and produces feature vectors + pseudo-labels
for fine-tuning the pretrained model.

Since external data doesn't come with RawAccel curve labels, we use
the pretrained model to generate pseudo-labels (self-training approach):
  1. Extract features from real mouse data
  2. Use the synthetic-trained model to predict initial labels
  3. Fine-tune the model on these + augmented versions

Usage:
    python -m data.convert
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ml.features import extract_features, get_feature_dim


EXTERNAL_DIR = Path("data/external")
PROCESSED_DIR = Path("data/processed")


def convert_desktop_sessions(raw_dir: str = "data/raw") -> List[np.ndarray]:
    """Convert sessions recorded with collector/logger.py."""
    features_list = []
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        return features_list

    for f in raw_path.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            samples = data.get("samples", [])
            moves = [s for s in samples if s.get("event_type", "move") == "move"]

            if len(moves) < 50:
                continue

            dx = np.array([s["dx"] for s in moves], dtype=np.float64)
            dy = np.array([s["dy"] for s in moves], dtype=np.float64)
            t = np.array([s["t"] for s in moves], dtype=np.float64)

            feat = extract_features(dx, dy, t)
            features_list.append(feat)
        except Exception as e:
            print(f"  ⚠️  Skipping {f.name}: {e}")

    print(f"  📁 Desktop sessions: {len(features_list)} valid files")
    return features_list


def convert_csgo_data() -> List[np.ndarray]:
    """
    Convert CS:GO behavioral cloning data.

    The HuggingFace dataset has .npy files with gameplay actions
    including mouse dx/dy at each frame.
    """
    features_list = []
    csgo_dir = EXTERNAL_DIR / "csgo"

    if not csgo_dir.exists():
        return features_list

    # Look for .npy files in chunk directories
    npy_files = list(csgo_dir.rglob("*.npy"))
    if not npy_files:
        # Try .hdf5 files
        try:
            import h5py
            hdf5_files = list(csgo_dir.rglob("*.hdf5"))
            for hf in hdf5_files[:100]:  # limit
                try:
                    with h5py.File(str(hf), 'r') as f:
                        if 'mouse_x' in f and 'mouse_y' in f:
                            dx = np.array(f['mouse_x'], dtype=np.float64)
                            dy = np.array(f['mouse_y'], dtype=np.float64)
                            t = np.linspace(0, len(dx) / 64.0, len(dx))  # ~64 tick
                            feat = extract_features(dx, dy, t)
                            features_list.append(feat)
                except Exception:
                    continue
        except ImportError:
            print("  ⚠️  Install h5py for HDF5 support: pip install h5py")

    for nf in npy_files[:200]:  # limit to 200 files
        try:
            data = np.load(str(nf), allow_pickle=True)
            # Extract mouse deltas from the action data
            if isinstance(data, np.ndarray) and data.ndim >= 2:
                # Typical format: columns include mouse_x, mouse_y
                if data.shape[1] >= 2:
                    dx = data[:, 0].astype(np.float64)
                    dy = data[:, 1].astype(np.float64)
                    t = np.linspace(0, len(dx) / 64.0, len(dx))
                    if len(dx) >= 50:
                        feat = extract_features(dx, dy, t)
                        features_list.append(feat)
        except Exception:
            continue

    print(f"  📁 CS:GO sessions: {len(features_list)} valid files")
    return features_list


def convert_redeclipse_data() -> List[np.ndarray]:
    """
    Convert Red Eclipse FPS mouse data.

    Each game file is JSON with a list of events including mouse movements.
    """
    features_list = []
    re_dir = EXTERNAL_DIR / "redeclipse"

    if not re_dir.exists():
        return features_list

    json_files = list(re_dir.rglob("*.json"))
    for jf in json_files[:200]:
        try:
            data = json.loads(jf.read_text())

            # Red Eclipse format: events list with type, time, and data
            events = data if isinstance(data, list) else data.get("events", [])
            mouse_events = [e for e in events if isinstance(e, dict)
                           and e.get("type") in ("mouse", "mousemove", "move")]

            if len(mouse_events) < 50:
                continue

            dx = np.array([e.get("dx", e.get("x", 0)) for e in mouse_events], dtype=np.float64)
            dy = np.array([e.get("dy", e.get("y", 0)) for e in mouse_events], dtype=np.float64)
            t = np.array([e.get("time", e.get("t", i)) for i, e in enumerate(mouse_events)],
                        dtype=np.float64)

            # Normalize timestamps to seconds if needed
            if t[-1] > 1e6:  # likely milliseconds
                t = t / 1000.0

            feat = extract_features(dx, dy, t)
            features_list.append(feat)
        except Exception:
            continue

    print(f"  📁 Red Eclipse sessions: {len(features_list)} valid files")
    return features_list


def convert_all() -> Tuple[np.ndarray, int]:
    """
    Convert all available data sources into a feature matrix.

    Returns:
        (features_matrix, num_samples)
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_features = []

    print("🔄 Converting datasets...")

    # 1. Desktop logger sessions
    desktop = convert_desktop_sessions()
    all_features.extend(desktop)

    # 2. CS:GO data
    csgo = convert_csgo_data()
    all_features.extend(csgo)

    # 3. Red Eclipse data
    re = convert_redeclipse_data()
    all_features.extend(re)

    if not all_features:
        print("⚠️  No data found. Run: python -m data.download")
        return np.array([]), 0

    features = np.stack(all_features).astype(np.float32)

    # Save processed features
    out_path = PROCESSED_DIR / "real_features.npz"
    np.savez_compressed(str(out_path), features=features)
    print(f"\n✅ Saved {len(features)} feature vectors → {out_path}")

    return features, len(features)


def main():
    convert_all()


if __name__ == "__main__":
    main()
