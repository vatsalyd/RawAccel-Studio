"""
Shared acceleration-curve parameterisation used by every component in the
project (RL, inverse modelling, ideal-curve prediction, anomaly detection).
"""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------

@dataclass
class AccelParams:
    """
    Two-branch power-law acceleration curve:
        sens(v) = k1 * v^a   for v < v0
        sens(v) = k2 * v^b   for v >= v0
    clamped to [sens_min, sens_max].
    """
    k1: float
    a: float
    k2: float
    b: float
    v0: float
    sens_min: float
    sens_max: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "k1": self.k1,
            "a": self.a,
            "k2": self.k2,
            "b": self.b,
            "v0": self.v0,
            "sens_min": self.sens_min,
            "sens_max": self.sens_max,
        }


# Ordered key list used when converting to / from tensors
PARAM_KEYS: List[str] = ["k1", "a", "k2", "b", "v0", "sens_min", "sens_max"]
TUNE_KEYS: List[str] = ["k1", "a", "k2", "b", "v0"]  # subset tuned by RL

# Per-parameter valid ranges (used for clamping & normalisation)
PARAM_BOUNDS: Dict[str, tuple] = {
    "k1":       (1e-6, 10.0),
    "a":        (0.2,  3.0),
    "k2":       (1e-6, 10.0),
    "b":        (0.2,  3.0),
    "v0":       (10.0, 5000.0),
    "sens_min": (0.01, 20.0),
    "sens_max": (0.01, 20.0),
}


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_ACCEL_PARAMS = AccelParams(
    k1=0.002,
    a=1.0,
    k2=0.0008,
    b=1.2,
    v0=400.0,
    sens_min=0.2,
    sens_max=6.0,
)


# ---------------------------------------------------------------------------
# Clamping / validation
# ---------------------------------------------------------------------------

def clamp_params(params: AccelParams) -> AccelParams:
    """Clamp every field to its valid range; ensure sens_min <= sens_max."""
    d = params.as_dict()
    for key in PARAM_KEYS:
        lo, hi = PARAM_BOUNDS[key]
        d[key] = float(max(lo, min(d[key], hi)))
    d["sens_min"] = float(min(d["sens_min"], d["sens_max"]))
    d["sens_max"] = float(max(d["sens_min"], d["sens_max"]))
    return AccelParams(**d)


# ---------------------------------------------------------------------------
# Tensor ↔ AccelParams conversion
# ---------------------------------------------------------------------------

def params_to_tensor(params: AccelParams, keys: List[str] = None) -> torch.Tensor:
    """Convert AccelParams to a 1-D float tensor (default: all 7 params)."""
    keys = keys or PARAM_KEYS
    d = params.as_dict()
    return torch.tensor([d[k] for k in keys], dtype=torch.float32)


def tensor_to_params(t: torch.Tensor, keys: List[str] = None) -> AccelParams:
    """Reconstruct AccelParams from a 1-D tensor. Missing keys use defaults."""
    keys = keys or PARAM_KEYS
    d = DEFAULT_ACCEL_PARAMS.as_dict()
    for i, k in enumerate(keys):
        d[k] = float(t[i].item())
    return AccelParams(**d)


def normalise_params(params: AccelParams, keys: List[str] = None) -> torch.Tensor:
    """Return params normalised to [0, 1] based on PARAM_BOUNDS."""
    keys = keys or PARAM_KEYS
    d = params.as_dict()
    vals = []
    for k in keys:
        lo, hi = PARAM_BOUNDS[k]
        vals.append((d[k] - lo) / (hi - lo + 1e-12))
    return torch.tensor(vals, dtype=torch.float32)


def denormalise_params(t: torch.Tensor, keys: List[str] = None) -> AccelParams:
    """Inverse of normalise_params."""
    keys = keys or PARAM_KEYS
    d = DEFAULT_ACCEL_PARAMS.as_dict()
    for i, k in enumerate(keys):
        lo, hi = PARAM_BOUNDS[k]
        d[k] = float(t[i].item()) * (hi - lo) + lo
    return AccelParams(**d)


# ---------------------------------------------------------------------------
# Random sampling
# ---------------------------------------------------------------------------

def sample_random_params(rng: np.random.Generator = None) -> AccelParams:
    """Sample a random, clamped AccelParams set."""
    rng = rng or np.random.default_rng()
    return clamp_params(
        AccelParams(
            k1=float(10 ** rng.uniform(-4, -1)),
            a=float(rng.uniform(0.6, 1.6)),
            k2=float(10 ** rng.uniform(-4, -1)),
            b=float(rng.uniform(0.6, 1.8)),
            v0=float(rng.uniform(100, 2000)),
            sens_min=float(rng.uniform(0.1, 0.8)),
            sens_max=float(rng.uniform(2.0, 10.0)),
        )
    )


# ---------------------------------------------------------------------------
# Curve evaluation (differentiable via PyTorch)
# ---------------------------------------------------------------------------

def sensitivity_from_speed(speed: torch.Tensor, params: AccelParams) -> torch.Tensor:
    """
    Compute sensitivity for each mouse speed (counts/s).

    speed : tensor of shape (...,)
    returns: tensor of same shape
    """
    v = torch.clamp(speed, min=1e-6)
    low = params.k1 * (v ** params.a)
    high = params.k2 * (v ** params.b)
    sens = torch.where(v < params.v0, low, high)
    return torch.clamp(sens, min=params.sens_min, max=params.sens_max)


def apply_accel(
    mouse_dx: torch.Tensor,
    dt: torch.Tensor,
    params: AccelParams,
) -> torch.Tensor:
    """
    Convert raw mouse delta (counts per step) to view-angle delta.

    mouse_dx : tensor (...,) in counts per step (1-D = yaw only)
    dt       : tensor (...,) in seconds
    returns  : tensor (...,) — view delta in degrees
    """
    speed = torch.abs(mouse_dx) / torch.clamp(dt, min=1e-4)
    sens = sensitivity_from_speed(speed, params)
    return mouse_dx * sens