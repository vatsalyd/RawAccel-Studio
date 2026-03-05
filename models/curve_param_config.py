from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class AccelParams:
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


DEFAULT_ACCEL_PARAMS = AccelParams(
    k1=0.002,
    a=1.0,
    k2=0.0008,
    b=1.2,
    v0=400.0,
    sens_min=0.2,
    sens_max=6.0,
)


def clamp_params(params: AccelParams) -> AccelParams:
    return AccelParams(
        k1=float(max(1e-6, min(params.k1, 10.0))),
        a=float(max(0.2, min(params.a, 3.0))),
        k2=float(max(1e-6, min(params.k2, 10.0))),
        b=float(max(0.2, min(params.b, 3.0))),
        v0=float(max(10.0, min(params.v0, 5000.0))),
        sens_min=float(max(0.01, min(params.sens_min, params.sens_max))),
        sens_max=float(max(params.sens_min, min(params.sens_max, 20.0))),
    )


def sensitivity_from_speed(speed: torch.Tensor, params: AccelParams) -> torch.Tensor:
    """
    speed: tensor of shape (...,) in counts/second
    returns: per-speed sensitivity
    """
    v = torch.clamp(speed, min=1e-6)
    v0 = params.v0
    low = params.k1 * (v ** params.a)
    high = params.k2 * (v ** params.b)

    sens = torch.where(v < v0, low, high)
    sens = torch.clamp(sens, min=params.sens_min, max=params.sens_max)
    return sens


def apply_accel(mouse_dx: torch.Tensor, dt: torch.Tensor, params: AccelParams) -> torch.Tensor:
    """
    mouse_dx: tensor (...,) in counts per step (1D for yaw)
    dt: tensor (...,) in seconds
    returns: view delta (same shape)
    """
    speed = torch.abs(mouse_dx) / torch.clamp(dt, min=1e-4)
    sens = sensitivity_from_speed(speed, params)
    return mouse_dx * sens