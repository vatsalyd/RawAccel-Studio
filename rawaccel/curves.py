"""
RawAccel curve math — replicates the acceleration formulas from
https://github.com/RawAccelOfficial/rawaccel

Each function takes an input speed (counts/ms) and curve parameters,
and returns the sensitivity multiplier at that speed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AccelStyle(str, Enum):
    """Acceleration curve types supported by RawAccel."""
    LINEAR = "linear"
    CLASSIC = "classic"
    NATURAL = "natural"
    POWER = "power"
    SYNCHRONOUS = "synchronous"
    JUMP = "jump"


@dataclass
class CurveParams:
    """
    Complete set of parameters for one RawAccel acceleration curve.

    Mirrors the parameters in RawAccel's settings.json.
    """
    style: AccelStyle = AccelStyle.LINEAR

    # ── Core parameters (meaning varies by style) ──
    acceleration: float = 0.5       # rate of sensitivity increase
    exponent: float = 2.0           # curve exponent (Classic / Power)
    scale: float = 1.0              # speed scale factor (Power)
    motivity: float = 1.5           # proportional change (Synchronous)
    gamma: float = 1.0              # speed of change (Synchronous)
    smooth: float = 0.5             # tailing smoothness (Synchronous / Jump)
    decay_rate: float = 0.1         # sensitivity approach rate (Natural)
    sync_speed: float = 5.0         # synchronous speed

    # ── Offset & Cap ──
    offset: float = 0.0             # speed before accel kicks in (counts/ms)
    cap_input: float = 0.0          # cap by input speed (0 = no cap)
    cap_output: float = 0.0         # cap by output sensitivity (0 = no cap)
    output_offset: float = 0.0      # output starting ratio (Power mode, usually 1)

    # ── Global ──
    sens_multiplier: float = 1.0    # base sensitivity multiplier
    yx_ratio: float = 1.0           # vertical / horizontal ratio
    gain: bool = False              # True = shape applied as gain curve

    # ── Jump mode ──
    jump_input: float = 10.0        # speed threshold for jump
    jump_output: float = 1.5        # sensitivity above threshold

    def as_dict(self) -> dict:
        """Return a flat parameter dict for ML training."""
        return {
            "style": self.style.value,
            "acceleration": self.acceleration,
            "exponent": self.exponent,
            "scale": self.scale,
            "motivity": self.motivity,
            "gamma": self.gamma,
            "smooth": self.smooth,
            "decay_rate": self.decay_rate,
            "sync_speed": self.sync_speed,
            "offset": self.offset,
            "cap_input": self.cap_input,
            "cap_output": self.cap_output,
            "output_offset": self.output_offset,
            "sens_multiplier": self.sens_multiplier,
            "yx_ratio": self.yx_ratio,
            "gain": self.gain,
            "jump_input": self.jump_input,
            "jump_output": self.jump_output,
        }


# ─── Parameter bounds for each style (for random sampling) ─────────────────

STYLE_PARAM_BOUNDS: dict[AccelStyle, dict[str, tuple[float, float]]] = {
    AccelStyle.LINEAR: {
        "acceleration": (0.01, 2.0),
        "offset": (0.0, 20.0),
        "cap_output": (1.0, 5.0),
    },
    AccelStyle.CLASSIC: {
        "acceleration": (0.01, 1.0),
        "exponent": (1.5, 4.0),
        "offset": (0.0, 20.0),
        "cap_output": (1.0, 5.0),
    },
    AccelStyle.NATURAL: {
        "decay_rate": (0.01, 1.0),
        "offset": (0.0, 20.0),
        "cap_output": (1.0, 5.0),
    },
    AccelStyle.POWER: {
        "acceleration": (0.01, 2.0),
        "exponent": (0.01, 1.0),
        "scale": (0.5, 5.0),
        "output_offset": (0.5, 1.5),
        "cap_output": (1.0, 5.0),
    },
    AccelStyle.SYNCHRONOUS: {
        "sync_speed": (1.0, 20.0),
        "motivity": (1.1, 4.0),
        "gamma": (0.5, 3.0),
        "smooth": (0.0, 1.0),
    },
    AccelStyle.JUMP: {
        "jump_input": (2.0, 30.0),
        "jump_output": (1.1, 4.0),
        "smooth": (0.0, 1.0),
    },
}


# ─── Curve evaluation functions ────────────────────────────────────────────

def _apply_offset(speed: float, offset: float) -> float:
    """Apply gain offset — shift the speed by the offset amount."""
    return max(0.0, speed - offset)


def _apply_cap(sens: float, cap_output: float) -> float:
    """Cap the sensitivity at cap_output (if > 0)."""
    if cap_output > 0:
        return min(sens, cap_output)
    return sens


def linear(speed: float, p: CurveParams) -> float:
    """
    Linear: sensitivity = 1 + acceleration * speed
    Simplest curve — straight line.
    """
    v = _apply_offset(speed, p.offset)
    sens = 1.0 + p.acceleration * v
    return _apply_cap(sens, p.cap_output)


def classic(speed: float, p: CurveParams) -> float:
    """
    Classic (Quake 3 / InterAccel style):
    sensitivity = 1 + acceleration * speed ^ exponent
    """
    v = _apply_offset(speed, p.offset)
    sens = 1.0 + p.acceleration * (v ** p.exponent)
    return _apply_cap(sens, p.cap_output)


def natural(speed: float, p: CurveParams) -> float:
    """
    Natural: sensitivity starts at 1 and approaches a maximum.
    sensitivity = cap * (1 - e^(-decay_rate * speed))
    A concave curve that rises quickly then flattens.
    """
    v = _apply_offset(speed, p.offset)
    cap = p.cap_output if p.cap_output > 0 else 3.0
    sens = 1.0 + (cap - 1.0) * (1.0 - math.exp(-p.decay_rate * v))
    return sens


def power(speed: float, p: CurveParams) -> float:
    """
    Power (CS:GO / Source Engine style):
    sensitivity = output_offset + acceleration * (speed * scale) ^ exponent
    """
    v = speed  # Power mode doesn't use gain offset
    base = p.output_offset if p.output_offset > 0 else 0.0
    sens = base + p.acceleration * ((v * p.scale) ** p.exponent)
    return _apply_cap(sens, p.cap_output) if p.cap_output > 0 else sens


def synchronous(speed: float, p: CurveParams) -> float:
    """
    Synchronous: logarithmically symmetrical change around sync_speed.
    sensitivity ranges from 1/motivity to motivity.
    Uses tanh-like transition controlled by gamma and smooth.
    """
    if speed <= 0:
        return 1.0 / p.motivity

    log_ratio = math.log(speed / p.sync_speed) if speed > 0 else -10.0
    t = log_ratio * p.gamma

    # Smooth function (tanh for smooth=0.5, sigmoid-like otherwise)
    if p.smooth <= 0:
        sig = 1.0 if t >= 0 else 0.0
    else:
        sig = 0.5 * (1.0 + math.tanh(t / (2.0 * p.smooth)))

    log_mot = math.log(p.motivity)
    log_sens = (2.0 * sig - 1.0) * log_mot
    return math.exp(log_sens)


def jump(speed: float, p: CurveParams) -> float:
    """
    Jump: one sensitivity below threshold, another above.
    Smooth parameter controls the transition sharpness.
    """
    if p.smooth <= 0:
        return p.jump_output if speed > p.jump_input else 1.0

    t = (speed - p.jump_input) / (p.jump_input * 0.1 + 0.01)
    sig = 0.5 * (1.0 + math.tanh(t * (1.0 / max(p.smooth, 0.01))))
    return 1.0 + (p.jump_output - 1.0) * sig


# ─── Dispatcher ────────────────────────────────────────────────────────────

STYLE_FN = {
    AccelStyle.LINEAR: linear,
    AccelStyle.CLASSIC: classic,
    AccelStyle.NATURAL: natural,
    AccelStyle.POWER: power,
    AccelStyle.SYNCHRONOUS: synchronous,
    AccelStyle.JUMP: jump,
}


def evaluate(speed: float, params: CurveParams) -> float:
    """
    Evaluate the RawAccel sensitivity multiplier at a given input speed.

    Args:
        speed: input speed in counts/ms
        params: complete curve parameters

    Returns:
        sensitivity multiplier (applied to mouse output)
    """
    fn = STYLE_FN[params.style]
    sens = fn(speed, params)

    # Apply global sensitivity multiplier
    return sens * params.sens_multiplier


def evaluate_curve(params: CurveParams, max_speed: float = 50.0,
                   n_points: int = 200) -> tuple[list[float], list[float]]:
    """
    Generate a full curve: input speeds → sensitivity values.

    Args:
        params: curve parameters
        max_speed: maximum input speed (counts/ms)
        n_points: number of evaluation points

    Returns:
        (speeds, sensitivities) — two parallel lists
    """
    speeds = [max_speed * i / (n_points - 1) for i in range(n_points)]
    sens = [evaluate(s, params) for s in speeds]
    return speeds, sens
