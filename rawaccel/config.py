"""
RawAccel settings.json builder — generates valid config files
that can be imported directly into the RawAccel v1.7 application.

The output matches the exact format from the RawAccel GUI export.
"""
from __future__ import annotations

import json
from pathlib import Path

from .curves import CurveParams, AccelStyle


# Map our style names to RawAccel's mode names
_STYLE_TO_MODE = {
    AccelStyle.LINEAR: "classic",       # linear is classic with exponent=1
    AccelStyle.CLASSIC: "classic",
    AccelStyle.NATURAL: "natural",
    AccelStyle.POWER: "power",
    AccelStyle.SYNCHRONOUS: "synchronous",
    AccelStyle.JUMP: "jump",
}


def _build_accel_block(params: CurveParams) -> dict:
    """Build one accel parameters block (used for both horizontal + vertical)."""
    mode = _STYLE_TO_MODE[params.style]

    cap_jump = {"x": 0.0, "y": 0.0}
    cap_mode = "output"

    if params.style == AccelStyle.JUMP:
        cap_jump = {"x": params.jump_input, "y": params.jump_output}
    elif params.cap_output > 0:
        cap_jump = {"x": 15.0, "y": params.cap_output}

    return {
        "mode": mode,
        "Gain / Velocity": params.gain,
        "inputOffset": params.offset,
        "outputOffset": params.output_offset,
        "acceleration": params.acceleration,
        "decayRate": params.decay_rate,
        "gamma": params.gamma,
        "motivity": params.motivity,
        "exponentClassic": params.exponent,
        "scale": params.scale,
        "exponentPower": params.exponent if params.style == AccelStyle.POWER else 0.05,
        "limit": params.cap_output if params.cap_output > 0 else 1.5,
        "syncSpeed": params.sync_speed,
        "smooth": params.smooth,
        "Cap / Jump": cap_jump,
        "Cap mode": cap_mode,
        "data": [],
    }


def build_settings_dict(params: CurveParams, dpi: int = 800,
                        poll_rate: int = 1000) -> dict:
    """
    Build a complete RawAccel v1.7 settings dictionary.

    Returns a dict matching the exact format from RawAccel's GUI export.
    """
    accel = _build_accel_block(params)

    return {
        "### Accel modes ###": "classic | jump | natural | synchronous | power | lut | noaccel",
        "### Cap modes ###": "in_out | input | output",
        "version": "1.7.0",
        "defaultDeviceConfig": {
            "disable": False,
            "Use constant time interval based on polling rate": False,
            "DPI (normalizes input speed unit: counts/ms -> in/s)": dpi,
            "Polling rate Hz (keep at 0 for automatic adjustment)": poll_rate,
        },
        "profiles": [
            {
                "name": "predicted",
                "Stretches domain for horizontal vs vertical inputs": {
                    "x": 1.0, "y": 1.0
                },
                "Stretches accel range for horizontal vs vertical inputs": {
                    "x": 1.0, "y": 1.0
                },
                "Whole or horizontal accel parameters": accel,
                "Vertical accel parameters": accel,
                "Input speed calculation parameters": {
                    "Whole/combined accel (set false for 'by component' mode)": True,
                    "lpNorm": 2.0,
                    "Time in ms after which an input is weighted at half its original value.": 0.0,
                    "Time in ms after which scale is weighted at half its original value.": 0.0,
                    "Time in ms after which an output is weighted at half its original value.": 0.0,
                },
                "Output DPI": float(dpi),
                "Y/X output DPI ratio (vertical sens multiplier)": params.yx_ratio,
                "L/R output DPI ratio (left sens multiplier)": 1.0,
                "U/D output DPI ratio (up sens multiplier)": 1.0,
                "Degrees of rotation": 1.0,
                "Degrees of angle snapping": 20.0,
                "Input Speed Cap": 0.0,
            }
        ],
        "devices": [],
    }


def build_settings_json(params: CurveParams, dpi: int = 800,
                        poll_rate: int = 1000, indent: int = 2) -> str:
    """Build a JSON string matching RawAccel v1.7's settings.json format."""
    settings = build_settings_dict(params, dpi, poll_rate)
    return json.dumps(settings, indent=indent)


def save_settings(params: CurveParams, path: str | Path,
                  dpi: int = 800, poll_rate: int = 1000) -> Path:
    """Save settings.json to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json_str = build_settings_json(params, dpi, poll_rate)
    path.write_text(json_str)
    return path
