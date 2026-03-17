"""
RawAccel settings.json builder — generates valid config files
that can be imported directly into the RawAccel application.

Usage:
    from rawaccel.config import build_settings_json
    from rawaccel.curves import CurveParams, AccelStyle

    params = CurveParams(style=AccelStyle.CLASSIC, acceleration=0.05, exponent=2.5)
    json_str = build_settings_json(params)
    # Save to file and import into RawAccel
"""
from __future__ import annotations

import json
from pathlib import Path

from .curves import CurveParams, AccelStyle


# Map our style names to RawAccel's internal mode names
_STYLE_TO_MODE = {
    AccelStyle.LINEAR: "linear",
    AccelStyle.CLASSIC: "classic",
    AccelStyle.NATURAL: "naturalgain",
    AccelStyle.POWER: "power",
    AccelStyle.SYNCHRONOUS: "motivity",
    AccelStyle.JUMP: "jump",
}


def build_accel_args(params: CurveParams) -> dict:
    """Build the 'Accel Args' block for one axis."""

    # Base args common to all styles
    args: dict = {
        "mode": _STYLE_TO_MODE[params.style],
        "gainSwitch": params.gain,
    }

    # Style-specific parameters
    if params.style == AccelStyle.LINEAR:
        args["acceleration"] = params.acceleration
        if params.offset > 0:
            args["offset"] = params.offset
        if params.cap_output > 0:
            args["capStyle"] = "output"
            args["cap"] = params.cap_output

    elif params.style == AccelStyle.CLASSIC:
        args["acceleration"] = params.acceleration
        args["exponent"] = params.exponent
        if params.offset > 0:
            args["offset"] = params.offset
        if params.cap_output > 0:
            args["capStyle"] = "output"
            args["cap"] = params.cap_output

    elif params.style == AccelStyle.NATURAL:
        args["decayRate"] = params.decay_rate
        if params.offset > 0:
            args["offset"] = params.offset
        if params.cap_output > 0:
            args["limit"] = params.cap_output

    elif params.style == AccelStyle.POWER:
        args["acceleration"] = params.acceleration
        args["exponent"] = params.exponent
        args["scale"] = params.scale
        if params.output_offset > 0:
            args["outputOffset"] = params.output_offset
        if params.cap_output > 0:
            args["capStyle"] = "output"
            args["cap"] = params.cap_output

    elif params.style == AccelStyle.SYNCHRONOUS:
        args["syncSpeed"] = params.sync_speed
        args["motivity"] = params.motivity
        args["gamma"] = params.gamma
        args["smooth"] = params.smooth

    elif params.style == AccelStyle.JUMP:
        args["jumpInput"] = params.jump_input
        args["jumpOutput"] = params.jump_output
        args["smooth"] = params.smooth

    return args


def build_settings_dict(params: CurveParams, dpi: int = 800,
                        poll_rate: int = 1000) -> dict:
    """
    Build a complete RawAccel settings dictionary.

    Args:
        params: curve parameters
        dpi: mouse DPI
        poll_rate: mouse polling rate in Hz

    Returns:
        dict matching RawAccel's settings.json structure
    """
    accel_args = build_accel_args(params)

    settings = {
        "Accel Args": {
            "x": accel_args,
            "y": accel_args,  # same curve for both axes
        },
        "Sensitivity": {
            "x": params.sens_multiplier,
            "y": params.sens_multiplier * params.yx_ratio,
        },
        "DPI": dpi,
        "Poll Rate Hz": poll_rate,
        "Rotation": 0.0,
        "Whole/Combined": True,
    }

    return settings


def build_settings_json(params: CurveParams, dpi: int = 800,
                        poll_rate: int = 1000, indent: int = 2) -> str:
    """Build a JSON string matching RawAccel's settings.json format."""
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
