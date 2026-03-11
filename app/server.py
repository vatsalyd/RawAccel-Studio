"""
RawAccel Studio — FastAPI backend.

Serves the frontend and provides REST API endpoints for:
- Curve preview (param set → sensitivity curve data points)
- Parameter optimization (player profile → recommended settings)
- Session recording (in-browser aim task → stored data)
- Settings export (RawAccel JSON config)

Usage:
    python -m app.server
    uvicorn app.server:app --reload
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.curve_param_config import (
    AccelParams,
    sensitivity_from_speed,
    PARAM_BOUNDS,
    TUNE_KEYS,
)
from app.optimizer import (
    optimize_params,
    evaluate_params,
    params_to_rawaccel_config,
)
import torch
import numpy as np

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="RawAccel Studio", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = PROJECT_ROOT / "data" / "recorded_sessions"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CurvePreviewRequest(BaseModel):
    k1: float = 0.002
    a: float = 1.0
    k2: float = 0.001
    b: float = 1.2
    v0: float = 400.0
    sens_min: float = 0.2
    sens_max: float = 6.0
    speed_min: float = 0.0
    speed_max: float = 3000.0
    num_points: int = 200


class PlayerProfile(BaseModel):
    dpi: int = 800
    sensitivity: float = 0.5
    rank: str = "gold"          # iron, bronze, silver, gold, platinum, diamond, ascendant, immortal, radiant
    agent_role: str = "duelist"  # duelist, initiator, controller, sentinel
    play_style: str = "balanced"  # balanced, flicker, tracker
    iterations: int = 200


class RecordedSession(BaseModel):
    events: List[dict]
    profile: Optional[dict] = None
    metadata: Optional[dict] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/curve-preview")
async def curve_preview(req: CurvePreviewRequest):
    """Return sensitivity curve data points for plotting."""
    speeds = torch.linspace(
        max(req.speed_min, 1.0), req.speed_max, req.num_points
    )
    params = AccelParams(
        k1=req.k1, a=req.a, k2=req.k2, b=req.b,
        v0=req.v0, sens_min=req.sens_min, sens_max=req.sens_max,
    )
    sens_values = sensitivity_from_speed(speeds, params)

    return {
        "speeds": speeds.tolist(),
        "sensitivities": sens_values.tolist(),
        "params": params.as_dict(),
    }


RANK_TO_TIER = {
    "iron": 1, "bronze": 1,
    "silver": 2,
    "gold": 3,
    "platinum": 3,
    "diamond": 4,
    "ascendant": 4,
    "immortal": 5,
    "radiant": 5,
}


@app.post("/api/optimize")
async def optimize(profile: PlayerProfile):
    """Run SA optimization and return recommended accel params."""
    rank_tier = RANK_TO_TIER.get(profile.rank.lower(), 3)

    result = optimize_params(
        play_style=profile.play_style,
        dpi=profile.dpi,
        current_sens=profile.sensitivity,
        rank_tier=rank_tier,
        iterations=profile.iterations,
    )

    # Also generate the curve data for the result
    speeds = torch.linspace(1.0, 3000.0, 200)
    opt_params = AccelParams(**{
        **result["params"],
        "sens_min": result["params"].get("sens_min", 0.2),
        "sens_max": result["params"].get("sens_max", 6.0),
    })
    sens_values = sensitivity_from_speed(speeds, opt_params)

    # Compare with default
    from models.curve_param_config import DEFAULT_ACCEL_PARAMS
    default_sens = sensitivity_from_speed(speeds, DEFAULT_ACCEL_PARAMS)
    default_metrics = evaluate_params(DEFAULT_ACCEL_PARAMS, num_trials=30)

    return {
        "recommended": result["params"],
        "metrics": result["metrics"],
        "score": result["score"],
        "curve": {
            "speeds": speeds.tolist(),
            "optimized": sens_values.tolist(),
            "default": default_sens.tolist(),
        },
        "baseline_metrics": default_metrics,
        "history": result["history"][-5:],  # last 5 checkpoints
    }


@app.post("/api/record-session")
async def record_session(session: RecordedSession):
    """Save a recorded aim session from the browser."""
    timestamp = int(time.time())
    filename = f"session_{timestamp}.json"
    filepath = DATA_DIR / filename

    data = {
        "events": session.events,
        "profile": session.profile,
        "metadata": session.metadata,
        "timestamp": timestamp,
    }
    with open(filepath, "w") as f:
        json.dump(data, f)

    return {"status": "saved", "filename": filename, "num_events": len(session.events)}


@app.get("/api/export/rawaccel")
async def export_rawaccel(
    k1: float = 0.002, a: float = 1.0,
    k2: float = 0.001, b: float = 1.2,
    v0: float = 400.0,
    sens_min: float = 0.2, sens_max: float = 6.0,
):
    """Export current params as a RawAccel-compatible JSON config."""
    params = {
        "k1": k1, "a": a, "k2": k2, "b": b,
        "v0": v0, "sens_min": sens_min, "sens_max": sens_max,
    }
    config = params_to_rawaccel_config(params)
    return config


@app.get("/api/param-bounds")
async def get_param_bounds():
    """Return valid parameter ranges for the UI sliders."""
    return {
        key: {"min": bounds[0], "max": bounds[1]}
        for key, bounds in PARAM_BOUNDS.items()
        if key in TUNE_KEYS
    }


@app.get("/api/sessions")
async def list_sessions():
    """List all recorded aim sessions with summary metadata."""
    sessions = []
    for filename in sorted(DATA_DIR.iterdir(), reverse=True):
        if not filename.suffix == ".json":
            continue
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            sessions.append({
                "filename": filename.name,
                "source": data.get("source", "browser"),
                "num_events": len(data.get("events", data.get("samples", []))),
                "profile": data.get("profile", {}),
                "metadata": data.get("metadata", data.get("summary", {})),
                "timestamp": data.get("timestamp", data.get("recorded_at", 0)),
            })
        except (json.JSONDecodeError, IOError):
            continue
    return {"sessions": sessions, "total": len(sessions)}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
