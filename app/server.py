"""
RawAccel Studio — FastAPI backend.

Endpoints:
    POST /api/predict        Upload session JSON → predicted RawAccel curve
    POST /api/curve-preview  Preview a curve given params
    POST /api/export         Export as RawAccel settings.json
    GET  /api/sessions       List saved sessions
    GET  /api/model-status   Check if ML model is loaded
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from rawaccel.curves import AccelStyle, CurveParams, evaluate_curve
from rawaccel.config import build_settings_json, build_settings_dict

# ── Load ML model at startup ──
_predictor = None
try:
    from ml.predict import AccelCurvePredictor
    ckpt = Path("checkpoints/best_model.pt")
    if ckpt.exists():
        _predictor = AccelCurvePredictor(str(ckpt))
except Exception as e:
    print(f"⚠️  Model not loaded: {e}")

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RawAccel Studio", version="2.0")


# ─── Request models ─────────────────────────────────────────────────────────

class ExportRequest(BaseModel):
    params: dict
    dpi: int = 800
    poll_rate: int = 1000


class CurvePreviewRequest(BaseModel):
    style: str = "linear"
    acceleration: float = 0.5
    exponent: float = 2.0
    scale: float = 1.0
    motivity: float = 1.5
    gamma: float = 1.0
    smooth: float = 0.5
    decay_rate: float = 0.1
    sync_speed: float = 5.0
    offset: float = 0.0
    cap_output: float = 0.0
    output_offset: float = 0.0
    sens_multiplier: float = 1.0
    yx_ratio: float = 1.0
    gain: bool = False
    jump_input: float = 10.0
    jump_output: float = 1.5
    max_speed: float = 50.0


def _dict_to_params(d: dict) -> CurveParams:
    d = dict(d)  # copy
    style = AccelStyle(d.pop("style"))
    d.pop("max_speed", None)
    d.pop("dpi", None)
    d.pop("poll_rate", None)
    return CurveParams(style=style, **d)


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/api/predict")
async def predict_curve(
    file: UploadFile = File(...),
    dpi: int = Form(800),
    poll_rate: int = Form(1000),
):
    """Upload session JSON → predict optimal RawAccel curve."""
    if _predictor is None:
        raise HTTPException(503, "Model not loaded. Run: python -m ml.train")

    try:
        raw = await file.read()
        session = json.loads(raw.decode("utf-8"))
    except Exception:
        raise HTTPException(400, "Invalid JSON file")

    # Save uploaded session
    save_path = DATA_DIR / f"upload_{int(time.time())}.json"
    save_path.write_text(json.dumps(session), encoding="utf-8")

    try:
        # Predict
        from ml.features import load_session_arrays
        dx, dy, t = load_session_arrays(session)
        result = _predictor.predict_from_arrays(dx, dy, t)

        # Curve preview
        speeds, sens = evaluate_curve(result["params"], max_speed=50.0)

        # Flat response for the frontend
        return {
            "style": result["style"],
            "confidence": result["confidence"],
            "style_probs": result["style_probs"],
            "params_dict": result["params"].as_dict(),
            "feature_dim": int(len(dx)),
            "curve_preview": {
                "input_speeds": [float(s) for s in speeds],
                "output_speeds": [float(s) for s in sens],
            },
        }
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")


@app.post("/api/export")
async def export_settings(req: ExportRequest):
    """Generate RawAccel v1.7 settings.json from params dict."""
    try:
        params = _dict_to_params(req.params)
        json_str = build_settings_json(params, dpi=req.dpi, poll_rate=req.poll_rate)
        return {"settings_json": json_str}
    except Exception as e:
        raise HTTPException(400, f"Export error: {e}")


@app.post("/api/curve-preview")
async def curve_preview(req: CurvePreviewRequest):
    params = _dict_to_params(req.model_dump())
    speeds, sens = evaluate_curve(params, max_speed=req.max_speed)
    return {"input_speeds": speeds, "output_speeds": sens}


@app.get("/api/sessions")
async def list_sessions():
    sessions = []
    for f in sorted(DATA_DIR.iterdir(), reverse=True):
        if f.suffix != ".json":
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            sessions.append({
                "filename": f.name,
                "source": d.get("source", "unknown"),
                "num_events": len(d.get("samples", d.get("events", []))),
                "timestamp": d.get("recorded_at", 0),
            })
        except Exception:
            continue
    return {"sessions": sessions, "total": len(sessions)}


@app.get("/api/model-status")
async def model_status():
    return {"loaded": _predictor is not None}


# ─── Static + SPA ───────────────────────────────────────────────────────────

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    p = static_dir / "index.html"
    return HTMLResponse(p.read_text(encoding="utf-8")) if p.exists() else HTMLResponse("<h1>RawAccel Studio</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=True)
