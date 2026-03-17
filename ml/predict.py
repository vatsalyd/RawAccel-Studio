"""
Inference — predict RawAccel curve from mouse data.

Usage:
    from ml.predict import AccelCurvePredictor
    predictor = AccelCurvePredictor("checkpoints/best_model.pt")
    params = predictor.predict_from_file("data/raw/session_123.json")
    print(params)  # CurveParams(style=classic, acceleration=0.3, ...)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from rawaccel.curves import CurveParams
from rawaccel.config import build_settings_json
from ml.model import AccelPredictor
from ml.features import extract_features, load_session_arrays
from ml.dataset import labels_to_params, STYLES_LIST, NUM_STYLES


class AccelCurvePredictor:
    """Load a trained model and predict RawAccel params from mouse data."""

    def __init__(self, checkpoint_path: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = AccelPredictor()
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Loaded model from {checkpoint_path} (val_loss={ckpt.get('val_loss', '?')})")

    def predict(self, features: np.ndarray) -> dict:
        """
        Predict curve params from a feature vector.

        Returns:
            dict with 'params' (CurveParams), 'style_probs', 'confidence'
        """
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0).to(self.device)
            style_logits, params_pred = self.model(x)

            style_probs = torch.softmax(style_logits, dim=1)[0].cpu().numpy()
            style_idx = int(style_probs.argmax())
            continuous = params_pred[0].cpu().numpy()

        params = labels_to_params(style_idx, continuous)

        return {
            "params": params,
            "style": params.style.value,
            "style_probs": {s.value: float(p) for s, p in zip(STYLES_LIST, style_probs)},
            "confidence": float(style_probs[style_idx]),
        }

    def predict_from_arrays(self, dx: np.ndarray, dy: np.ndarray,
                            timestamps: np.ndarray) -> dict:
        """Predict from raw mouse arrays."""
        features = extract_features(dx, dy, timestamps)
        return self.predict(features)

    def predict_from_file(self, session_path: str) -> dict:
        """Predict from a saved session JSON file."""
        with open(session_path, "r") as f:
            data = json.load(f)
        dx, dy, t = load_session_arrays(data)
        return self.predict_from_arrays(dx, dy, t)

    def predict_and_export(self, session_path: str, output_path: str = None,
                           dpi: int = 800) -> str:
        """
        Full pipeline: session file → prediction → RawAccel settings.json

        Returns the JSON string.
        """
        result = self.predict_from_file(session_path)
        json_str = build_settings_json(result["params"], dpi=dpi)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(json_str)
            print(f"💾 Settings saved → {output_path}")

        return json_str
