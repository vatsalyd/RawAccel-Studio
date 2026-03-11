"""
Lightweight optimizer for finding good accel parameters.

Uses simulated annealing over the aim simulator — much faster than
full RL training. Returns results in seconds.
"""
import math
import copy
from typing import Dict, Optional

import numpy as np

from models.curve_param_config import (
    AccelParams,
    DEFAULT_ACCEL_PARAMS,
    clamp_params,
    sample_random_params,
    TUNE_KEYS,
    PARAM_BOUNDS,
)
from env.aim_sim.env_core import SimpleAimTask, HumanLikeControllerConfig


# ---------------------------------------------------------------------------
# Evaluate a parameter set
# ---------------------------------------------------------------------------

def evaluate_params(
    params: AccelParams,
    num_trials: int = 30,
    seed: int = 0,
    target_range_deg: float = 120.0,
    max_time: float = 0.6,
) -> Dict[str, float]:
    """Run aim tasks and return aggregate metrics."""
    rng = np.random.default_rng(seed)
    ctrl = HumanLikeControllerConfig()
    hits, times, errors, overshoots = [], [], [], []

    for _ in range(num_trials):
        task = SimpleAimTask(
            target_range_deg=target_range_deg,
            max_time=max_time,
            rng=rng,
        )
        done = False
        while not done:
            done, m = task.step_with_params(params, ctrl)
        hits.append(m["hit"])
        times.append(m["time"])
        errors.append(m["error"])
        overshoots.append(m["overshoot"])

    return {
        "hit_rate": float(np.mean(hits)),
        "avg_time": float(np.mean(times)),
        "avg_error": float(np.mean(errors)),
        "overshoot_rate": float(np.mean(overshoots)),
    }


def score_metrics(
    metrics: Dict[str, float],
    style: str = "balanced",
) -> float:
    """
    Score metrics based on play style.

    Styles:
        balanced : equal weight on hit rate, speed, and stability
        flicker  : heavier weight on speed (time-to-hit)
        tracker  : heavier weight on precision (low error, low overshoot)
    """
    hit = metrics["hit_rate"]
    speed = math.exp(-metrics["avg_time"] / 0.4)
    precision = math.exp(-metrics["avg_error"] / 20.0)
    stability = 1.0 - metrics["overshoot_rate"]

    if style == "flicker":
        return 1.5 * hit + 1.2 * speed + 0.5 * precision - 0.3 * (1 - stability)
    elif style == "tracker":
        return 1.5 * hit + 0.5 * speed + 1.2 * precision - 0.8 * (1 - stability)
    else:  # balanced
        return 1.5 * hit + 0.7 * speed + 0.7 * precision - 0.5 * (1 - stability)


# ---------------------------------------------------------------------------
# Simulated annealing optimizer
# ---------------------------------------------------------------------------

def _perturb_params(
    params: AccelParams,
    temperature: float,
    rng: np.random.Generator,
) -> AccelParams:
    """Randomly perturb params; larger perturbation at higher temperature."""
    d = params.as_dict()
    for key in TUNE_KEYS:
        lo, hi = PARAM_BOUNDS[key]
        scale = (hi - lo) * temperature * 0.1
        d[key] += rng.normal(0, scale)
    return clamp_params(AccelParams(**d))


def optimize_params(
    play_style: str = "balanced",
    dpi: int = 800,
    current_sens: float = 0.5,
    rank_tier: int = 3,
    iterations: int = 200,
    num_eval_trials: int = 20,
    seed: int = 42,
    progress_callback=None,
) -> Dict:
    """
    Find optimal accel params via simulated annealing.

    Parameters
    ----------
    play_style : 'balanced', 'flicker', or 'tracker'
    dpi : player's mouse DPI
    current_sens : current in-game sensitivity
    rank_tier : 1-5 (Iron→Radiant)
    iterations : SA iterations
    num_eval_trials : aim tasks per evaluation
    seed : RNG seed

    Returns
    -------
    dict with 'params', 'metrics', 'score', 'history'
    """
    rng = np.random.default_rng(seed)

    # Generate initial params influenced by player profile
    # Higher DPI → lower k values; higher rank → tighter curves
    dpi_factor = 800.0 / max(dpi, 100)
    rank_factor = rank_tier / 5.0  # 0.2 to 1.0

    initial = AccelParams(
        k1=0.002 * dpi_factor * (1.5 - 0.5 * rank_factor),
        a=0.8 + 0.4 * rank_factor,
        k2=0.001 * dpi_factor * (1.5 - 0.5 * rank_factor),
        b=1.0 + 0.3 * rank_factor,
        v0=300 + 400 * rank_factor,
        sens_min=0.15 + 0.1 * current_sens,
        sens_max=4.0 + 2.0 * current_sens,
    )
    initial = clamp_params(initial)

    best_params = copy.deepcopy(initial)
    best_metrics = evaluate_params(best_params, num_eval_trials, seed=seed)
    best_score = score_metrics(best_metrics, play_style)

    current_params = copy.deepcopy(best_params)
    current_score = best_score

    history = [{
        "iteration": 0,
        "score": best_score,
        "params": best_params.as_dict(),
        "metrics": best_metrics,
    }]

    for i in range(1, iterations + 1):
        temperature = 1.0 - (i / iterations)  # linear cooling

        candidate = _perturb_params(current_params, temperature, rng)
        candidate_metrics = evaluate_params(
            candidate, num_eval_trials, seed=seed + i
        )
        candidate_score = score_metrics(candidate_metrics, play_style)

        # Accept or reject
        delta = candidate_score - current_score
        if delta > 0 or rng.random() < math.exp(delta / max(temperature * 0.5, 0.01)):
            current_params = candidate
            current_score = candidate_score

            if current_score > best_score:
                best_params = copy.deepcopy(current_params)
                best_score = current_score
                best_metrics = candidate_metrics

        if i % 10 == 0:
            history.append({
                "iteration": i,
                "score": best_score,
                "params": best_params.as_dict(),
                "metrics": best_metrics,
            })
            if progress_callback:
                progress_callback(i, iterations, best_score)

    return {
        "params": best_params.as_dict(),
        "metrics": best_metrics,
        "score": best_score,
        "history": history,
    }


def params_to_rawaccel_config(params: Dict[str, float]) -> Dict:
    """Convert our AccelParams to a RawAccel-compatible JSON config."""
    return {
        "Sensitivity": {
            "x": 1.0,
            "y": 1.0,
        },
        "Acceleration": {
            "mode": "custom",
            "customCurve": {
                "type": "power",
                "segments": [
                    {
                        "speedRange": [0, params.get("v0", 400)],
                        "gain": params.get("k1", 0.002),
                        "exponent": params.get("a", 1.0),
                    },
                    {
                        "speedRange": [params.get("v0", 400), 99999],
                        "gain": params.get("k2", 0.001),
                        "exponent": params.get("b", 1.2),
                    },
                ],
                "cap": {
                    "min": params.get("sens_min", 0.2),
                    "max": params.get("sens_max", 6.0),
                },
            },
        },
    }
