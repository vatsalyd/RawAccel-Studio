"""
Aim simulation environment for RL-based accel parameter tuning.

Provides:
- SimpleAimTask: 1D yaw-only aiming task with human-like controller
- AimEnv: Gymnasium environment for RL agents to optimise AccelParams
"""
import math
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional

import gymnasium as gym
import numpy as np
import torch

from models.curve_param_config import (
    AccelParams,
    DEFAULT_ACCEL_PARAMS,
    clamp_params,
    apply_accel,
    sample_random_params,
)


# ---------------------------------------------------------------------------
# Human-like controller
# ---------------------------------------------------------------------------

@dataclass
class HumanLikeControllerConfig:
    """Simple proportional controller with noise to mimic human aim."""
    base_gain: float = 0.8
    noise_std: float = 0.05
    reaction_time: float = 0.05  # seconds (unused for now, placeholder)


# ---------------------------------------------------------------------------
# Aim task
# ---------------------------------------------------------------------------

class SimpleAimTask:
    """
    1-D yaw-only target at a random angular offset.
    A human-like controller moves the crosshair via mouse → accel curve → view.
    """

    def __init__(
        self,
        target_range_deg: float = 120.0,
        max_time: float = 0.6,
        hit_threshold_deg: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.target_range_deg = target_range_deg
        self.max_time = max_time
        self.hit_threshold_deg = hit_threshold_deg
        self._rng = rng or np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        self.time = 0.0
        self.view_yaw = 0.0
        self.target_yaw = float(
            self._rng.uniform(-self.target_range_deg, self.target_range_deg)
        )
        self.hit = False
        self.overshoot = False

    def step_with_params(
        self,
        accel_params: AccelParams,
        ctrl_cfg: HumanLikeControllerConfig,
        dt: float = 0.01,
    ) -> Tuple[bool, Dict[str, float]]:
        if self.hit or self.time >= self.max_time:
            return True, self._metrics()

        self.time += dt
        error = self.target_yaw - self.view_yaw

        # Proportional controller + noise
        controller_output = ctrl_cfg.base_gain * error
        controller_output += (
            self._rng.standard_normal() * ctrl_cfg.noise_std * abs(error)
        )

        mouse_dx_counts = controller_output * 100.0
        mouse_dx = torch.tensor(mouse_dx_counts, dtype=torch.float32)
        dt_t = torch.tensor(dt, dtype=torch.float32)

        view_delta = apply_accel(mouse_dx, dt_t, accel_params)
        self.view_yaw += float(view_delta.item())

        # Detect overshoot
        prev_sign = math.copysign(1.0, error) if error != 0 else 0.0
        new_error = self.target_yaw - self.view_yaw
        new_sign = math.copysign(1.0, new_error) if new_error != 0 else 0.0
        if prev_sign != 0 and new_sign != 0 and prev_sign != new_sign:
            self.overshoot = True

        if abs(new_error) < self.hit_threshold_deg:
            self.hit = True

        done = self.hit or self.time >= self.max_time
        return done, self._metrics()

    def _metrics(self) -> Dict[str, float]:
        error = abs(self.target_yaw - self.view_yaw)
        return {
            "time": self.time,
            "error": error,
            "hit": float(self.hit),
            "overshoot": float(self.overshoot),
        }


# ---------------------------------------------------------------------------
# Gymnasium RL environment
# ---------------------------------------------------------------------------

@dataclass
class AimEnvConfig:
    """Configuration for the RL aim environment."""
    num_eval_trials: int = 5
    target_range_deg: float = 120.0
    max_aim_time: float = 0.6
    hit_threshold_deg: float = 1.0
    max_episode_steps: int = 50
    action_scale: float = 0.05  # max relative delta per step


class AimEnv(gym.Env):
    """
    RL environment whose **state** is summary performance of the current
    accel parameters on randomly sampled aiming tasks.

    **Action**: small continuous deltas to the 5 tuneable accel parameters
    (k1, a, k2, b, v0).

    **Reward**: weighted combination of hit-rate, time-to-hit, error, and
    overshoot penalty.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[AimEnvConfig] = None):
        super().__init__()
        self.cfg = config or AimEnvConfig()
        self.param_keys = ["k1", "a", "k2", "b", "v0"]

        obs_dim = 4 + len(self.param_keys)  # 4 metrics + 5 normalised params
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-self.cfg.action_scale,
            high=self.cfg.action_scale,
            shape=(len(self.param_keys),),
            dtype=np.float32,
        )

        self.params = DEFAULT_ACCEL_PARAMS
        self.ctrl_cfg = HumanLikeControllerConfig()
        self.episode_steps = 0
        self._rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.params = clamp_params(self.params)
        self.episode_steps = 0
        metrics = self._eval_params()
        obs = self._metrics_to_obs(metrics)
        return obs, {}

    def step(self, action):
        delta = np.clip(action, self.action_space.low, self.action_space.high)
        p = self.params.as_dict()
        for i, key in enumerate(self.param_keys):
            scale = abs(p[key]) if p[key] != 0 else 1.0
            p[key] += float(delta[i]) * scale
        self.params = clamp_params(AccelParams(**p))

        metrics = self._eval_params()
        obs = self._metrics_to_obs(metrics)
        reward = self._compute_reward(metrics)
        self.episode_steps += 1
        terminated = False
        truncated = self.episode_steps >= self.cfg.max_episode_steps

        info = {
            "metrics": metrics,
            "params": self.params.as_dict(),
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_params(self) -> Dict[str, float]:
        times, errors, hits, overshoots = [], [], [], []
        for _ in range(self.cfg.num_eval_trials):
            task = SimpleAimTask(
                target_range_deg=self.cfg.target_range_deg,
                max_time=self.cfg.max_aim_time,
                hit_threshold_deg=self.cfg.hit_threshold_deg,
                rng=self._rng,
            )
            done = False
            while not done:
                done, m = task.step_with_params(self.params, self.ctrl_cfg)
            times.append(m["time"])
            errors.append(m["error"])
            hits.append(m["hit"])
            overshoots.append(m["overshoot"])
        return {
            "hit_rate": float(np.mean(hits)),
            "avg_time": float(np.mean(times)),
            "avg_error": float(np.mean(errors)),
            "overshoot_rate": float(np.mean(overshoots)),
        }

    def _metrics_to_obs(self, m: Dict[str, float]) -> np.ndarray:
        metrics_vec = np.array(
            [
                m["hit_rate"],
                np.exp(-m["avg_time"] / 0.5),
                np.exp(-m["avg_error"] / 30.0),
                1.0 - m["overshoot_rate"],
            ],
            dtype=np.float32,
        )
        p = self.params.as_dict()
        norm_params = np.array(
            [
                float(p["k1"] / 10.0),
                float(p["a"] / 3.0),
                float(p["k2"] / 10.0),
                float(p["b"] / 3.0),
                float(p["v0"] / 5000.0),
            ],
            dtype=np.float32,
        )
        return np.clip(
            np.concatenate([metrics_vec, norm_params]), 0.0, 1.0
        )

    def _compute_reward(self, m: Dict[str, float]) -> float:
        hit_term = m["hit_rate"]
        time_term = np.exp(-m["avg_time"] / 0.4)
        error_term = np.exp(-m["avg_error"] / 20.0)
        overshoot_penalty = m["overshoot_rate"]
        return float(
            1.5 * hit_term + 0.7 * time_term + 0.7 * error_term - 0.5 * overshoot_penalty
        )

    def render(self):
        pass  # Placeholder for future visualisation
