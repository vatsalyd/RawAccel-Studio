import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import gymnasium as gym
import numpy as np

from models.curve_param_config import AccelParams, DEFAULT_ACCEL_PARAMS, clamp_params, apply_accel
import torch


@dataclass
class HumanLikeControllerConfig:
    base_gain: float = 0.8
    noise_std: float = 0.05
    reaction_time: float = 0.05  # seconds


class SimpleAimTask:
    """
    1D yaw-only target at angular offset; we approximate a simple
    “human” who moves mouse in proportion to remaining error.
    """

    def __init__(self, target_range_deg: float = 120.0, max_time: float = 0.6):
        self.target_range_deg = target_range_deg
        self.max_time = max_time
        self.reset()

    def reset(self) -> None:
        self.time = 0.0
        self.view_yaw = 0.0
        self.target_yaw = float(np.random.uniform(-self.target_range_deg, self.target_range_deg))
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
        controller_output = ctrl_cfg.base_gain * error
        controller_output += np.random.randn() * ctrl_cfg.noise_std * abs(error)

        mouse_dx_counts = controller_output * 100.0
        mouse_dx = torch.tensor(mouse_dx_counts, dtype=torch.float32)
        dt_t = torch.tensor(dt, dtype=torch.float32)

        view_delta = apply_accel(mouse_dx, dt_t, accel_params)
        self.view_yaw += float(view_delta.item())

        prev_error_sign = math.copysign(1.0, error) if error != 0 else 0.0
        new_error = self.target_yaw - self.view_yaw
        new_error_sign = math.copysign(1.0, new_error) if new_error != 0 else 0.0
        if prev_error_sign != 0 and new_error_sign != 0 and prev_error_sign != new_error_sign:
            self.overshoot = True

        if abs(new_error) < 1.0:
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


class AimEnv(gym.Env):
    """
    RL environment whose state is summary stats of how well current
    accel parameters perform on randomly sampled aiming tasks.
    Action: small deltas to accel parameters.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.param_keys = ["k1", "a", "k2", "b", "v0"]
        self.obs_low = np.array([0.0, 0.0, 0.0, 0.0] + [0.0] * len(self.param_keys), dtype=np.float32)
        self.obs_high = np.array([1.0, 1.0, 1.0, 1.0] + [1.0] * len(self.param_keys), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-0.05, high=0.05, shape=(len(self.param_keys),), dtype=np.float32)

        self.params = DEFAULT_ACCEL_PARAMS
        self.ctrl_cfg = HumanLikeControllerConfig()
        self.episode_steps = 0
        self.max_episode_steps = 50

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.params = clamp_params(self.params)
        self.episode_steps = 0
        metrics = self._eval_params()
        obs = self._metrics_to_obs(metrics)
        return obs, {}

    def step(self, action):
        delta = np.clip(action, self.action_space.low, self.action_space.high)
        p = self.params.as_dict()
        for i, key in enumerate(self.param_keys):
            p[key] += float(delta[i]) * abs(p[key] if p[key] != 0 else 1.0)
        self.params = clamp_params(AccelParams(**p))

        metrics = self._eval_params()
        obs = self._metrics_to_obs(metrics)
        reward = self._compute_reward(metrics)
        self.episode_steps += 1
        terminated = False
        truncated = self.episode_steps >= self.max_episode_steps
        return obs, reward, terminated, truncated, {}

    def _eval_params(self) -> Dict[str, float]:
        num_trials = 5
        times, errors, hits, overshoots = [], [], [], []
        for _ in range(num_trials):
            task = SimpleAimTask()
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
        hit_rate = m["hit_rate"]
        avg_time = m["avg_time"]
        avg_error = m["avg_error"]
        overshoot_rate = m["overshoot_rate"]

        time_score = np.exp(-avg_time / 0.5)
        error_score = np.exp(-avg_error / 30.0)

        metrics_vec = np.array(
            [hit_rate, time_score, error_score, 1.0 - overshoot_rate],
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

        return np.concatenate([metrics_vec, norm_params], axis=0)

    def _compute_reward(self, m: Dict[str, float]) -> float:
        hit_term = m["hit_rate"]
        time_term = np.exp(-m["avg_time"] / 0.4)
        error_term = np.exp(-m["avg_error"] / 20.0)
        overshoot_penalty = m["overshoot_rate"]
        return float(1.5 * hit_term + 0.7 * time_term + 0.7 * error_term - 0.5 * overshoot_penalty)