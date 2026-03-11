"""
P1: RL auto-tuning of acceleration curve parameters.

Usage (standalone):
    python -m experiments.rl_auto_tune.train_rl_auto_tune
    python -m experiments.rl_auto_tune.train_rl_auto_tune --config experiments/configs/rl_auto_tune.yaml

Usage (via dispatcher):
    python experiments/run_experiments.py --name rl_auto_tune
"""
import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback

from env.aim_sim.env_core import AimEnv, AimEnvConfig
from models.curve_param_config import DEFAULT_ACCEL_PARAMS


# ---------------------------------------------------------------------------
# Custom callback: logs learned accel params & metrics to TensorBoard
# ---------------------------------------------------------------------------

class ParamLoggingCallback(BaseCallback):
    """Log accel-curve parameters and evaluation metrics at each rollout end."""

    def __init__(self, eval_env, eval_episodes: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_episodes = eval_episodes

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Log current params from the environment
        env = self.eval_env
        if hasattr(env, "envs"):
            inner = env.envs[0]
        elif hasattr(env, "env"):
            inner = env.env
        else:
            inner = env

        # Navigate through wrappers
        while hasattr(inner, "env"):
            inner = inner.env

        params = inner.params.as_dict()
        for key, val in params.items():
            self.logger.record(f"params/{key}", val)

        # Run eval episodes
        rewards = []
        for _ in range(self.eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                ep_reward += reward
                done = terminated or truncated
            rewards.append(ep_reward)

        self.logger.record("eval/mean_reward", float(np.mean(rewards)))
        self.logger.record("eval/std_reward", float(np.std(rewards)))


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str = None) -> dict:
    """Load YAML config or return defaults."""
    defaults = {
        "env": {
            "num_eval_trials": 5,
            "target_range_deg": 120.0,
            "max_aim_time": 0.6,
            "hit_threshold_deg": 1.0,
            "max_episode_steps": 50,
            "action_scale": 0.05,
        },
        "rl": {
            "algorithm": "PPO",
            "total_timesteps": 50_000,
            "n_steps": 512,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "policy": "MlpPolicy",
            "policy_kwargs": {"net_arch": [128, 128]},
        },
        "logging": {
            "tensorboard_dir": "runs/rl_auto_tune/tb",
            "save_dir": "runs/rl_auto_tune",
            "log_interval": 10,
            "eval_freq": 2000,
            "eval_episodes": 20,
        },
        "output": {
            "model_path": "runs/rl_auto_tune/ppo_aimenv",
            "params_json": "runs/rl_auto_tune/learned_params.json",
        },
    }
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # Merge loaded config into defaults
        for section in defaults:
            if section in cfg:
                defaults[section].update(cfg[section])
    return defaults


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(cfg: dict):
    env_cfg = AimEnvConfig(
        num_eval_trials=cfg["env"]["num_eval_trials"],
        target_range_deg=cfg["env"]["target_range_deg"],
        max_aim_time=cfg["env"]["max_aim_time"],
        hit_threshold_deg=cfg["env"]["hit_threshold_deg"],
        max_episode_steps=cfg["env"]["max_episode_steps"],
        action_scale=cfg["env"]["action_scale"],
    )
    env = AimEnv(config=env_cfg)
    return gym.wrappers.TimeLimit(env, max_episode_steps=cfg["env"]["max_episode_steps"])


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(config_path: str = None):
    cfg = load_config(config_path)
    rl_cfg = cfg["rl"]
    log_cfg = cfg["logging"]
    out_cfg = cfg["output"]

    os.makedirs(log_cfg["save_dir"], exist_ok=True)

    train_env = make_env(cfg)
    eval_env = make_env(cfg)

    # Select algorithm
    algo_name = rl_cfg["algorithm"].upper()
    policy_kwargs = dict(rl_cfg.get("policy_kwargs", {}))

    common_kwargs = dict(
        policy=rl_cfg["policy"],
        env=train_env,
        verbose=1,
        tensorboard_log=log_cfg["tensorboard_dir"],
        learning_rate=rl_cfg["learning_rate"],
        gamma=rl_cfg["gamma"],
        batch_size=rl_cfg["batch_size"],
        policy_kwargs=policy_kwargs,
    )

    if algo_name == "PPO":
        model = PPO(
            **common_kwargs,
            n_steps=rl_cfg["n_steps"],
            gae_lambda=rl_cfg["gae_lambda"],
            clip_range=rl_cfg["clip_range"],
            ent_coef=rl_cfg["ent_coef"],
        )
    elif algo_name == "SAC":
        model = SAC(**common_kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    # Callback
    callback = ParamLoggingCallback(
        eval_env=eval_env,
        eval_episodes=log_cfg["eval_episodes"],
    )

    print(f"Training {algo_name} for {rl_cfg['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=rl_cfg["total_timesteps"],
        callback=callback,
        log_interval=log_cfg["log_interval"],
    )

    # Save model
    model.save(out_cfg["model_path"])
    print(f"Model saved to {out_cfg['model_path']}")

    # Extract and save learned params
    inner_env = train_env
    while hasattr(inner_env, "env"):
        inner_env = inner_env.env
    learned_params = inner_env.params.as_dict()

    with open(out_cfg["params_json"], "w", encoding="utf-8") as f:
        json.dump(learned_params, f, indent=2)
    print(f"Learned params: {learned_params}")
    print(f"Params saved to {out_cfg['params_json']}")

    return learned_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P1: RL auto-tuning of accel params")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    args = parser.parse_args()
    main(config_path=args.config)