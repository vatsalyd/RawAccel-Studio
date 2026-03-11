"""
Evaluate a trained RL model and compare against baseline DEFAULT_ACCEL_PARAMS.

Usage:
    python -m experiments.rl_auto_tune.evaluate_rl
    python -m experiments.rl_auto_tune.evaluate_rl --model runs/rl_auto_tune/ppo_aimenv.zip
"""
import argparse
import json
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC

from env.aim_sim.env_core import AimEnv, AimEnvConfig, SimpleAimTask, HumanLikeControllerConfig
from models.curve_param_config import DEFAULT_ACCEL_PARAMS, AccelParams


def evaluate_params(params: AccelParams, num_trials: int = 100, seed: int = 0):
    """Run many aim tasks with the given params and return aggregate metrics."""
    rng = np.random.default_rng(seed)
    ctrl = HumanLikeControllerConfig()
    hits, times, errors, overshoots = [], [], [], []

    for _ in range(num_trials):
        task = SimpleAimTask(rng=rng)
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


def run_rl_episodes(model, env, num_episodes: int = 20):
    """Run the RL agent for a number of episodes and return mean reward."""
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    return float(np.mean(rewards)), float(np.std(rewards))


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL-learned accel params vs baseline")
    parser.add_argument("--model", type=str, default="runs/rl_auto_tune/ppo_aimenv.zip")
    parser.add_argument("--params-json", type=str, default="runs/rl_auto_tune/learned_params.json")
    parser.add_argument("--num-trials", type=int, default=200)
    parser.add_argument("--output", type=str, default="runs/rl_auto_tune/eval_results.json")
    args = parser.parse_args()

    # Load learned params
    if os.path.isfile(args.params_json):
        with open(args.params_json, "r") as f:
            p = json.load(f)
        learned = AccelParams(**p)
    else:
        print(f"Warning: {args.params_json} not found. Run training first.")
        learned = DEFAULT_ACCEL_PARAMS

    # Evaluate both
    print("Evaluating DEFAULT_ACCEL_PARAMS (baseline)...")
    baseline_metrics = evaluate_params(DEFAULT_ACCEL_PARAMS, num_trials=args.num_trials)

    print("Evaluating RL-learned params...")
    learned_metrics = evaluate_params(learned, num_trials=args.num_trials)

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"{'Metric':<20} {'Baseline':>15} {'RL-Learned':>15}")
    print("-" * 60)
    for key in baseline_metrics:
        print(f"{key:<20} {baseline_metrics[key]:>15.4f} {learned_metrics[key]:>15.4f}")
    print("=" * 60)

    # Print param comparison
    print(f"\n{'Parameter':<12} {'Baseline':>12} {'RL-Learned':>12}")
    print("-" * 40)
    bd = DEFAULT_ACCEL_PARAMS.as_dict()
    ld = learned.as_dict()
    for key in bd:
        print(f"{key:<12} {bd[key]:>12.6f} {ld[key]:>12.6f}")

    # Save results
    results = {
        "baseline_params": bd,
        "learned_params": ld,
        "baseline_metrics": baseline_metrics,
        "learned_metrics": learned_metrics,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
