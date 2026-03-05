import os

import gymnasium as gym
from stable_baselines3 import PPO

from env.aim_sim.env_core import AimEnv


def make_env():
    return AimEnv()


def main():
    os.makedirs("runs/rl_auto_tune", exist_ok=True)
    env = gym.wrappers.TimeLimit(make_env(), max_episode_steps=50)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="runs/rl_auto_tune/tb",
        n_steps=512,
        batch_size=64,
    )

    model.learn(total_timesteps=50_000)
    model.save("runs/rl_auto_tune/ppo_aimenv")

    obs, _ = env.reset()
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()