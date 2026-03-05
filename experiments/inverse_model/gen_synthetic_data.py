import os
from typing import List

import numpy as np
import torch

from env.aim_sim.env_core import SimpleAimTask, HumanLikeControllerConfig
from models.curve_param_config import AccelParams, clamp_params, apply_accel


def sample_random_params() -> AccelParams:
    return clamp_params(
        AccelParams(
            k1=float(10 ** np.random.uniform(-4, -1)),
            a=float(np.random.uniform(0.6, 1.6)),
            k2=float(10 ** np.random.uniform(-4, -1)),
            b=float(np.random.uniform(0.6, 1.8)),
            v0=float(np.random.uniform(100, 2000)),
            sens_min=float(np.random.uniform(0.1, 0.8)),
            sens_max=float(np.random.uniform(2.0, 10.0)),
        )
    )


def simulate_sequence(params: AccelParams, steps: int = 60):
    task = SimpleAimTask()
    ctrl = HumanLikeControllerConfig()
    dt = 0.01

    seq = []
    for _ in range(steps):
        error = task.target_yaw - task.view_yaw
        controller_output = ctrl.base_gain * error + np.random.randn() * ctrl.noise_std * abs(error)
        mouse_dx_counts = controller_output * 100.0

        mouse_dx = torch.tensor(mouse_dx_counts, dtype=torch.float32)
        dt_t = torch.tensor(dt, dtype=torch.float32)
        view_delta = apply_accel(mouse_dx, dt_t, params)
        task.view_yaw += float(view_delta.item())
        done, _ = task.step_with_params(params, ctrl, dt=dt)

        seq.append([mouse_dx_counts, dt, float(view_delta.item())])
        if done:
            break

    return np.array(seq, dtype=np.float32)


def main():
    out_dir = os.path.join("data", "sim_inverse")
    os.makedirs(out_dir, exist_ok=True)

    num_curves = 200
    sequences: List[np.ndarray] = []
    params_list = []

    for i in range(num_curves):
        p = sample_random_params()
        seq = simulate_sequence(p)
        sequences.append(seq)
        params_list.append(
            [p.k1, p.a, p.k2, p.b, p.v0]
        )
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/{num_curves} curves")

    np.savez_compressed(
        os.path.join(out_dir, "inverse_data.npz"),
        sequences=sequences,
        params=np.array(params_list, dtype=np.float32),
    )
    print("Saved to", os.path.join(out_dir, "inverse_data.npz"))


if __name__ == "__main__":
    main()