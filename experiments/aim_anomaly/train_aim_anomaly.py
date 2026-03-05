import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from env.aim_sim.env_core import SimpleAimTask, HumanLikeControllerConfig
from models.curve_param_config import DEFAULT_ACCEL_PARAMS, apply_accel
from models.aim_anomaly_net import AimAnomalyNet


def simulate_normal_seq(steps: int = 60):
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
        view_delta = apply_accel(mouse_dx, dt_t, DEFAULT_ACCEL_PARAMS)
        task.view_yaw += float(view_delta.item())
        seq.append([mouse_dx_counts, dt, float(view_delta.item()), float(error)])
        done, _ = task.step_with_params(DEFAULT_ACCEL_PARAMS, ctrl, dt=dt)
        if done:
            break
    return np.array(seq, dtype=np.float32)


def simulate_assisted_seq(steps: int = 60):
    task = SimpleAimTask()
    dt = 0.01
    seq = []
    for i in range(steps):
        error = task.target_yaw - task.view_yaw
        if i < 3:
            mouse_dx_counts = 0.0
        else:
            mouse_dx_counts = error * 500.0
        mouse_dx = torch.tensor(mouse_dx_counts, dtype=torch.float32)
        dt_t = torch.tensor(dt, dtype=torch.float32)
        view_delta = apply_accel(mouse_dx, dt_t, DEFAULT_ACCEL_PARAMS)
        task.view_yaw += float(view_delta.item())
        seq.append([mouse_dx_counts, dt, float(view_delta.item()), float(error)])
        done, _ = task.step_with_params(DEFAULT_ACCEL_PARAMS, HumanLikeControllerConfig(), dt=dt)
        if done:
            break
    return np.array(seq, dtype=np.float32)


class AnomalyDataset(Dataset):
    def __init__(self, n_per_class: int = 200, max_len: int = 80):
        normal_seqs = [simulate_normal_seq() for _ in range(n_per_class)]
        assisted_seqs = [simulate_assisted_seq() for _ in range(n_per_class)]
        self.seqs = normal_seqs + assisted_seqs
        self.labels = [0] * n_per_class + [1] * n_per_class
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        pad_len = self.max_len - len(seq)
        if pad_len > 0:
            pad = np.zeros((pad_len, seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
        else:
            seq = seq[: self.max_len]
        x = torch.from_numpy(seq)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def main():
    dataset = AnomalyDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AimAnomalyNet(input_dim=4).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(15):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.numel()
        print(f"Epoch {epoch+1}: loss={total_loss/len(dataset):.4f}, acc={correct/total:.3f}")

    os.makedirs("runs/aim_anomaly", exist_ok=True)
    torch.save(model.state_dict(), "runs/aim_anomaly/aim_anomaly_net.pt")


if __name__ == "__main__":
    main()