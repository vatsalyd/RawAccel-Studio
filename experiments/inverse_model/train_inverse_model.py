import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from models.inverse_curve_net import InverseCurveNet


class InverseDataset(Dataset):
    def __init__(self, path: str, max_len: int = 80):
        data = np.load(path, allow_pickle=True)
        self.sequences = data["sequences"]
        self.params = data["params"]
        self.max_len = max_len

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        pad_len = self.max_len - len(seq)
        if pad_len > 0:
            pad = np.zeros((pad_len, seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
        else:
            seq = seq[: self.max_len]
        x = torch.from_numpy(seq)
        y = torch.from_numpy(self.params[idx])
        return x, y


def main():
    data_path = os.path.join("data", "sim_inverse", "inverse_data.npz")
    dataset = InverseDataset(data_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InverseCurveNet(input_dim=3, num_params=5).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(20):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

    os.makedirs("runs/inverse_model", exist_ok=True)
    torch.save(model.state_dict(), "runs/inverse_model/inverse_curve_net.pt")


if __name__ == "__main__":
    main()