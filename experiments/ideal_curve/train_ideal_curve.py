import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from models.ideal_curve_net import IdealCurveNet


class SyntheticProfileDataset(Dataset):
    def __init__(self, n: int = 5000, input_dim: int = 8, num_params: int = 5, seed: int = 123):
        rng = np.random.default_rng(seed)
        self.features = rng.uniform(0, 1, size=(n, input_dim)).astype(np.float32)
        w = rng.normal(size=(input_dim, num_params)).astype(np.float32)
        self.params = self.features @ w + rng.normal(scale=0.1, size=(n, num_params)).astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])
        y = torch.from_numpy(self.params[idx])
        return x, y


def main():
    dataset = SyntheticProfileDataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IdealCurveNet(input_dim=8, num_params=5).to(device)
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
        print(f"Epoch {epoch+1}: loss={total_loss/len(dataset):.4f}")

    os.makedirs("runs/ideal_curve", exist_ok=True)
    torch.save(model.state_dict(), "runs/ideal_curve/ideal_curve_net.pt")


if __name__ == "__main__":
    main()