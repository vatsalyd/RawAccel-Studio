"""
Shared dataset and data-loading utilities used across all experiments.
"""
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def window_sequence(
    seq: np.ndarray,
    window_len: int,
    stride: int = 1,
) -> np.ndarray:
    """
    Slice a (T, F) sequence into overlapping windows of shape (N, window_len, F).

    Parameters
    ----------
    seq : np.ndarray of shape (T, F)
    window_len : number of time steps per window
    stride : step size between windows (default 1)

    Returns
    -------
    np.ndarray of shape (N, window_len, F)
    """
    T, F = seq.shape
    starts = range(0, T - window_len + 1, stride)
    return np.stack([seq[s : s + window_len] for s in starts], axis=0)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def compute_stats(
    sequences: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std across a list of (T_i, F) arrays."""
    all_data = np.concatenate(sequences, axis=0)  # (sum(T_i), F)
    mean = all_data.mean(axis=0).astype(np.float32)
    std = all_data.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0  # avoid division by zero
    return mean, std


def normalize_features(
    seq: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Z-score normalise a (T, F) or (..., F) array."""
    return ((seq - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Generic padded-sequence dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """
    PyTorch Dataset that wraps variable-length sequences into fixed-length
    tensors by padding or truncating.

    Parameters
    ----------
    sequences : list of np.ndarray, each of shape (T_i, F)
    targets : np.ndarray of shape (N,) or (N, P)
    max_len : fixed sequence length (pad / truncate)
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        targets: np.ndarray,
        max_len: int = 80,
    ):
        assert len(sequences) == len(targets)
        self.sequences = sequences
        self.targets = targets.astype(np.float32)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx].astype(np.float32)
        T, F = seq.shape

        if T < self.max_len:
            pad = np.zeros((self.max_len - T, F), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
        else:
            seq = seq[: self.max_len]

        x = torch.from_numpy(seq)
        y = torch.from_numpy(self.targets[idx])
        return x, y


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def create_dataloaders(
    dataset: Dataset,
    val_fraction: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Split a Dataset into train / val DataLoaders."""
    n = len(dataset)
    n_val = int(n * val_fraction)
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
