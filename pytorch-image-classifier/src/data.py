"""Data loading for the digit-classification experiment.

Uses scikit-learn's bundled 8x8 handwritten digits dataset, so the project runs
fully offline. Swap `load_digit_tensors` for your own loader to use other data.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def load_digit_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    digits = load_digits()
    images = digits.images.astype(np.float32) / 16.0  # pixel values are 0..16
    images = (images - 0.5) / 0.5  # scale to roughly [-1, 1]
    x = torch.from_numpy(images).unsqueeze(1)  # (N, 1, 8, 8)
    y = torch.from_numpy(digits.target.astype(np.int64))
    return x, y


def make_splits(seed: int = 42, val_size: float = 0.15, test_size: float = 0.15):
    """Stratified train/val/test split returned as tensors."""
    x, y = load_digit_tensors()
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=y.numpy(), random_state=seed
    )
    rel_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=rel_val, stratify=y.numpy()[train_idx], random_state=seed
    )
    return {
        "train": (x[train_idx], y[train_idx]),
        "val": (x[val_idx], y[val_idx]),
        "test": (x[test_idx], y[test_idx]),
    }


def make_loaders(batch_size: int = 64, seed: int = 42):
    splits = make_splits(seed=seed)
    gen = torch.Generator().manual_seed(seed)
    loaders = {}
    for name, (x, y) in splits.items():
        loaders[name] = DataLoader(
            TensorDataset(x, y),
            batch_size=batch_size,
            shuffle=(name == "train"),
            generator=gen if name == "train" else None,
        )
    return loaders
