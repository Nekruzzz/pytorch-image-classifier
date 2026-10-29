"""Train the CNN and write the weights plus a metrics file.

Usage:
    python -m src.train --epochs 20 --out runs/baseline
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import make_loaders
from .model import SmallCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def accuracy(model: nn.Module, loader) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        correct += (model(xb).argmax(dim=1) == yb).sum().item()
        total += len(yb)
    return correct / max(total, 1)


def train(epochs: int = 20, lr: float = 1e-3, batch_size: int = 64, seed: int = 42,
          out_dir: str | Path = "runs/baseline") -> dict:
    set_seed(seed)
    loaders = make_loaders(batch_size=batch_size, seed=seed)
    model = SmallCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_val, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loaders["train"]:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(yb)
        train_loss = running / len(loaders["train"].dataset)
        val_acc = accuracy(model, loaders["val"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:02d}  loss {train_loss:.4f}  val_acc {val_acc:.4f}")

    model.load_state_dict(best_state)
    test_acc = accuracy(model, loaders["test"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    metrics = {"best_val_acc": best_val, "test_acc": test_acc, "history": history,
               "config": {"epochs": epochs, "lr": lr, "batch_size": batch_size, "seed": seed}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"best val acc {best_val:.4f}  |  test acc {test_acc:.4f}  ->  {out}")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs/baseline")
    a = p.parse_args()
    train(a.epochs, a.lr, a.batch_size, a.seed, a.out)
