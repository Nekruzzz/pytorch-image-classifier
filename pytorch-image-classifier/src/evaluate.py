"""Evaluate a trained model on the held-out test split.

Writes a per-class report and a confusion matrix (CSV) next to the weights.

Usage:
    python -m src.evaluate --run runs/baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .data import make_splits
from .model import SmallCNN


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    return model(x).argmax(dim=1).numpy()


def evaluate(run_dir: str | Path, seed: int = 42) -> dict:
    run = Path(run_dir)
    model = SmallCNN()
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))

    x_test, y_test = make_splits(seed=seed)["test"]
    y_pred = predict(model, x_test)
    y_true = y_test.numpy()

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(run / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    (run / "classification_report.json").write_text(json.dumps(report, indent=2))

    print(classification_report(y_true, y_pred, zero_division=0))
    print("confusion matrix saved to", run / "confusion_matrix.csv")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/baseline")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    evaluate(a.run, a.seed)
