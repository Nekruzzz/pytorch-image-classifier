import json

import torch

from src.data import make_splits
from src.evaluate import evaluate
from src.model import SmallCNN
from src.train import train


def test_splits_are_disjoint_and_cover_dataset():
    s = make_splits(seed=1)
    total = sum(len(y) for _, y in s.values())
    assert total == 1797  # size of scikit-learn digits
    assert all(x.shape[1:] == (1, 8, 8) for x, _ in s.values())


def test_model_output_shape():
    out = SmallCNN()(torch.zeros(4, 1, 8, 8))
    assert out.shape == (4, 10)


def test_training_learns_and_evaluation_writes_reports(tmp_path):
    metrics = train(epochs=8, out_dir=tmp_path)
    assert metrics["test_acc"] > 0.9
    assert (tmp_path / "model.pt").exists()

    report = evaluate(tmp_path)
    assert report["accuracy"] > 0.9
    assert (tmp_path / "confusion_matrix.csv").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["config"]["epochs"] == 8
