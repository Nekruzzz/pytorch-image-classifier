# PyTorch Image Classifier

A small, fully reproducible image-classification project: a **PyTorch** CNN trained on handwritten digits, with a proper train / validation / test split, a saved metrics file, and an evaluation step that writes a per-class report and a confusion matrix. An optional **Hugging Face** script compares it with a pretrained Vision Transformer.

It is a learning project that shows an end-to-end workflow (data, model, training, evaluation, tests, CI) rather than a state-of-the-art result.

## What it uses

| Piece | Technology |
|---|---|
| Model and training loop | **PyTorch** |
| Data and metrics | scikit-learn (bundled 8x8 digits dataset, classification report, confusion matrix) |
| Pretrained baseline (optional) | **Hugging Face** `transformers` (`google/vit-base-patch16-224`) |
| Tests and CI | pytest, GitHub Actions |

## Result

Trained for 20 epochs with the default settings (seed 42, Adam, learning rate 1e-3), using the best epoch on the validation split:

| Split | Accuracy |
|---|---|
| Validation | 96.7% |
| Test (270 held-out images) | 96.3% |

The weakest classes on the test split are digits 1 and 8 (recall 0.85 each). Numbers can shift slightly across PyTorch versions and hardware.

## Quick start

```bash
pip install -r requirements.txt
python -m src.train --epochs 20 --out runs/baseline     # writes model.pt and metrics.json
python -m src.evaluate --run runs/baseline              # report + confusion_matrix.csv
python -m pytest -q
```

Optional pretrained baseline (needs internet the first time and the `transformers` package):

```bash
python -m src.hf_baseline path/to/image.jpg --top-k 3
```

## What is tested

`python -m pytest -q` runs 3 tests: the data splits cover all 1,797 images with the right shape, the model outputs the right shape, and an 8-epoch training run reaches more than 90% test accuracy and writes the model, metrics and confusion matrix files.

## Limitations

- The 8x8 digits dataset is tiny and easy. It is used so the project runs offline in seconds, not to show a hard vision problem.
- `src/hf_baseline.py` is included but was **not run** while building this project, because the build environment could not download models from the Hugging Face Hub.
- Only one seed and one architecture were evaluated. There is no hyperparameter search or error analysis beyond the per-class report.

## Project layout

```
src/data.py          dataset loading and stratified splits
src/model.py         the CNN
src/train.py         training loop, best-epoch selection, metrics.json
src/evaluate.py      classification report and confusion matrix
src/hf_baseline.py   optional Hugging Face baseline
tests/               pytest suite
```

## About how this was built

This project was written with AI assistance (Claude) as a learning exercise, then run and tested locally.
