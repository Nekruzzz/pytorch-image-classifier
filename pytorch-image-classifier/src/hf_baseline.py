"""Optional baseline: classify an image with a pretrained Hugging Face model.

This compares the small CNN trained from scratch against a pretrained Vision
Transformer. It needs internet access to download the model from the Hugging
Face Hub the first time it runs.

Usage:
    python -m src.hf_baseline path/to/image.jpg --top-k 3
"""
from __future__ import annotations

import argparse


def classify(image_path: str, model_name: str = "google/vit-base-patch16-224", top_k: int = 3):
    from transformers import pipeline  # imported lazily so the rest works offline

    classifier = pipeline("image-classification", model=model_name)
    return classifier(image_path, top_k=top_k)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--model", default="google/vit-base-patch16-224")
    p.add_argument("--top-k", type=int, default=3)
    a = p.parse_args()
    for r in classify(a.image, a.model, a.top_k):
        print(f"{r['label']:<30} {r['score']:.3f}")
