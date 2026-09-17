"""Command-line inference: classify an image as AI-generated or real.

Usage: python src/predict.py <image_path> [--model KEY]
List available model keys: python src/predict.py --list-models
"""
import argparse
from pathlib import Path

from PIL import Image

import model_registry as mr

CAVEAT = ("Note: trained on legacy/mid-generation generators (ADM, BigGAN, GLIDE, SD1.4, "
          "SD1.5, VQDM, Wukong); held-out accuracy on modern generators (SDXL, SD3, DALL-E 3) "
          "is meaningfully lower than the ID number -- see data/model_comparison.md, "
          "and treat confidence on generators not in either list as unverified.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image_path", nargs="?")
    p.add_argument("--model", default="logreg", choices=list(mr.MODEL_SPECS.keys()))
    p.add_argument("--list-models", action="store_true")
    args = p.parse_args()

    if args.list_models:
        for key, spec in mr.MODEL_SPECS.items():
            print(f"  {key:14s} {spec['label']}")
        return
    if not args.image_path:
        p.error("image_path is required unless --list-models is passed")

    image_path = Path(args.image_path)
    if not image_path.is_file():
        raise SystemExit(f"ERROR: no such file: {image_path}")

    predictor = mr.get_predictor(args.model)
    with Image.open(image_path) as img:
        proba_fake = predictor.predict_proba_single(img)

    threshold = 0.5
    if proba_fake >= threshold:
        label, confidence = "AI-GENERATED", proba_fake
    else:
        label, confidence = "REAL", 1 - proba_fake

    print(f"[{mr.MODEL_SPECS[args.model]['label']}] {label} (confidence: {confidence * 100:.0f}%)")
    print(CAVEAT)


if __name__ == "__main__":
    main()
