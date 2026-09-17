"""Shared PyTorch Dataset for the two fine-tuning scripts (CLIP partial FT,
ResNet50 FT) -- both read the same standardized_path images but need
different normalization, so the transform is injected rather than baked in.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_SIZE = (224, 224)  # matches standardize.py's TARGET_SIZE

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _normalize(img: Image.Image, mean, std) -> torch.Tensor:
    # Training/eval images from standardize.py are already exactly 224x224
    # (this resize is then a no-op check), but model_registry.py's
    # CLIPFineTunedPredictor/ResNet50FineTunedPredictor call this directly
    # on arbitrary-sized live-uploaded images -- without this, CLIP's
    # position embeddings (sized for exactly 224x224 -> 49 patches + 1 CLS)
    # mismatch against whatever patch count the real image size produces
    # and crash. Bug found via a real browser upload; local testing had
    # missed it because every test image happened to reuse pre-standardized
    # 224x224 files from data/standardized/.
    if img.size != TARGET_SIZE:
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)


class StandardizedImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, normalization: str, weights: np.ndarray | None = None):
        assert normalization in ("clip", "imagenet")
        self.paths = df["standardized_path"].tolist()
        self.labels = (df["cls"] == "fake").astype("float32").tolist()
        self.weights = weights.astype("float32").tolist() if weights is not None else None
        self.mean, self.std = (CLIP_MEAN, CLIP_STD) if normalization == "clip" else (IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(PROJECT_ROOT / self.paths[idx]) as img:
            img = img.convert("RGB")
            x = _normalize(img, self.mean, self.std)
        if self.weights is not None:
            return x, self.labels[idx], self.weights[idx]
        return x, self.labels[idx]
