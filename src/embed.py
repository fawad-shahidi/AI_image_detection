"""Phase 4: frozen CLIP ViT-B/32 embedding cache. One 512-dim embedding per
image (no frame-pooling needed -- unlike the video project, there's only one
frame here)."""
import os
import time
from pathlib import Path

# transformers auto-detects the installed TensorFlow and tries to import
# TF-specific image-processing code on load; that import chain hits a
# Windows Smart App Control DLL block on this machine (same class of issue
# as the video project's `av` pin). We only use the torch backend, so tell
# transformers not to bother with TF at all.
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest_split.csv"
OUT_PATH = PROJECT_ROOT / "data" / "manifests" / "embeddings.npz"

BATCH_SIZE = 64
MODEL_NAME = "openai/clip-vit-base-patch32"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    df = pd.read_csv(MANIFEST_PATH)
    rows = df.to_dict("records")
    n = len(rows)

    print(f"loading {MODEL_NAME}...", flush=True)
    t_load0 = time.time()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    print(f"model loaded in {time.time() - t_load0:.1f}s", flush=True)

    image_ids, embeddings, errors = [], [], []
    decode_time_total, gpu_time_total = 0.0, 0.0
    t0 = time.time()
    batch_imgs, batch_ids = [], []

    def flush_batch():
        nonlocal batch_imgs, batch_ids, gpu_time_total
        if not batch_imgs:
            return
        tg0 = time.time()
        with torch.no_grad():
            inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            feats = model.get_image_features(**inputs)
        gpu_time_total += time.time() - tg0
        feats = feats.cpu().numpy()
        for img_id, feat in zip(batch_ids, feats):
            image_ids.append(img_id)
            embeddings.append(feat)
        batch_imgs, batch_ids = [], []

    for i, row in enumerate(rows):
        td0 = time.time()
        try:
            with Image.open(PROJECT_ROOT / row["standardized_path"]) as img:
                img = img.convert("RGB")
                img.load()
        except Exception as e:
            errors.append((row["image_id"], repr(e)))
            decode_time_total += time.time() - td0
            continue
        decode_time_total += time.time() - td0
        batch_imgs.append(img)
        batch_ids.append(row["image_id"])
        if len(batch_ids) >= BATCH_SIZE:
            flush_batch()
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{n}, {elapsed:.1f}s elapsed ({(i + 1) / elapsed:.1f} img/sec) "
                  f"[decode={decode_time_total:.1f}s gpu={gpu_time_total:.1f}s]", flush=True)

    flush_batch()
    total_time = time.time() - t0
    print(f"\ndone: {total_time:.1f}s for {n} images ({n / total_time:.1f} img/sec, "
          f"{len(errors)} errors)", flush=True)
    if errors:
        print(f"  failed image_ids (first 20): {errors[:20]}", flush=True)

    # FAIL-OPEN GUARD: same rationale as embed.py in the video project -- an
    # image that fails to decode here just vanishes from training/eval with
    # no trace unless bounded and asserted.
    error_rate = len(errors) / n
    assert error_rate < 0.02, (
        f"{len(errors)}/{n} images ({error_rate:.1%}) failed embedding -- "
        f"too high to be stray corruption, refusing to silently proceed")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, image_ids=np.array(image_ids), embeddings=np.stack(embeddings))
    print(f"wrote {OUT_PATH} ({len(image_ids)} embeddings, dim={embeddings[0].shape[0]})", flush=True)


if __name__ == "__main__":
    main()
