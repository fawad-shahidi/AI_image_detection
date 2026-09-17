"""Phase 1c: standardize every image to a common size/format/quality before
the confound gate is trusted.

The first confound_baseline.py run failed hard on raw metadata (AUC 1.0 for
tiny_genimage, 0.945 for defactify) because of two trivial shortcuts:
  - tiny_genimage: ALL fake images are PNG, ALL real images are JPEG --
    100% separable by file format alone, no pixel content needed.
  - defactify: each generator outputs at its own characteristic resolution
    (fakes mean width 704, std 306; real mean 578, std 91) -- resolution
    alone is a usable, non-generalizing shortcut.

Both are metadata artifacts a real deployed classifier could latch onto
via the pixel content itself (PNG has no JPEG blocking artifacts; a
generator's fixed native resolution survives as texture/frequency
characteristics even after CLIP's internal resize) rather than genuine
forgery signal. Resizing every image to the same fixed size and
re-encoding through the same JPEG quality erases both distinctions before
anything touches the encoder.
"""
import time
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest_deduped.csv"
OUT_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest_standardized.csv"
STD_DIR = PROJECT_ROOT / "data" / "standardized"

TARGET_SIZE = (224, 224)  # CLIP ViT-B/32's native input resolution
JPEG_QUALITY = 90


def standardize_one(raw_path: Path, out_path: Path) -> tuple[int, int, int]:
    with Image.open(raw_path) as img:
        img = img.convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="JPEG", quality=JPEG_QUALITY)
    size_bytes = out_path.stat().st_size
    return TARGET_SIZE[0], TARGET_SIZE[1], size_bytes


def main():
    df = pd.read_csv(IN_PATH)
    rows = df.to_dict("records")
    n = len(rows)
    print(f"standardizing {n} images to {TARGET_SIZE} JPEG q={JPEG_QUALITY}...", flush=True)

    t0 = time.time()
    errors = []
    for i, r in enumerate(rows):
        out_path = STD_DIR / r["source"] / r["generator"] / f"{r['image_id']}.jpg"
        try:
            w, h, size_bytes = standardize_one(PROJECT_ROOT / r["raw_path"], out_path)
        except Exception as e:
            errors.append((r["image_id"], repr(e)))
            r["standardized_path"] = ""
            r["std_width"] = r["std_height"] = r["std_file_size_bytes"] = None
            continue
        r["standardized_path"] = str(out_path.relative_to(PROJECT_ROOT))
        r["std_width"], r["std_height"], r["std_file_size_bytes"] = w, h, size_bytes
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{n}, {elapsed:.1f}s elapsed ({(i + 1) / elapsed:.1f} img/sec)", flush=True)

    elapsed = time.time() - t0
    print(f"done: {elapsed:.1f}s ({n / elapsed:.1f} img/sec, {len(errors)} errors)", flush=True)
    if errors:
        print(f"  failed image_ids (first 20): {errors[:20]}", flush=True)

    # FAIL-OPEN GUARD: same rationale as dedup.py/embed.py -- an image that
    # fails to standardize silently drops out downstream with no trace
    # unless bounded and asserted here.
    error_rate = len(errors) / n
    assert error_rate < 0.02, (
        f"{len(errors)}/{n} images ({error_rate:.1%}) failed to standardize -- "
        f"too high to be stray corruption, refusing to silently proceed")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
