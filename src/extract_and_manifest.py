"""Phase 1a: extract images from the two HF parquet datasets to disk and
build the unified manifest.

Reads raw pyarrow (not the `datasets` library's own loader) so the original
encoded image bytes are written to disk untouched -- no re-encode/re-compress
pass. This matters for confound_baseline.py: if every image got re-saved
through PIL uniformly, real compression differences between sources would be
erased before the confound gate ever got to check for them.

Generator naming keeps Midjourney's two vintages distinct
(midjourney_legacy vs midjourney_v6) rather than collapsing them to
"midjourney" -- make_splits.py needs to route both into a single
family-disjoint reporting bucket, never split across train/OOD.
"""
import csv
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTRACT_DIR = PROJECT_ROOT / "data" / "extracted"
OUT_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest.csv"

# Tiny-GenImage: columns image/label/generator, BOTH label and generator are
# HF ClassLabel int64 columns (verified against a landed parquet file --
# `generator` is NOT a plain string column despite how the dataset card
# reads). The int->name mapping is read from each file's own schema metadata
# (schema["huggingface"].info.features.generator.names) rather than
# hardcoded, so a schema change trips the assertion below instead of
# silently mislabeling generators.
TINY_GENIMAGE_NAME_MAP = {
    "Real": "real", "ADM": "ADM", "BigGAN": "BigGAN", "GLIDE": "GLIDE",
    "Midjourney": "midjourney_legacy", "SD14": "SD14", "SD15": "SD15",
    "VQDM": "VQDM", "Wukong": "Wukong",
}


def _classlabel_names(table, column: str) -> list[str]:
    meta = table.schema.metadata
    assert meta and b"huggingface" in meta, f"no huggingface schema metadata found -- can't decode {column!r} ClassLabel ints"
    hf_info = json.loads(meta[b"huggingface"])
    names = hf_info["info"]["features"][column]["names"]
    assert isinstance(names, list) and names, f"unexpected ClassLabel names for {column!r}: {names!r}"
    return names

# Defactify: Label_B 0-5 = Real/SD2.1/SDXL/SD3/DALL-E3/Midjourney-v6 (Label_A
# is redundant with Label_B==0, not used).
DEFACTIFY_LABEL_B_MAP = {
    0: "real", 1: "SD21", 2: "SDXL", 3: "SD3", 4: "DALLE3", 5: "midjourney_v6",
}


def _image_field_to_bytes(val):
    """HF Image-feature columns land as a struct {bytes, path} (rarely just
    path, if the dataset points at external files) when read via plain
    pyarrow. Handle both; raise loudly on anything else so a schema mismatch
    is caught on file 1, not silently mis-extracted across 130K rows."""
    if isinstance(val, dict):
        if val.get("bytes") is not None:
            return val["bytes"]
        if val.get("path"):
            raise NotImplementedError(
                f"image struct has no inline bytes, only path={val['path']!r} -- "
                f"this dataset stores images externally, extraction logic needs updating")
    raise NotImplementedError(f"unrecognized image column value type: {type(val)}")


def _write_image(raw_bytes: bytes, out_path_no_ext: Path) -> tuple[Path, str, int, int]:
    img = Image.open(io.BytesIO(raw_bytes))
    fmt = (img.format or "JPEG").upper()
    ext = ".jpg" if fmt == "JPEG" else f".{fmt.lower()}"
    out_path = out_path_no_ext.with_suffix(ext)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw_bytes)
    return out_path, fmt, img.width, img.height


def extract_tiny_genimage(rows_out):
    src_dir = RAW_DIR / "tiny_genimage" / "data"
    files = sorted(src_dir.glob("*.parquet"))
    assert files, f"no parquet files found under {src_dir} -- did the download finish?"
    n = 0
    for pf in files:
        table = pq.read_table(pf, columns=["image", "label", "generator"])
        cols = table.column_names
        assert set(cols) == {"image", "label", "generator"}, (
            f"{pf.name}: unexpected columns {cols} -- schema assumption wrong, fix TINY_GENIMAGE_NAME_MAP logic")
        generator_names = _classlabel_names(table, "generator")
        split_tag = "train" if "train" in pf.name else "validation"
        for i, row in enumerate(table.to_pylist()):
            gen_idx = row["generator"]
            assert 0 <= gen_idx < len(generator_names), (
                f"{pf.name} row {i}: generator index {gen_idx} out of range for {generator_names!r}")
            gen_raw = generator_names[gen_idx]
            gen = TINY_GENIMAGE_NAME_MAP.get(gen_raw)
            assert gen is not None, f"{pf.name} row {i}: unrecognized generator value {gen_raw!r}"
            cls = "real" if gen == "real" else "fake"
            img_id = f"tgi_{pf.stem}_{i}"
            raw_bytes = _image_field_to_bytes(row["image"])
            out_no_ext = EXTRACT_DIR / "tiny_genimage" / gen / img_id
            out_path, fmt, w, h = _write_image(raw_bytes, out_no_ext)
            rows_out.append({
                "image_id": img_id, "source": "tiny_genimage", "generator": gen, "cls": cls,
                "raw_path": str(out_path.relative_to(PROJECT_ROOT)), "file_size_bytes": len(raw_bytes),
                "format": fmt, "width": w, "height": h, "hf_split": split_tag,
            })
            n += 1
        print(f"  [tiny_genimage] {pf.name}: {len(table)} rows extracted (running total {n})", flush=True)
    return n


def extract_defactify(rows_out):
    src_dir = RAW_DIR / "defactify" / "data"
    if not src_dir.is_dir():
        # some HF repos put parquet files at repo root instead of data/
        src_dir = RAW_DIR / "defactify"
    files = sorted(src_dir.rglob("*.parquet"))
    assert files, f"no parquet files found under {RAW_DIR / 'defactify'} -- did the download finish?"
    n = 0
    for pf in files:
        table = pq.read_table(pf)
        cols = set(table.column_names)
        assert {"Image", "Label_B"}.issubset(cols) or {"image", "Label_B"}.issubset(cols), (
            f"{pf.name}: unexpected columns {table.column_names} -- schema assumption wrong")
        image_col = "Image" if "Image" in cols else "image"
        split_tag = pf.stem.split("-")[0]  # train/validation/test prefix in filename
        for i, row in enumerate(table.to_pylist()):
            label_b = row["Label_B"]
            gen = DEFACTIFY_LABEL_B_MAP.get(label_b)
            assert gen is not None, f"{pf.name} row {i}: unrecognized Label_B value {label_b!r}"
            cls = "real" if gen == "real" else "fake"
            img_id = f"dft_{pf.stem}_{i}"
            raw_bytes = _image_field_to_bytes(row[image_col])
            out_no_ext = EXTRACT_DIR / "defactify" / gen / img_id
            out_path, fmt, w, h = _write_image(raw_bytes, out_no_ext)
            rows_out.append({
                "image_id": img_id, "source": "defactify", "generator": gen, "cls": cls,
                "raw_path": str(out_path.relative_to(PROJECT_ROOT)), "file_size_bytes": len(raw_bytes),
                "format": fmt, "width": w, "height": h, "hf_split": split_tag,
            })
            n += 1
        print(f"  [defactify] {pf.name}: {len(table)} rows extracted (running total {n})", flush=True)
    return n


def main():
    rows = []
    print("extracting tiny_genimage...", flush=True)
    n1 = extract_tiny_genimage(rows)
    print("extracting defactify...", flush=True)
    n2 = extract_defactify(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "source", "generator", "cls", "raw_path",
                  "file_size_bytes", "format", "width", "height", "hf_split"]
    with OUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {OUT_PATH} (tiny_genimage={n1}, defactify={n2})", flush=True)

    by_key = {}
    for r in rows:
        key = (r["source"], r["cls"], r["generator"])
        by_key[key] = by_key.get(key, 0) + 1
    for key in sorted(by_key):
        print(f"  {key[0]:14s} {key[1]:5s} {key[2]:16s}: {by_key[key]}")


if __name__ == "__main__":
    main()
