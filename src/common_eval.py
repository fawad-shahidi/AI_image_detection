"""Shared data-loading/masking/eval logic for all 10 models, factored out of
the original train_head.py (which keeps working unchanged, still training
the Logistic Regression baseline against these exact same masks). Every
model script in the 10-model comparison imports this instead of
reimplementing split logic -- the whole point of the comparison is that all
10 see identical train/val/OOD/Midjourney data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest_split.csv"
MODELS_DIR = PROJECT_ROOT / "data" / "models"

OOD_GENERATORS = ["SD21", "SDXL", "SD3", "DALLE3"]
MIDJOURNEY_GENERATORS = ["midjourney_legacy", "midjourney_v6"]


def load_embeddings(emb_path: Path):
    """Merge manifest_split.csv onto an embeddings .npz (image_ids + embeddings
    keys, same schema for every embeddings file -- CLIP's or DINOv2's)."""
    df = pd.read_csv(MANIFEST_PATH)
    emb = np.load(emb_path)
    emb_df = pd.DataFrame({"image_id": emb["image_ids"]})
    emb_df["_row"] = np.arange(len(emb_df))
    merged = df.merge(emb_df, on="image_id", how="inner")
    assert len(merged) > 0, f"no rows survived the manifest/{emb_path.name} merge -- check image_id consistency"
    X = emb["embeddings"][merged["_row"].values]
    return merged.drop(columns=["_row"]).reset_index(drop=True), X


def inverse_freq_weights(sub: pd.DataFrame) -> np.ndarray:
    counts = sub.groupby(["source", "generator"])["image_id"].transform("count")
    w = 1.0 / counts
    return (w / w.mean()).values  # normalize to mean 1 so regularization scale is unaffected


def combined_sample_weights(df_train: pd.DataFrame, y_train: np.ndarray) -> np.ndarray:
    """inverse-frequency (source,generator) weight * standard class-balance
    weight -- same balancing philosophy as class_weight='balanced' +
    sample_weight combined for the sklearn models, in a form usable as a
    manual per-sample loss weight (BCEWithLogitsLoss has no class_weight arg).
    Shared by train_mlp.py, train_clip_finetuned.py, train_resnet50_finetuned.py."""
    freq_w = inverse_freq_weights(df_train)
    n, n_pos = len(y_train), y_train.sum()
    n_neg = n - n_pos
    class_w = np.where(y_train == 1, n / (2 * n_pos), n / (2 * n_neg))
    combined = freq_w * class_w
    return combined / combined.mean()


def build_eval_masks(df: pd.DataFrame) -> dict:
    """One dict of boolean masks, computed once, shared by every model's eval
    pass -- train_head.py's OOD/Midjourney per-generator mask logic exactly,
    just precomputed instead of inlined per script."""
    masks = {
        "train": (df["split"] == "train").values,
        "val": (df["split"] == "val").values,
        "ood_aggregate": (df["split"] == "ood_test").values,
    }
    ood_real_mask = (df["split"] == "ood_test") & (df["generator"] == "real")
    for gen in OOD_GENERATORS:
        masks[f"ood_{gen}"] = (((df["split"] == "ood_test") & (df["generator"] == gen)) | ood_real_mask).values

    mj_real_mask = df["in_midjourney_eval_real"] == 1
    for gen in MIDJOURNEY_GENERATORS:
        masks[f"midjourney_{gen}"] = (((df["split"] == "midjourney_eval") & (df["generator"] == gen)) | mj_real_mask).values
    return masks


def eval_auc(proba_fn, X, y, label: str):
    """proba_fn: callable X -> P(fake), so sklearn's `.predict_proba(X)[:,1]`
    and a PyTorch model's own inference wrapper plug in the same way."""
    if len(set(y)) < 2:
        print(f"  [{label}] skipped: only one class present ({len(y)} rows)", flush=True)
        return None
    proba = proba_fn(X)
    auc = roc_auc_score(y, proba)
    print(f"  [{label}] AUC={auc:.4f} (n={len(y)}, fake={int(y.sum())}, real={int(len(y) - y.sum())})", flush=True)
    return auc


def run_all_evals(proba_fn, df, X, y, masks) -> dict:
    """Runs the full standard eval suite (ID val, OOD aggregate + per-generator,
    Midjourney per-vintage) and returns the results dict train_head.py's
    config schema expects."""
    results = {}
    results["id_val"] = eval_auc(proba_fn, X[masks["val"]], y[masks["val"]], "ID val")
    results["ood_aggregate"] = eval_auc(proba_fn, X[masks["ood_aggregate"]], y[masks["ood_aggregate"]], "OOD aggregate")
    for gen in OOD_GENERATORS:
        results[f"ood_{gen}"] = eval_auc(proba_fn, X[masks[f"ood_{gen}"]], y[masks[f"ood_{gen}"]], f"OOD [{gen}]")
    for gen in MIDJOURNEY_GENERATORS:
        results[f"midjourney_{gen}"] = eval_auc(
            proba_fn, X[masks[f"midjourney_{gen}"]], y[masks[f"midjourney_{gen}"]], f"Midjourney [{gen}]")
    return results


def save_results(key: str, label: str, kind: str, config_extra: dict, results_auc: dict, train_generators: list):
    """Writes data/models/<key>_config.json in the same schema model_config.json
    already uses, plus model_key/label/kind fields the registry needs."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "model_key": key,
        "label": label,
        "kind": kind,
        "decision_threshold": 0.5,
        "train_generators": train_generators,
        "ood_generators": OOD_GENERATORS,
        "midjourney_generators_excluded_from_train_and_ood": MIDJOURNEY_GENERATORS,
        "results_auc": results_auc,
        **config_extra,
    }
    out_path = MODELS_DIR / f"{key}_config.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print(f"wrote {out_path}", flush=True)
    return out_path
