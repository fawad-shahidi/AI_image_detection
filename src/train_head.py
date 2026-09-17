"""Phase 5: train the logistic regression head on cached CLIP embeddings,
then report AUC per split -- ID (val), headline OOD (ood_test, aggregate +
per-generator), and the Midjourney reporting-only bucket kept separate from
both (see make_splits.py's docstring for why).

Sample weighting: inverse frequency by (source, generator), same fix the
video project used for its generator/class imbalance (Tiny-GenImage's real
class is ~1:8 vs any single fake generator after its own uniform per-class
subsampling).
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest_split.csv"
EMB_PATH = PROJECT_ROOT / "data" / "manifests" / "embeddings.npz"
MODEL_OUT = PROJECT_ROOT / "data" / "model_head.pkl"
CONFIG_OUT = PROJECT_ROOT / "data" / "model_config.json"

OOD_GENERATORS = ["SD21", "SDXL", "SD3", "DALLE3"]
MIDJOURNEY_GENERATORS = ["midjourney_legacy", "midjourney_v6"]


def load_merged():
    df = pd.read_csv(MANIFEST_PATH)
    emb = np.load(EMB_PATH)
    emb_df = pd.DataFrame({"image_id": emb["image_ids"]})
    emb_df["_row"] = np.arange(len(emb_df))
    merged = df.merge(emb_df, on="image_id", how="inner")
    assert len(merged) > 0, "no rows survived the manifest/embeddings merge -- check image_id consistency"
    X = emb["embeddings"][merged["_row"].values]
    return merged.drop(columns=["_row"]), X


def inverse_freq_weights(sub: pd.DataFrame) -> np.ndarray:
    counts = sub.groupby(["source", "generator"])["image_id"].transform("count")
    w = 1.0 / counts
    return (w / w.mean()).values  # normalize to mean 1 so LR's regularization scale is unaffected


def eval_auc(model, X, y, label):
    if len(set(y)) < 2:
        print(f"  [{label}] skipped: only one class present ({len(y)} rows)", flush=True)
        return None
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    print(f"  [{label}] AUC={auc:.4f} (n={len(y)}, fake={int(y.sum())}, real={int(len(y) - y.sum())})", flush=True)
    return auc


def main():
    df, X_all = load_merged()
    y_all = (df["cls"] == "fake").astype(int).values

    train_mask = df["split"] == "train"
    X_train, y_train = X_all[train_mask.values], y_all[train_mask.values]
    w_train = inverse_freq_weights(df[train_mask])
    print(f"training on {len(y_train)} rows ({int(y_train.sum())} fake, {int(len(y_train) - y_train.sum())} real)", flush=True)

    model = LogisticRegression(max_iter=5000, class_weight="balanced")
    model.fit(X_train, y_train, sample_weight=w_train)

    results = {}

    val_mask = (df["split"] == "val").values
    results["id_val"] = eval_auc(model, X_all[val_mask], y_all[val_mask], "ID val")

    ood_mask = (df["split"] == "ood_test").values
    results["ood_aggregate"] = eval_auc(model, X_all[ood_mask], y_all[ood_mask], "OOD aggregate")

    ood_real_mask = (df["split"] == "ood_test") & (df["generator"] == "real")
    for gen in OOD_GENERATORS:
        gen_mask = ((df["split"] == "ood_test") & (df["generator"] == gen)) | ood_real_mask
        gen_mask = gen_mask.values
        results[f"ood_{gen}"] = eval_auc(model, X_all[gen_mask], y_all[gen_mask], f"OOD [{gen}]")

    mj_real_mask = (df["in_midjourney_eval_real"] == 1)
    for gen in MIDJOURNEY_GENERATORS:
        gen_mask = ((df["split"] == "midjourney_eval") & (df["generator"] == gen)) | mj_real_mask
        gen_mask = gen_mask.values
        results[f"midjourney_{gen}"] = eval_auc(model, X_all[gen_mask], y_all[gen_mask], f"Midjourney [{gen}]")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_OUT.open("wb") as fh:
        pickle.dump(model, fh)
    config = {
        "clip_model_hf_id": "openai/clip-vit-base-patch32",
        "decision_threshold": 0.5,
        "embedding_dim": int(X_all.shape[1]),
        "train_generators": sorted(df.loc[train_mask, "generator"].unique().tolist()),
        "ood_generators": OOD_GENERATORS,
        "midjourney_generators_excluded_from_train_and_ood": MIDJOURNEY_GENERATORS,
        "results_auc": results,
    }
    with CONFIG_OUT.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print(f"\nwrote {MODEL_OUT} and {CONFIG_OUT}", flush=True)


if __name__ == "__main__":
    main()
