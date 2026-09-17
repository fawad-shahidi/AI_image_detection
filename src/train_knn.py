"""Model 5/10: k-NN on cached CLIP embeddings, cosine distance (more
appropriate than Euclidean for CLIP's feature space). n_neighbors swept on
val AUC, best kept -- not cross-validated on train, since we want the same
val split every other model is scored on to pick the winner too."""
import pickle
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier

import common_eval as ce

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMB_PATH = PROJECT_ROOT / "data" / "manifests" / "embeddings.npz"

KEY = "knn"
LABEL = "k-NN (CLIP)"
K_CANDIDATES = [15, 31, 51, 101]


def main():
    df, X_all = ce.load_embeddings(EMB_PATH)
    y_all = (df["cls"] == "fake").astype(int).values
    masks = ce.build_eval_masks(df)

    X_train, y_train = X_all[masks["train"]], y_all[masks["train"]]
    X_val, y_val = X_all[masks["val"]], y_all[masks["val"]]
    print(f"training on {len(y_train)} rows ({int(y_train.sum())} fake, {int(len(y_train) - y_train.sum())} real)", flush=True)

    best_k, best_auc, best_model = None, -1.0, None
    for k in K_CANDIDATES:
        model = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
        model.fit(X_train, y_train)  # kNN has no sample_weight; class balance handled by cosine + large train pool
        auc = ce.eval_auc(lambda X: model.predict_proba(X)[:, 1], X_val, y_val, f"val, k={k}")
        if auc is not None and auc > best_auc:
            best_k, best_auc, best_model = k, auc, model
    print(f"selected k={best_k} (val AUC={best_auc:.4f})", flush=True)
    model = best_model

    proba_fn = lambda X: model.predict_proba(X)[:, 1]
    results = ce.run_all_evals(proba_fn, df, X_all, y_all, masks)

    ce.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ce.MODELS_DIR / f"{KEY}.pkl"
    with model_path.open("wb") as fh:
        pickle.dump(model, fh)
    ce.save_results(KEY, LABEL, "sklearn_embed", {
        "clip_model_hf_id": "openai/clip-vit-base-patch32",
        "embeddings_source": "clip",
        "artifact_path": str(model_path.relative_to(PROJECT_ROOT)),
        "selected_k": best_k,
    }, results, sorted(df.loc[masks["train"], "generator"].unique().tolist()))


if __name__ == "__main__":
    main()
