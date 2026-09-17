"""Model 2/10: Linear SVM on cached CLIP embeddings.

LinearSVC has no predict_proba (only decision_function), and both the AUC
reporting and the app need a real probability for the confidence %, so wrap
it in CalibratedClassifierCV (sigmoid/Platt scaling, 3-fold CV on the
training set only -- val/OOD stay untouched by calibration fitting).
"""
import pickle
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

import common_eval as ce

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMB_PATH = PROJECT_ROOT / "data" / "manifests" / "embeddings.npz"

KEY = "linsvm"
LABEL = "Linear SVM (CLIP)"


def main():
    df, X_all = ce.load_embeddings(EMB_PATH)
    y_all = (df["cls"] == "fake").astype(int).values
    masks = ce.build_eval_masks(df)

    X_train, y_train = X_all[masks["train"]], y_all[masks["train"]]
    w_train = ce.inverse_freq_weights(df[masks["train"]])
    print(f"training on {len(y_train)} rows ({int(y_train.sum())} fake, {int(len(y_train) - y_train.sum())} real)", flush=True)

    base = LinearSVC(class_weight="balanced", max_iter=10000)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(X_train, y_train, sample_weight=w_train)

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
    }, results, sorted(df.loc[masks["train"], "generator"].unique().tolist()))


if __name__ == "__main__":
    main()
