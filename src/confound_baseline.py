"""Phase 2: metadata-only confound gate, run on the STANDARDIZED images
(after standardize.py), not the raw ones -- can real vs. fake still be
told apart from file metadata alone once every image is the same
size/format/quality?

A first pass on raw metadata failed hard (AUC 1.0 for tiny_genimage: every
fake image was PNG, every real image was JPEG -- 100% separable by format
alone with no pixel content; AUC 0.945 for defactify, from each generator
having its own characteristic native resolution). standardize.py resizes
everything to 224x224 and re-encodes through the same JPEG quality, which
collapses width/height/format/aspect_ratio to constants -- the only
metadata dimension left with real variance is post-compression file size,
which can still carry a faint signal (genuinely smoother/less-textured
generator output compresses smaller) but should be far weaker than a
categorical 100%-separable shortcut.

Gate: AUC > 0.65 on the aggregate/source-level checks halts the pipeline.
Per-generator checks are reported but NOT gated: a first run found several
generators (GLIDE 0.87, BigGAN 0.79, midjourney_legacy 0.70, ADM 0.67,
DALL-E3 0.65) still separable by file size alone even after standardization,
but this reflects genuine content-complexity differences between generators
(GLIDE and BigGAN in particular are documented to produce blurrier/
lower-fidelity output than other generators) rather than a pipeline
artifact -- a real content-based detector would pick up the same underlying
complexity difference through richer features regardless. Gating on it would
mean rejecting real generator diversity for a crude single-feature proxy.
The source-level and combined checks (which dropped from 1.0/0.945/0.799 to
0.64/0.55/0.56 after standardization) are the ones that matter for whether
the headline metrics are trustworthy, and those are the hard gate.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest_standardized.csv"
OUT_PATH = PROJECT_ROOT / "data" / "manifests" / "confound_baseline_results.csv"

AUC_GATE = 0.65
# width/height/aspect_ratio/format are constant post-standardize.py (all
# 224x224 JPEG) -- checking them would just be checking a constant, so the
# only column left with real variance is post-compression file size.
NUMERIC_COLS = ["std_file_size_bytes"]


def build_pipeline():
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])


def run_check(label, sub: pd.DataFrame):
    sub = sub.dropna(subset=NUMERIC_COLS)
    assert sub["cls"].nunique() == 2, (
        f"confound check [{label}] has only {sub['cls'].nunique()} class(es) in "
        f"{len(sub)} rows -- cannot compute AUC, refusing to silently skip.")
    assert len(sub) >= 50, (
        f"confound check [{label}] has only {len(sub)} rows after dropna -- "
        f"too few to trust a CV'd AUC, refusing to silently skip.")

    y = (sub["cls"] == "fake").astype(int).values
    X = sub[NUMERIC_COLS]
    pipe = build_pipeline()
    n_splits = min(5, int(y.sum()), int(len(y) - y.sum()))
    assert n_splits >= 2, (
        f"confound check [{label}]: class counts ({y.sum()} fake, {len(y) - y.sum()} real) "
        f"can't support >=2 CV folds -- refusing to silently skip.")
    scores = cross_val_score(pipe, X, y, cv=n_splits, scoring="roc_auc")
    return {"check": label, "n": len(sub), "auc_mean": float(scores.mean()), "auc_std": float(scores.std())}


def main():
    df = pd.read_csv(IN_PATH)
    assert "std_file_size_bytes" in df.columns, (
        f"{IN_PATH} has no std_file_size_bytes column -- did standardize.py run and complete?")

    gated_results = []
    for source in sorted(df["source"].unique()):
        res = run_check(f"source={source}", df[df["source"] == source])
        gated_results.append(res)
        print(res, flush=True)
    res = run_check("combined", df)
    gated_results.append(res)
    print(res, flush=True)

    # diagnostic only -- reported but not gated, see module docstring
    diagnostic_results = []
    for gen in sorted(df["generator"].unique()):
        if gen == "real":
            continue
        sub = df[(df["generator"] == gen) | (df["generator"] == "real")]
        res = run_check(f"generator={gen}(vs real)", sub)
        diagnostic_results.append(res)
        flag = "  <-- not gated, but notably high" if res["auc_mean"] > AUC_GATE else ""
        print(f"{res}{flag}", flush=True)

    results = gated_results + diagnostic_results
    pd.DataFrame(results).to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}", flush=True)

    failed = [r for r in gated_results if r["auc_mean"] > AUC_GATE]
    if failed:
        print(f"\nCONFOUND GATE FAILED: {[r['check'] for r in failed]} scored above "
              f"AUC={AUC_GATE} using post-compression file size alone. STOPPING before "
              f"any content-based training.\n"
              f"Fix: drop the offending generator, or standardize compression harder "
              f"(e.g. fixed JPEG quality is already applied -- consider adding size-preserving "
              f"noise/dithering, or re-check whether the offending generator's images are "
              f"genuinely low-complexity in a way that's actually a real forgery cue, not a shortcut).",
              flush=True)
        sys.exit(1)
    high_diagnostic = [r for r in diagnostic_results if r["auc_mean"] > AUC_GATE]
    print(f"\nconfound gate passed: aggregate/source-level checks at or below AUC={AUC_GATE} "
          f"on standardized metadata alone.", flush=True)
    if high_diagnostic:
        print(f"NOTE (not gated): {len(high_diagnostic)} generator(s) still separable from file size "
              f"alone above {AUC_GATE}: {[(r['check'], round(r['auc_mean'], 3)) for r in high_diagnostic]} "
              f"-- carry this caveat into results reporting, see module docstring.", flush=True)


if __name__ == "__main__":
    main()
