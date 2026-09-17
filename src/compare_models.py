"""Builds the side-by-side comparison table across all 10 models. The Pashto
project's equivalent (model_comparison.ipynb, evaluate.py) was never
actually populated -- this one loads every trained model's config.json and
writes a real table.
"""
import json
from pathlib import Path

import model_registry as mr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "data" / "model_comparison.md"


def load_all_results():
    rows = []
    missing = []
    for key, spec in mr.MODEL_SPECS.items():
        if not spec["config_path"].is_file():
            missing.append(key)
            continue
        with spec["config_path"].open() as fh:
            config = json.load(fh)
        results = config.get("results_auc", {})
        rows.append({
            "key": key,
            "label": spec["label"],
            "id_val": results.get("id_val"),
            "ood_aggregate": results.get("ood_aggregate"),
            "ood_DALLE3": results.get("ood_DALLE3"),
        })
    return rows, missing


def format_auc(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def main():
    rows, missing = load_all_results()
    if missing:
        print(f"NOTE: {len(missing)} model(s) not yet trained, skipped: {missing}", flush=True)

    rows.sort(key=lambda r: (r["ood_aggregate"] is None, -(r["ood_aggregate"] or 0)))

    lines = ["| Model | ID val AUC | OOD aggregate AUC | DALL-E 3 AUC |",
             "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['label']} | {format_auc(r['id_val'])} | "
                      f"{format_auc(r['ood_aggregate'])} | {format_auc(r['ood_DALLE3'])} |")
    table = "\n".join(lines)

    print(table, flush=True)
    OUT_PATH.write_text(table + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_PATH}", flush=True)

    if missing:
        raise SystemExit(f"{len(missing)} model(s) missing from the comparison -- "
                          f"train them before treating this table as final: {missing}")


if __name__ == "__main__":
    main()
