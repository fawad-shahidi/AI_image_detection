"""Gradio web UI for the AI-generated image detector. Visual style matches
the Pashto fake-news project's Flask app (design mockup approved before
this was built: cream background, orange gradient primary actions, teal
checkmarks, card layout, dark-mode toggle) -- see theme.py for the theme
object and CSS. Model registry/factory pattern (model_registry.py's
MODEL_SPECS/get_predictor) mirrors that same Pashto project's predict.py.

Layout uses gr.Group() for the card sections and variant="primary"/
"secondary" on buttons rather than custom elem_classes -- see theme.py's
docstring for why (a first pass styling raw Button/Checkbox DOM via global
CSS mostly didn't render; Gradio's Theme API is the reliable mechanism for
native-component appearance).

Deviations from the mockup, forced by Gradio's native component model
rather than a stylistic choice: checkboxes render checkbox-then-label (not
label-then-checkmark); the inline AUC figure next to each Deep-tier model
is plain text, not the mockup's monospace face, since a Checkbox label
can't mix two fonts in one string.
"""
import json

import gradio as gr

import model_registry as mr
from predict import CAVEAT
from theme import CSS, HEAD_HTML, THEME, TOGGLE_THEME_JS

DEFAULT_KEYS = {"clip_ft", "mlp_deep", "dinov2_mlp"}

TIERS = [
    ("Simple · CLIP embeddings", ["logreg", "linsvm", "rf"]),
    ("Middle · CLIP embeddings", ["lgbm", "knn", "mlp_small"]),
    ("Deep", ["mlp_deep", "clip_ft", "resnet50_ft", "dinov2_mlp"]),
]
ALL_KEYS = [key for _, keys in TIERS for key in keys]


def _load_ood_aggregate(key: str):
    config_path = mr.MODEL_SPECS[key]["config_path"]
    if not config_path.is_file():
        return None
    with config_path.open() as fh:
        config = json.load(fh)
    return config.get("results_auc", {}).get("ood_aggregate")


AUC_LOOKUP = {key: _load_ood_aggregate(key) for key in ALL_KEYS}


def checkbox_label(key: str) -> str:
    label = mr.MODEL_SPECS[key]["label"].rsplit(" [", 1)[0]  # strip "[Tier]" -- tier is now a section header
    auc = AUC_LOOKUP[key]
    return f"{label}   ·   {auc:.3f}" if auc is not None else label


def model_tier(key: str) -> str:
    label = mr.MODEL_SPECS[key]["label"]
    return label.rsplit("[", 1)[-1].rstrip("]") if "[" in label else ""


def model_short_name(key: str) -> str:
    return mr.MODEL_SPECS[key]["label"].rsplit(" [", 1)[0]


def render_results_html(results: list[dict]) -> str:
    n_fake = sum(1 for r in results if r["label"] == "AI-generated")
    n_real = len(results) - n_fake
    if n_fake > n_real:
        majority_label = "AI-generated"
    elif n_real > n_fake:
        majority_label = "Real"
    else:
        avg_all = sum(r["proba_fake"] for r in results) / len(results)
        majority_label = "AI-generated" if avg_all >= 0.5 else "Real"
    verdict_class = "fake" if majority_label == "AI-generated" else "real"
    winning = [r for r in results if r["label"] == majority_label]
    avg_conf = sum(r["confidence"] for r in winning) / len(winning)
    n_agree = len(winning)

    rows = "".join(
        f'<div class="result-row">'
        f'<div class="rmeta"><div class="rname">{r["name"]}</div><div class="rtier">{r["tier"]}</div></div>'
        f'<span class="chip {"fake" if r["label"] == "AI-generated" else "real"}">{r["label"]}</span>'
        f'<span class="rconf">{r["confidence"] * 100:.0f}%</span>'
        f'</div>'
        for r in results
    )
    verdict = (
        f'<div class="verdict {verdict_class}">'
        f'<div><div class="vlabel">Majority verdict</div><div class="vtitle">{majority_label}</div></div>'
        f'<div class="vstat"><div class="big">{n_agree} / {len(results)}</div>'
        f'<div class="small">models agree · {avg_conf * 100:.0f}% avg. confidence</div></div>'
        f'</div>'
    )
    return f'<div class="results-html">{verdict}{rows}</div>'


def analyze(image, *checkbox_values, progress=gr.Progress()):
    if image is None:
        return '<div class="results-html"><p>Please upload an image first.</p></div>'
    selected = [key for key, checked in zip(ALL_KEYS, checkbox_values) if checked]
    if not selected:
        return '<div class="results-html"><p>Please select at least one model.</p></div>'

    results = []
    for i, key in enumerate(selected):
        progress((i + 0.5) / len(selected), desc=f"Running {model_short_name(key)}...")
        predictor = mr.get_predictor(key)
        proba_fake = predictor.predict_proba_single(image)
        label = "AI-generated" if proba_fake >= 0.5 else "Real"
        confidence = proba_fake if label == "AI-generated" else 1 - proba_fake
        results.append({
            "name": model_short_name(key), "tier": model_tier(key),
            "label": label, "confidence": confidence, "proba_fake": proba_fake,
        })
    return render_results_html(results)


with gr.Blocks(title="AI Image Detector") as demo:
    with gr.Row():
        with gr.Row():
            gr.HTML('<div class="badge">\U0001F6E1️</div>')
            with gr.Column():
                gr.Markdown("AI-Generated Image Detector", elem_classes=["brand-title"])
                gr.Markdown("Upload a photo, pick which models weigh in", elem_classes=["brand-sub"])
        theme_btn = gr.Button("\U0001F319", variant="secondary", elem_classes=["theme-toggle-btn"])
    theme_btn.click(fn=None, js=TOGGLE_THEME_JS)

    with gr.Group():
        gr.Markdown("**1 · Upload image**", elem_classes=["section-label"])
        image_in = gr.Image(label=None, show_label=False, type="pil", sources=["upload"])

        with gr.Row():
            gr.Markdown("**2 · Select models**", elem_classes=["section-label"])
            select_all_btn = gr.Button("Select all", variant="secondary", size="sm")
            default_btn = gr.Button("Default", variant="primary", size="sm")

        checkbox_by_key = {}
        for tier_name, keys in TIERS:
            gr.Markdown(f'<p class="tier-label">{tier_name}</p>')
            with gr.Row():
                with gr.Column():
                    for j, key in enumerate(keys):
                        if j % 2 == 0:
                            checkbox_by_key[key] = gr.Checkbox(
                                label=checkbox_label(key), value=(key in DEFAULT_KEYS), container=False)
                with gr.Column():
                    for j, key in enumerate(keys):
                        if j % 2 == 1:
                            checkbox_by_key[key] = gr.Checkbox(
                                label=checkbox_label(key), value=(key in DEFAULT_KEYS), container=False)
        checkboxes = [checkbox_by_key[key] for key in ALL_KEYS]

        select_all_btn.click(fn=lambda: [True] * len(ALL_KEYS), outputs=checkboxes)
        default_btn.click(fn=lambda: [key in DEFAULT_KEYS for key in ALL_KEYS], outputs=checkboxes)

        analyze_btn = gr.Button("Analyze image", variant="primary", size="lg")

    with gr.Group():
        gr.Markdown("**3 · Results**", elem_classes=["section-label"])
        results_out = gr.HTML()

    gr.Markdown(f"*{CAVEAT}*", elem_classes=["note-text"])

    analyze_btn.click(fn=analyze, inputs=[image_in] + checkboxes, outputs=results_out)

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, head=HEAD_HTML)

# How to run: `python src/app.py`, then open http://localhost:7860 in a browser.
