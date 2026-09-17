"""Model registry + predictor factory for the 10-model comparison, mirrors
the Pashto fake-news project's MODEL_SPECS dict + get_predictor() dispatch
pattern.

VRAM discipline: only ONE heavy GPU-resident object is kept at a time --
the frozen CLIP encoder, the frozen DINOv2 encoder, or a full fine-tuned
model (CLIP-FT / ResNet50-FT). Switching the app's dropdown to a model that
needs a different one evicts the previous one first (`del`, then
`torch.cuda.empty_cache()`) rather than accumulating all of them, since on
a 4GB card holding more than one resident risks OOM. sklearn/LightGBM heads
and the tiny PyTorch MLP heads are cheap and stay cached indefinitely.
"""
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("USE_TF", "0")  # see embed.py -- avoids a TF import that hits Smart App Control

import torch
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

from image_dataset import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, _normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_HF_ID = "openai/clip-vit-base-patch32"
DINOV2_HF_ID = "facebook/dinov2-base"

# key -> (label, kind, artifact_path, config_path). Order = display order in
# the app dropdown (Simple -> Middle -> Deep, matching the plan's tiers).
MODEL_SPECS = {
    "logreg":      {"label": "Logistic Regression (CLIP) [Simple]", "kind": "sklearn_embed",
                     "artifact_path": DATA_DIR / "model_head.pkl", "config_path": DATA_DIR / "model_config.json"},
    "linsvm":      {"label": "Linear SVM (CLIP) [Simple]", "kind": "sklearn_embed",
                     "artifact_path": MODELS_DIR / "linsvm.pkl", "config_path": MODELS_DIR / "linsvm_config.json"},
    "rf":          {"label": "Random Forest (CLIP) [Simple]", "kind": "sklearn_embed",
                     "artifact_path": MODELS_DIR / "rf.pkl", "config_path": MODELS_DIR / "rf_config.json"},
    "lgbm":        {"label": "LightGBM (CLIP) [Middle]", "kind": "sklearn_embed",
                     "artifact_path": MODELS_DIR / "lgbm.pkl", "config_path": MODELS_DIR / "lgbm_config.json"},
    "knn":         {"label": "k-NN (CLIP) [Middle]", "kind": "sklearn_embed",
                     "artifact_path": MODELS_DIR / "knn.pkl", "config_path": MODELS_DIR / "knn_config.json"},
    "mlp_small":   {"label": "Small MLP (CLIP) [Middle]", "kind": "pytorch_mlp_embed",
                     "artifact_path": MODELS_DIR / "mlp_small.pt", "config_path": MODELS_DIR / "mlp_small_config.json"},
    "mlp_deep":    {"label": "Deep MLP (CLIP) [Deep]", "kind": "pytorch_mlp_embed",
                     "artifact_path": MODELS_DIR / "mlp_deep.pt", "config_path": MODELS_DIR / "mlp_deep_config.json"},
    "clip_ft":     {"label": "CLIP (partial fine-tune) [Deep]", "kind": "clip_finetuned",
                     "artifact_path": MODELS_DIR / "clip_ft.pt", "config_path": MODELS_DIR / "clip_ft_config.json"},
    "resnet50_ft": {"label": "ResNet50 (fine-tuned) [Deep]", "kind": "resnet50_finetuned",
                     "artifact_path": MODELS_DIR / "resnet50_ft.pt", "config_path": MODELS_DIR / "resnet50_ft_config.json"},
    "dinov2_mlp":  {"label": "DINOv2 (frozen) + MLP [Deep]", "kind": "pytorch_mlp_embed",
                     "artifact_path": MODELS_DIR / "dinov2_mlp.pt", "config_path": MODELS_DIR / "dinov2_mlp_config.json"},
}

_backbone_cache = {"kind": None, "model": None, "processor": None}
_head_cache = {}  # model_key -> loaded head object; cheap, never evicted


def _evict_backbone():
    if _backbone_cache["model"] is not None:
        del _backbone_cache["model"]
        _backbone_cache["model"] = None
        _backbone_cache["processor"] = None
        _backbone_cache["kind"] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _get_backbone(kind: str):
    if _backbone_cache["kind"] == kind:
        return _backbone_cache["model"], _backbone_cache["processor"]
    _evict_backbone()
    if kind == "clip":
        processor = CLIPProcessor.from_pretrained(CLIP_HF_ID)
        model = CLIPModel.from_pretrained(CLIP_HF_ID).to(DEVICE).eval()
    elif kind == "dinov2":
        processor = AutoImageProcessor.from_pretrained(DINOV2_HF_ID)
        model = AutoModel.from_pretrained(DINOV2_HF_ID).to(DEVICE).eval()
    elif kind == "clip_ft":
        from train_clip_finetuned import CLIPFineTuneModel
        model = CLIPFineTuneModel()
        model.load_state_dict(torch.load(MODELS_DIR / "clip_ft.pt", map_location=DEVICE))
        model.to(DEVICE).eval()
        processor = None
    elif kind == "resnet50_ft":
        from train_resnet50_finetuned import ResNet50FineTuneModel
        model = ResNet50FineTuneModel()
        model.load_state_dict(torch.load(MODELS_DIR / "resnet50_ft.pt", map_location=DEVICE))
        model.to(DEVICE).eval()
        processor = None
    else:
        raise ValueError(f"unknown backbone kind: {kind!r}")
    _backbone_cache.update(kind=kind, model=model, processor=processor)
    return model, processor


def _embed(pil_image, encoder_kind: str):
    model, processor = _get_backbone(encoder_kind)
    img = pil_image.convert("RGB")
    with torch.no_grad():
        inputs = processor(images=[img], return_tensors="pt").to(DEVICE)
        if encoder_kind == "clip":
            feat = model.get_image_features(**inputs).cpu().numpy()
        else:  # dinov2 -- CLS token, no trained pooler (see embed_dinov2.py)
            feat = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
    return feat


class SklearnEmbedPredictor:
    def __init__(self, key, spec, config):
        if key not in _head_cache:
            with spec["artifact_path"].open("rb") as fh:
                _head_cache[key] = pickle.load(fh)
        self.clf = _head_cache[key]
        self.encoder_kind = config.get("embeddings_source", "clip")

    def predict_proba_single(self, pil_image) -> float:
        feat = _embed(pil_image, self.encoder_kind)
        return float(self.clf.predict_proba(feat)[0, 1])


class PyTorchMLPPredictor:
    def __init__(self, key, spec, config):
        self.encoder_kind = config["embeddings_source"]
        if key not in _head_cache:
            from train_mlp import MLPHead
            mlp = MLPHead(config["input_dim"], config["hidden_dims"], config["dropout"])
            mlp.load_state_dict(torch.load(spec["artifact_path"], map_location=DEVICE))
            mlp.to(DEVICE).eval()
            _head_cache[key] = mlp
        self.mlp = _head_cache[key]

    def predict_proba_single(self, pil_image) -> float:
        feat = _embed(pil_image, self.encoder_kind)
        with torch.no_grad():
            logit = self.mlp(torch.from_numpy(feat).float().to(DEVICE))
        return float(torch.sigmoid(logit).cpu().item())


class CLIPFineTunedPredictor:
    def predict_proba_single(self, pil_image) -> float:
        model, _ = _get_backbone("clip_ft")
        x = _normalize(pil_image.convert("RGB"), CLIP_MEAN, CLIP_STD).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = model(x)
        return float(torch.sigmoid(logit).cpu().item())


class ResNet50FineTunedPredictor:
    def predict_proba_single(self, pil_image) -> float:
        model, _ = _get_backbone("resnet50_ft")
        x = _normalize(pil_image.convert("RGB"), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = model(x)
        return float(torch.sigmoid(logit).cpu().item())


def get_model_options():
    """[(label, key), ...] in MODEL_SPECS order, for a UI dropdown."""
    return [(spec["label"], key) for key, spec in MODEL_SPECS.items()]


def get_predictor(key: str):
    """Cached-per-key predictor factory. Heavy backbone eviction happens
    inside _get_backbone/_embed on first use, not here -- the predictor
    object itself is cheap regardless of kind."""
    spec = MODEL_SPECS[key]
    with spec["config_path"].open() as fh:
        config = json.load(fh)

    if spec["kind"] == "sklearn_embed":
        return SklearnEmbedPredictor(key, spec, config)
    elif spec["kind"] == "pytorch_mlp_embed":
        return PyTorchMLPPredictor(key, spec, config)
    elif spec["kind"] == "clip_finetuned":
        return CLIPFineTunedPredictor()
    elif spec["kind"] == "resnet50_finetuned":
        return ResNet50FineTunedPredictor()
    else:
        raise ValueError(f"unknown model kind: {spec['kind']!r}")
