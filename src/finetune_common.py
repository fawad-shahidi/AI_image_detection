"""Shared image-based inference/eval helpers for the two fine-tuning scripts
(train_clip_finetuned.py, train_resnet50_finetuned.py) -- neither has cached
embeddings to index into like the sklearn/MLP scripts, so eval means running
the model over each split's images directly."""
import numpy as np
import torch
from torch.utils.data import DataLoader

import common_eval as ce
from image_dataset import StandardizedImageDataset

NUM_WORKERS = 2


def predict_proba_for_df(model, df_subset, device, normalization, batch_size=32, use_amp=True):
    """use_amp=False for ResNet50: fp16 autocast at inference (not just
    training) produced NaN logits through its BatchNorm-heavy unfrozen
    layers even though fp32 training itself was stable -- see
    train_resnet50_finetuned.py's module docstring. CLIP-FT has no such
    issue (no BatchNorm in a ViT), so it keeps autocast for the modest
    inference speedup."""
    ds = StandardizedImageDataset(df_subset, normalization=normalization)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _yb in dl:
            xb = xb.to(device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda" and use_amp)):
                logits = model(xb)
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probs)


def run_image_evals(model, df, masks, device, normalization, use_amp=True):
    """Same result schema as common_eval.run_all_evals (id_val/ood_aggregate/
    ood_<gen>/midjourney_<gen>)."""
    split_specs = [("id_val", "val", "ID val"), ("ood_aggregate", "ood_aggregate", "OOD aggregate")]
    split_specs += [(f"ood_{g}", f"ood_{g}", f"OOD [{g}]") for g in ce.OOD_GENERATORS]
    split_specs += [(f"midjourney_{g}", f"midjourney_{g}", f"Midjourney [{g}]") for g in ce.MIDJOURNEY_GENERATORS]

    results = {}
    for result_key, mask_key, label in split_specs:
        sub = df[masks[mask_key]]
        y_sub = (sub["cls"] == "fake").astype(int).values
        proba_fn = lambda _X, sub=sub: predict_proba_for_df(model, sub, device, normalization, use_amp=use_amp)
        results[result_key] = ce.eval_auc(proba_fn, sub, y_sub, label)
    return results
