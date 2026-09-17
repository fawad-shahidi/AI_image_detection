"""Model 9/10: fine-tuned ResNet50 (ImageNet-pretrained), freeze
conv1/bn1/layer1/layer2, unfreeze layer3/layer4 + a new linear head.

Much smaller footprint than CLIP ViT-B (25.6M params total vs 87.8M), so
batch_size=32 fits comfortably on 4GB in plain fp32. Two separate NaN issues
surfaced getting here, both from fp16 autocast (originally added as a
"safety margin", not because it was needed): (1) autocast+lr=1e-4 during
*training* diverged to NaN loss by step 500 -- fixed by dropping autocast
from the training loop, lowering the backbone LR 10x, and adding gradient
clipping; (2) with training fixed (finished a full stable epoch), the
*eval-time* autocast forward pass in finetune_common.py still produced NaN
logits on validation data, through ResNet50's BatchNorm-heavy unfrozen
layers -- fixed by passing use_amp=False to predict_proba_for_df/
run_image_evals for this model specifically (CLIP-FT has no BatchNorm and
keeps eval-time autocast). Trains on the same standardized_path images as
the CLIP fine-tune, but with ImageNet normalization (not CLIP's) -- a real,
not cosmetic, difference.
"""
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.utils.data import DataLoader

import common_eval as ce
from finetune_common import predict_proba_for_df, run_image_evals
from image_dataset import StandardizedImageDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEY = "resnet50_ft"
LABEL = "ResNet50 (fine-tuned)"

BATCH_SIZE = 32
EPOCHS = 5
BACKBONE_LR = 1e-5
HEAD_LR = 1e-3
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 2
FROZEN_MODULE_NAMES = ("conv1", "bn1", "layer1", "layer2")


class ResNet50FineTuneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # drop the 1000-way ImageNet head, use our own
        for name, module in self.backbone.named_children():
            if name in FROZEN_MODULE_NAMES:
                for p in module.parameters():
                    p.requires_grad = False
        self.head = nn.Linear(in_features, 1)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats).squeeze(-1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    df = pd.read_csv(ce.MANIFEST_PATH)
    masks = ce.build_eval_masks(df)
    df_train = df[masks["train"]].reset_index(drop=True)
    y_train = (df_train["cls"] == "fake").astype(int).values
    w_train = ce.combined_sample_weights(df_train, y_train)
    print(f"training on {len(df_train)} rows ({int(y_train.sum())} fake, {int(len(y_train) - y_train.sum())} real)", flush=True)

    train_ds = StandardizedImageDataset(df_train, normalization="imagenet", weights=w_train)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))

    model = ResNet50FineTuneModel().to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable / 1e6:.1f}M / {n_total / 1e6:.1f}M total", flush=True)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": model.head.parameters(), "lr": HEAD_LR},
    ])
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    best_val_auc, best_state = -1.0, None
    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        total_loss, n_seen = 0.0, 0
        for step, (xb, yb, wb) in enumerate(train_dl):
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = (loss_fn(logits, yb) * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            total_loss += loss.item() * len(xb)
            n_seen += len(xb)
            assert not (loss.item() != loss.item()), (  # NaN check: NaN != NaN
                f"loss went NaN at epoch {epoch + 1} step {step + 1} -- training diverged, "
                f"refusing to silently continue on garbage gradients")
            if (step + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"  epoch {epoch + 1}, step {step + 1}/{len(train_dl)}, "
                      f"{elapsed:.1f}s elapsed ({n_seen / elapsed:.1f} img/sec), "
                      f"running_loss={total_loss / n_seen:.4f}", flush=True)

        val_sub = df[masks["val"]]
        val_y = (val_sub["cls"] == "fake").astype(int).values
        val_proba = predict_proba_for_df(model, val_sub, device, normalization="imagenet", use_amp=False)
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_y, val_proba)
        print(f"epoch {epoch + 1}/{EPOCHS} done in {time.time() - t0:.1f}s: "
              f"train_loss={total_loss / n_seen:.4f} val_auc={val_auc:.4f}", flush=True)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device).eval()
    print(f"\nbest val AUC during training: {best_val_auc:.4f} -- running full eval suite", flush=True)
    results = run_image_evals(model, df, masks, device, normalization="imagenet", use_amp=False)

    ce.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ce.MODELS_DIR / f"{KEY}.pt"
    torch.save(model.state_dict(), model_path)
    ce.save_results(KEY, LABEL, "resnet50_finetuned", {
        "clip_model_hf_id": None,
        "embeddings_source": None,
        "artifact_path": str(model_path.relative_to(PROJECT_ROOT)),
        "frozen_modules": list(FROZEN_MODULE_NAMES),
        "epochs": EPOCHS,
    }, results, sorted(df_train["generator"].unique().tolist()))


if __name__ == "__main__":
    main()
