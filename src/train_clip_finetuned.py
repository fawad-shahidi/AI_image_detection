"""Model 8/10: partially fine-tuned CLIP (unfreeze last 2 vision-encoder
layers only, given the 4GB VRAM budget).

Uses CLIPVisionModelWithProjection (vision tower + projection only, NOT the
full CLIPModel) -- verified it has no text encoder loaded at all (87.8M
params vs CLIPModel's ~151M), which matters on a 4GB card. Trains directly
on standardized_path images (224x224 already), not cached embeddings, since
the whole point is updating the encoder's weights.

VRAM mitigations: batch_size=8 physical, gradient-accumulated to an
effective batch of 32; fp16 autocast + GradScaler; only the last 2 of 12
encoder layers + post_layernorm + visual_projection get gradients (frozen
layers still run forward, since the graph must pass through them to reach
the unfrozen ones, but carry no backward cost).
"""
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPVisionModelWithProjection

import common_eval as ce
from finetune_common import predict_proba_for_df, run_image_evals
from image_dataset import StandardizedImageDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEY = "clip_ft"
LABEL = "CLIP (partial fine-tune)"
CLIP_HF_ID = "openai/clip-vit-base-patch32"

N_UNFROZEN_LAYERS = 2
BATCH_SIZE = 8
ACCUM_STEPS = 4  # effective batch 32
EPOCHS = 3
BACKBONE_LR = 2e-5
HEAD_LR = 1e-3
NUM_WORKERS = 2


class CLIPFineTuneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CLIPVisionModelWithProjection.from_pretrained(CLIP_HF_ID)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for layer in self.backbone.vision_model.encoder.layers[-N_UNFROZEN_LAYERS:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.backbone.vision_model.post_layernorm.parameters():
            p.requires_grad = True
        for p in self.backbone.visual_projection.parameters():
            p.requires_grad = True
        self.head = nn.Linear(self.backbone.config.projection_dim, 1)

    def forward(self, pixel_values):
        image_embeds = self.backbone(pixel_values=pixel_values).image_embeds
        return self.head(image_embeds).squeeze(-1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    df = pd.read_csv(ce.MANIFEST_PATH)
    masks = ce.build_eval_masks(df)
    df_train = df[masks["train"]].reset_index(drop=True)
    y_train = (df_train["cls"] == "fake").astype(int).values
    w_train = ce.combined_sample_weights(df_train, y_train)
    print(f"training on {len(df_train)} rows ({int(y_train.sum())} fake, {int(len(y_train) - y_train.sum())} real)", flush=True)

    train_ds = StandardizedImageDataset(df_train, normalization="clip", weights=w_train)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))

    model = CLIPFineTuneModel().to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable / 1e6:.1f}M / {n_total / 1e6:.1f}M total", flush=True)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": model.head.parameters(), "lr": HEAD_LR},
    ])
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    best_val_auc, best_state = -1.0, None
    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        total_loss, n_seen = 0.0, 0
        opt.zero_grad()
        for step, (xb, yb, wb) in enumerate(train_dl):
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(xb)
                loss = (loss_fn(logits, yb) * wb).mean() / ACCUM_STEPS
            scaler.scale(loss).backward()
            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            total_loss += loss.item() * ACCUM_STEPS * len(xb)
            n_seen += len(xb)
            if (step + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"  epoch {epoch + 1}, step {step + 1}/{len(train_dl)}, "
                      f"{elapsed:.1f}s elapsed ({n_seen / elapsed:.1f} img/sec), "
                      f"running_loss={total_loss / n_seen:.4f}", flush=True)

        val_sub = df[masks["val"]]
        val_y = (val_sub["cls"] == "fake").astype(int).values
        val_proba = predict_proba_for_df(model, val_sub, device, normalization="clip")
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
    results = run_image_evals(model, df, masks, device, normalization="clip")

    ce.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ce.MODELS_DIR / f"{KEY}.pt"
    torch.save(model.state_dict(), model_path)
    ce.save_results(KEY, LABEL, "clip_finetuned", {
        "clip_model_hf_id": CLIP_HF_ID,
        "embeddings_source": None,
        "artifact_path": str(model_path.relative_to(PROJECT_ROOT)),
        "n_unfrozen_layers": N_UNFROZEN_LAYERS,
        "epochs": EPOCHS,
    }, results, sorted(df_train["generator"].unique().tolist()))


if __name__ == "__main__":
    main()
