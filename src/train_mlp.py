"""Shared PyTorch MLP-head trainer -- used for models 6 (small MLP, CLIP),
7 (deep MLP, CLIP), and 10 (DINOv2 + MLP), differing only by --hidden-dims
and --embeddings. Keeping one script means the "does head depth matter" and
"does backbone matter" comparisons are apples-to-apples: model 10 reuses
model 6's exact architecture on a different embeddings file.

Usage:
  python src/train_mlp.py --key mlp_small --label "Small MLP (CLIP)" \
      --embeddings data/manifests/embeddings.npz --embeddings-source clip \
      --hidden-dims 256,128 --dropout 0.2 --lr 1e-3
  python src/train_mlp.py --key mlp_deep --label "Deep MLP (CLIP)" \
      --embeddings data/manifests/embeddings.npz --embeddings-source clip \
      --hidden-dims 512,256,128,64 --dropout 0.3 --lr 5e-4
  python src/train_mlp.py --key dinov2_mlp --label "DINOv2 (frozen) + MLP" \
      --embeddings data/manifests/embeddings_dinov2.npz --embeddings-source dinov2 \
      --hidden-dims 256,128 --dropout 0.2 --lr 1e-3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import common_eval as ce

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_EPOCHS = 200
PATIENCE = 15
BATCH_SIZE = 256


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--embeddings-source", required=True, choices=["clip", "dinov2"])
    p.add_argument("--hidden-dims", required=True, help="comma-separated, e.g. 256,128")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    df, X_all = ce.load_embeddings(PROJECT_ROOT / args.embeddings)
    y_all = (df["cls"] == "fake").astype(int).values.astype(np.float32)
    masks = ce.build_eval_masks(df)

    X_train, y_train = X_all[masks["train"]], y_all[masks["train"]]
    w_train = ce.combined_sample_weights(df[masks["train"]], y_train)
    X_val, y_val = X_all[masks["val"]], y_all[masks["val"]]
    print(f"training on {len(y_train)} rows ({int(y_train.sum())} fake, {int(len(y_train) - y_train.sum())} real), "
          f"input_dim={X_all.shape[1]}, hidden_dims={hidden_dims}", flush=True)

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float(),
                              torch.from_numpy(w_train).float())
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    X_val_t = torch.from_numpy(X_val).float().to(device)

    model = MLPHead(X_all.shape[1], hidden_dims, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    best_val_auc, best_state, epochs_no_improve = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb, wb in train_dl:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = (loss_fn(logits, yb) * wb).mean()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        total_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).cpu().numpy()
        val_proba = 1 / (1 + np.exp(-val_logits))
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(y_val, val_proba)

        if val_auc > best_val_auc:
            best_val_auc, best_state, epochs_no_improve = val_auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0 or epochs_no_improve == 0:
            print(f"  epoch {epoch + 1}: train_loss={total_loss:.4f} val_auc={val_auc:.4f} "
                  f"(best={best_val_auc:.4f}, no_improve={epochs_no_improve})", flush=True)
        if epochs_no_improve >= PATIENCE:
            print(f"early stopping at epoch {epoch + 1} (no val AUC improvement for {PATIENCE} epochs)", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()

    def proba_fn(X):
        with torch.no_grad():
            logits = model(torch.from_numpy(X).float().to(device)).cpu().numpy()
        return 1 / (1 + np.exp(-logits))

    results = ce.run_all_evals(proba_fn, df, X_all, y_all, masks)

    ce.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ce.MODELS_DIR / f"{args.key}.pt"
    torch.save(model.state_dict(), model_path)
    ce.save_results(args.key, args.label, "pytorch_mlp_embed", {
        "clip_model_hf_id": "openai/clip-vit-base-patch32" if args.embeddings_source == "clip" else None,
        "dinov2_model_hf_id": "facebook/dinov2-base" if args.embeddings_source == "dinov2" else None,
        "embeddings_source": args.embeddings_source,
        "artifact_path": str(model_path.relative_to(PROJECT_ROOT)),
        "input_dim": int(X_all.shape[1]),
        "hidden_dims": hidden_dims,
        "dropout": args.dropout,
    }, results, sorted(df.loc[masks["train"], "generator"].unique().tolist()))


if __name__ == "__main__":
    main()
