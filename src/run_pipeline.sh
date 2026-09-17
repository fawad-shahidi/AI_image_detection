#!/bin/bash
# Full pipeline, gated: stops automatically if the confound baseline fires
# (make_splits/embed/train_head never run in that case).
set -e
cd /d/project/photo_detection
PY="python"

echo "=== [1/6] download (resumable) ==="
"$PY" src/download.py

echo "=== [2/6] extract_and_manifest ==="
"$PY" src/extract_and_manifest.py

echo "=== [3/7] dedup ==="
"$PY" src/dedup.py

echo "=== [4/7] standardize ==="
"$PY" src/standardize.py

echo "=== [5/7] confound_baseline (GATE) ==="
"$PY" src/confound_baseline.py
echo "confound gate passed"

echo "=== [6/7] make_splits ==="
"$PY" src/make_splits.py

echo "=== [7/7] embed ==="
"$PY" src/embed.py

echo "=== [logreg] Logistic Regression (CLIP) ==="
"$PY" src/train_head.py

echo "=== [linsvm] Linear SVM (CLIP) ==="
"$PY" src/train_linsvm.py

echo "=== [rf] Random Forest (CLIP) ==="
"$PY" src/train_rf.py

echo "=== [lgbm] LightGBM (CLIP) ==="
"$PY" src/train_lgbm.py

echo "=== [knn] k-NN (CLIP) ==="
"$PY" src/train_knn.py

echo "=== [mlp_small] Small MLP (CLIP) ==="
"$PY" src/train_mlp.py --key mlp_small --label "Small MLP (CLIP)" \
    --embeddings data/manifests/embeddings.npz --embeddings-source clip \
    --hidden-dims 256,128 --dropout 0.2 --lr 1e-3

echo "=== [mlp_deep] Deep MLP (CLIP) ==="
"$PY" src/train_mlp.py --key mlp_deep --label "Deep MLP (CLIP)" \
    --embeddings data/manifests/embeddings.npz --embeddings-source clip \
    --hidden-dims 512,256,128,64 --dropout 0.3 --lr 5e-4

echo "=== [embed_dinov2] frozen DINOv2-base embedding cache ==="
"$PY" src/embed_dinov2.py

echo "=== [dinov2_mlp] DINOv2 (frozen) + MLP ==="
"$PY" src/train_mlp.py --key dinov2_mlp --label "DINOv2 (frozen) + MLP" \
    --embeddings data/manifests/embeddings_dinov2.npz --embeddings-source dinov2 \
    --hidden-dims 256,128 --dropout 0.2 --lr 1e-3

echo "=== [clip_ft] CLIP (partial fine-tune) ==="
"$PY" src/train_clip_finetuned.py

echo "=== [resnet50_ft] ResNet50 (fine-tuned) ==="
"$PY" src/train_resnet50_finetuned.py

echo "=== compare_models ==="
"$PY" src/compare_models.py

echo "PIPELINE COMPLETE"
