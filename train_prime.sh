#!/bin/bash

# ==========================================
# PRIME Training Script
# ==========================================

# -----------------------------
# Task
# -----------------------------
# Options: PPI-site_Train335-Test60-Test315 | PPI-site_Train352-Test70 |
#          PPI-site_Train9982-Test355 | PepPI-site_Train640-Test639 |
#          PepPI-site_Train1154-Test125 | DPI-site | RPI-site |
#          MPI-site_CA | MPI-site_FE | MPI-site_MN | CarbPI-site | SMPI-site
TASK="SMPI-site"

# -----------------------------
# Config
# -----------------------------
DATA_CONFIG="./config/data_config.yaml"
MODEL_CONFIG="./config/model_config.yaml"

BATCH_SIZE=32
EPOCHS=100
LR=1e-3

# -----------------------------
# Loss
# -----------------------------
# Options: combined (weighted CE + Dice, default) | weighted_ce | ce (plain, matches DeepPANB/DeepProSite)
LOSS="combined"

# -----------------------------
# Feature augmentation
# -----------------------------
# Gaussian noise magnitude added to residue-level features during training
# only (DeepProSite-style; their sweep found eps=0.1 optimal). 0.0 = disabled.
AUGMENT_EPS=0.1

# -----------------------------
# GPU Selection
# -----------------------------
DEVICE_ID=2

# -----------------------------
# Hierarchy
# -----------------------------
ACTIVE_LEVELS=("surface" "atom" "residue" "sse" "protein")
READOUT_LEVEL="residue"   # residue level for all *PI-site tasks

# -----------------------------
# Early stopping / LR schedule
# -----------------------------
PATIENCE_ES=10   # epochs with no Val metric improvement before stopping
PATIENCE_LR=5    # epochs with no Val loss improvement before LR is reduced

# -----------------------------
# Cross-validation
# -----------------------------
K_FOLDS=5

# -----------------------------
# Seed
# -----------------------------
SEED=1

# -----------------------------
# Resume Option (single-split mode only, ignored if K_FOLDS > 1)
# -----------------------------
RESUME="false"

echo "===================================="
echo "Training PRIME ($TASK)"
echo "Batch Size:      $BATCH_SIZE"
echo "Epochs:          $EPOCHS"
echo "LR:              $LR"
echo "Loss:            $LOSS"
echo "Augment eps:     $AUGMENT_EPS"
echo "GPU:             $DEVICE_ID"
echo "Active Levels:   ${ACTIVE_LEVELS[@]}"
echo "Readout Level:   $READOUT_LEVEL"
echo "Patience (ES):   $PATIENCE_ES"
echo "Patience (LR):   $PATIENCE_LR"
echo "K-folds:         $K_FOLDS"
echo "Seed:            $SEED"
echo "Resume:          $RESUME"
echo "===================================="

export CUDA_VISIBLE_DEVICES=$DEVICE_ID

CMD="python train_prime.py \
    --task $TASK \
    --data_config $DATA_CONFIG \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --loss $LOSS \
    --augment_eps $AUGMENT_EPS \
    --active_levels ${ACTIVE_LEVELS[@]} \
    --readout_level $READOUT_LEVEL \
    --patience_es $PATIENCE_ES \
    --patience_lr $PATIENCE_LR \
    --k_folds $K_FOLDS \
    --seed $SEED"

# auto-resolve resume path (only meaningful when K_FOLDS=1)
if [ "$RESUME" == "true" ] && [ "$K_FOLDS" -le 1 ]; then
    LEVEL_TAG="${ACTIVE_LEVELS[*]}"
    LEVEL_TAG="${LEVEL_TAG// /_}"
    RESUME_PATH="./ckpts/best_prime_${TASK}_${LEVEL_TAG}_seed${SEED}.pt"

    if [ -f "$RESUME_PATH" ]; then
        echo "Resuming from: $RESUME_PATH"
        CMD="$CMD --resume \"$RESUME_PATH\""
    else
        echo "Warning: checkpoint not found at $RESUME_PATH — training from scratch"
    fi
elif [ "$RESUME" == "true" ] && [ "$K_FOLDS" -gt 1 ]; then
    echo "Note: --resume is not supported in k-fold mode, ignoring RESUME=true"
fi

eval $CMD