#!/bin/bash

# ==========================================
# PRIME Testing Script
# ==========================================

DATA_CONFIG="./config/data_config.yaml"
MODEL_CONFIG="./config/model_config.yaml"

# ---- Task to evaluate ----
# Options: PPI-site_Train335-Test60-Test315 | PPI-site_Train352-Test70 |
#          PPI-site_Train9982-Test355 | PepPI-site_Train640-Test639 |
#          PepPI-site_Train1154-Test125 | DPI-site | RPI-site |
#          MPI-site_CA | MPI-site_FE | MPI-site_MN | CarbPI-site | SMPI-site
TASK="CarbPI-site"

# ---- Test set name (only needed if the task has multiple named test sets) ----
# PPI-site_Train335-Test60-Test315: 60 | 315
# DPI-site: 129 | 181
# Leave empty ("") for tasks with a single test set
TEST_NAME=""

# ---- Cross-validation ----
# Set to match whatever K_FOLDS was used in train_prime.sh. If > 1, all K
# fold checkpoints are loaded and their predictions averaged (DeepProSite-
# style ensemble). Set to 1 to use a single checkpoint instead.
K_FOLDS=5

BATCH_SIZE=32
CUDA_DEVICE=3
SEED=1
THRESHOLD=0.5

ACTIVE_LEVELS=("surface" "atom" "residue" "sse" "protein")
READOUT_LEVEL="residue"

# ---- Auto-resolve checkpoint from training (single-checkpoint mode only) ----
LEVEL_TAG="${ACTIVE_LEVELS[*]}"
LEVEL_TAG="${LEVEL_TAG// /_}"
CKPT_PATH="./ckpts/best_prime_${TASK}_${LEVEL_TAG}_seed${SEED}.pt"

echo "===================================="
echo "Testing PRIME"
echo "Task:          $TASK"
if [ -n "$TEST_NAME" ]; then
    echo "Test name:     $TEST_NAME"
fi
echo "K-folds:       $K_FOLDS"
echo "CUDA device:   $CUDA_DEVICE"
echo "Seed:          $SEED"
echo "Threshold:     $THRESHOLD"
if [ "$K_FOLDS" -le 1 ]; then
    echo "Checkpoint:    $CKPT_PATH"
else
    echo "Checkpoints:   ${K_FOLDS} fold checkpoints (auto-resolved)"
fi
echo "===================================="

if [ "$K_FOLDS" -le 1 ] && [ ! -f "$CKPT_PATH" ]; then
    echo "Error: checkpoint not found at $CKPT_PATH"
    exit 1
fi

CMD="python test_prime.py \
    --task $TASK \
    --data_config $DATA_CONFIG \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --threshold $THRESHOLD \
    --k_folds $K_FOLDS \
    --active_levels ${ACTIVE_LEVELS[@]} \
    --readout_level $READOUT_LEVEL"

if [ "$K_FOLDS" -le 1 ]; then
    CMD="$CMD --ckpt_path \"$CKPT_PATH\""
fi

if [ -n "$TEST_NAME" ]; then
    CMD="$CMD --test_name $TEST_NAME"
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE eval $CMD