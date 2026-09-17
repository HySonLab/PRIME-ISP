#!/bin/bash

# ==========================================
# PRIME-ISP Binding Site Visualization
# ==========================================

DATA_CONFIG="./config/data_config.yaml"
MODEL_CONFIG="./config/model_config.yaml"

# ---- Task ----
# Options: PPI-site_Train335-Test60-Test315 | PPI-site_Train352-Test70 |
#          PPI-site_Train9982-Test355 | PepPI-site_Train640-Test639 |
#          PepPI-site_Train1154-Test125 | DPI-site | RPI-site |
#          MPI-site_CA | MPI-site_FE | MPI-site_MN | CarbPI-site | SMPI-site
TASK="DPI-site"

# ---- Protein selection ----
# Set RECORD_ID to visualize a specific protein, or leave empty ("") to
# randomly sample N_SAMPLES proteins from the test set instead.
RECORD_ID=""
N_SAMPLES=3

# ---- Test set name (only needed if the task has multiple named test sets) ----
TEST_NAME=""

# ---- Must match what was used in train_prime.sh for this task ----
K_FOLDS=5
SEED=1
ACTIVE_LEVELS=("surface" "atom" "residue" "sse" "protein")
READOUT_LEVEL="residue"

# ---- Prediction / rendering ----
THRESHOLD=0.5
CHAIN=""                    # leave empty to use all chains in the PDB
BACKEND="pymol"        # matplotlib | pymol
GRID_SPACING=0.7            # matplotlib backend only; lower = finer/slower

CUDA_DEVICE=0
OUTPUT_DIR="./visualizations/${TASK}"

echo "===================================="
echo "PRIME-ISP Binding Site Visualization"
echo "Task:          $TASK"
if [ -n "$RECORD_ID" ]; then
    echo "Record id:     $RECORD_ID"
else
    echo "N samples:     $N_SAMPLES (random)"
fi
if [ -n "$TEST_NAME" ]; then
    echo "Test name:     $TEST_NAME"
fi
echo "K-folds:       $K_FOLDS"
echo "Seed:          $SEED"
echo "Threshold:     $THRESHOLD"
echo "Backend:       $BACKEND"
echo "CUDA device:   $CUDA_DEVICE"
echo "Output dir:    $OUTPUT_DIR"
echo "===================================="

CMD="python ./utils/visualize_binding_site.py \
    --task $TASK \
    --data_config $DATA_CONFIG \
    --model_config $MODEL_CONFIG \
    --k_folds $K_FOLDS \
    --seed $SEED \
    --threshold $THRESHOLD \
    --backend $BACKEND \
    --grid_spacing $GRID_SPACING \
    --active_levels ${ACTIVE_LEVELS[@]} \
    --readout_level $READOUT_LEVEL \
    --output_dir $OUTPUT_DIR"

if [ -n "$RECORD_ID" ]; then
    CMD="$CMD --record_id $RECORD_ID"
else
    CMD="$CMD --n_samples $N_SAMPLES"
fi

if [ -n "$TEST_NAME" ]; then
    CMD="$CMD --test_name $TEST_NAME"
fi

if [ -n "$CHAIN" ]; then
    CMD="$CMD --chain $CHAIN"
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE eval $CMD