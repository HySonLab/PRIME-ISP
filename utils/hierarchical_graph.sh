#!/bin/bash

# ==========================================
# Hierarchical Graph Builder
# ==========================================

# ---- GPU device ----
CUDA_DEVICE=2

# ---- Task ----

# Which task to build graphs for.
# Options: PPI-site | PepPI-site | DPI-site | RPI-site | MPI-site | CarbPI-site | SMPI-site
TASK="PPI-site"

TASK_DIR="./Datasets/${TASK}"
OUTPUT_DIR="./Datasets/${TASK}/Graphs"

# ---- Pretrained encoder paths (OPTIONAL) ----
# ATOM_ENCODER_PATH="./ckpts/atom_egnn_encoder.pt"
# SURFACE_ENCODER_PATH="./ckpts/surface_emnn_encoder.pt"
ATOM_ENCODER_PATH=""
SURFACE_ENCODER_PATH=""

mkdir -p "$OUTPUT_DIR"

echo "===================================="
echo "Building Hierarchical Graphs"
echo "CUDA device: $CUDA_DEVICE"
echo "Task dir:    $TASK_DIR"
echo "Output:      $OUTPUT_DIR"
echo "===================================="

CMD="CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python ./utils/hierarchical_graph.py \
    --task_dir \"$TASK_DIR\" \
    --output_dir \"$OUTPUT_DIR\""

if [ -n "$ATOM_ENCODER_PATH" ]; then
    CMD="$CMD --atom_encoder_path \"$ATOM_ENCODER_PATH\""
    echo "Using atom encoder:    $ATOM_ENCODER_PATH"
else
    echo "No atom encoder"
fi

if [ -n "$SURFACE_ENCODER_PATH" ]; then
    CMD="$CMD --surface_encoder_path \"$SURFACE_ENCODER_PATH\""
    echo "Using surface encoder: $SURFACE_ENCODER_PATH"
else
    echo "No surface encoder"
fi

echo "===================================="

eval $CMD

echo "Finished building graphs for $TASK_DIR."