#!/bin/bash

# ==========================================
# ESMFold2 Structure Prediction Script
# ==========================================

# ---- Task ----
# Which task's sequences to fold.
# Options: PPI-site | PepPI-site | DPI-site | RPI-site | MPI-site | CarbPI-site | SMPI-site
TASK="PPI-site"

TASK_DIR="./Datasets/${TASK}"

# ---- GPU device ----
CUDA_DEVICE=2

echo "===================================="
echo "ESMFold2 Structure Prediction"
echo "Task:        $TASK"
echo "Task dir:    $TASK_DIR"
echo "CUDA device: $CUDA_DEVICE"
echo "===================================="

if [ ! -d "$TASK_DIR/splits" ]; then
    echo "Error: $TASK_DIR/splits/ not found -- run make_splits.py for this task first."
    exit 1
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python utils/folding.py "$TASK_DIR"

echo "===================================="
echo "Finished folding for $TASK_DIR."
echo "PDBs written to: $TASK_DIR/PDBs"
if [ -f "$TASK_DIR/failed_sequences.csv" ]; then
    echo "Some sequences failed/were skipped -- see $TASK_DIR/failed_sequences.csv"
fi
echo "===================================="