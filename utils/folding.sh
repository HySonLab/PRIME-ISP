#!/bin/bash

# ==========================================
# ESMFold2 Structure Prediction Script
# ==========================================
# Requires make_splits.py to have already been run for this task -- folding.py
# reads Datasets/{TASK}/splits/*.txt to determine which sequences to fold.

# ---- Task ----
# Which task's sequences to fold.
# Options: PPI-site | PepPI-site | DPI-site | RPI-site | MPI-site | CarbPI-site | SMPI-site
TASK="SMPI-site"

TASK_DIR="./Datasets/${TASK}"

# ---- GPU device ----
CUDA_DEVICE=3

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