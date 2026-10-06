#!/bin/bash

# ==========================================
# PRIME-ISP Binding Site Visualization -- ALL TASKS
# ==========================================

DATA_CONFIG="./config/data_config.yaml"
MODEL_CONFIG="./config/model_config.yaml"

# ---- Tasks to run. Format: "TASK_NAME:TEST_NAME" (TEST_NAME empty if N/A) ----
TASKS_AND_TESTS=(
    "PPI-site_Train335-Test60-Test315:60"
    "PPI-site_Train335-Test60-Test315:315"
    "PPI-site_Train352-Test70:"
    "PPI-site_Train9982-Test355:"
    "PepPI-site_Train640-Test639:"
    "PepPI-site_Train1154-Test125:"
    "DPI-site:129"
    "DPI-site:181"
    "RPI-site:"
    "MPI-site_CA:"
    "MPI-site_FE:"
    "MPI-site_MN:"
    "CarbPI-site:"
    "SMPI-site:"
)

# ---- Mode ----
MODE="best_worst"
RECORD_ID=""
N_SAMPLES=3

# ---- Must match what was used in train_prime.sh for each task ----
K_FOLDS=5
SEED=1
ACTIVE_LEVELS=("surface" "atom" "residue" "sse" "protein")
READOUT_LEVEL="residue"

# ---- Prediction / rendering ----
THRESHOLD=0.5
CHAIN=""
BACKEND="pymol"
GRID_SPACING=0.7

CUDA_DEVICE=2

STOP_ON_ERROR=0

run_one() {
    local TASK="$1"
    local TEST_NAME="$2"

    local OUTPUT_DIR="./visualizations/${TASK}"
    if [ -n "$TEST_NAME" ]; then
        OUTPUT_DIR="${OUTPUT_DIR}/${TEST_NAME}"
    fi

    echo "===================================="
    echo "PRIME-ISP Binding Site Visualization"
    echo "Task:          $TASK"
    echo "Mode:          $MODE"
    if [ "$MODE" == "specific" ]; then
        echo "Record id:     $RECORD_ID"
    elif [ "$MODE" == "random" ]; then
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

    local CMD="python ./utils/visualize_binding_site.py \
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

    if [ "$MODE" == "specific" ]; then
        if [ -z "$RECORD_ID" ]; then
            echo "Error: MODE=specific requires RECORD_ID to be set"
            return 1
        fi
        CMD="$CMD --record_id $RECORD_ID"
    elif [ "$MODE" == "random" ]; then
        CMD="$CMD --n_samples $N_SAMPLES"
    fi

    if [ -n "$TEST_NAME" ]; then
        CMD="$CMD --test_name $TEST_NAME"
    fi

    if [ -n "$CHAIN" ]; then
        CMD="$CMD --chain $CHAIN"
    fi

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE eval $CMD
    return $?
}

FAILED_TASKS=()

for entry in "${TASKS_AND_TESTS[@]}"; do
    TASK="${entry%%:*}"
    TEST_NAME="${entry#*:}"
    if [ "$TASK" == "$TEST_NAME" ]; then
        TEST_NAME=""
    fi

    run_one "$TASK" "$TEST_NAME"
    status=$?

    if [ $status -ne 0 ]; then
        label="$TASK"
        [ -n "$TEST_NAME" ] && label="${TASK} (${TEST_NAME})"
        echo "!!!! FAILED: $label (exit code $status)"
        FAILED_TASKS+=("$label")
        if [ "$STOP_ON_ERROR" == "1" ]; then
            echo "Stopping due to STOP_ON_ERROR=1"
            exit $status
        fi
    fi
    echo ""
done

echo "===================================="
echo "All tasks finished."
if [ ${#FAILED_TASKS[@]} -eq 0 ]; then
    echo "All tasks succeeded."
else
    echo "Failed tasks (${#FAILED_TASKS[@]}):"
    for t in "${FAILED_TASKS[@]}"; do
        echo "  - $t"
    done
fi
echo "===================================="