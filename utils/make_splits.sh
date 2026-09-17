#!/bin/bash

# ==========================================
# Create train/test splits for all tasks
# ==========================================

declare -A TRAIN_FILES=(
  ["PPI-site_Train335-Test60-Test315"]="Datasets/PPI-site/Train335-Test60-Test315/PPI-Train335.fa"
  ["PPI-site_Train352-Test70"]="Datasets/PPI-site/Train352-Test70/PPI-Train352.fa"
  ["PPI-site_Train9982-Test355"]="Datasets/PPI-site/Train9982-Test355/PPI-Train9982.fa"

  ["PepPI-site_Train640-Test639"]="Datasets/PepPI-site/Train640-Test639/PepPI-Train640.fa"
  ["PepPI-site_Train1154-Test125"]="Datasets/PepPI-site/Train1154-Test125/PepPI-Train1154.fa"

  ["DPI-site"]="Datasets/DPI-site/DPI-Train573.fa"
  ["RPI-site"]="Datasets/RPI-site/RPI-Train495.fa"
  ["MPI-site_CA"]="Datasets/MPI-site/CA-1670_Train.fa"
  ["MPI-site_FE"]="Datasets/MPI-site/FE-278_Train.fa"
  ["MPI-site_MN"]="Datasets/MPI-site/MN-578_Train.fa"

  ["CarbPI-site"]="Datasets/CarbPI-site/Carb-Train517.fa Datasets/CarbPI-site/Carb-Valid129.fa"
  ["SMPI-site"]="Datasets/SMPI-site/SMPI-Train1628.fa Datasets/SMPI-site/SMPI-Valid348.fa"
)

declare -A TEST_FILES=(
  ["PPI-site_Train335-Test60-Test315"]="Datasets/PPI-site/Train335-Test60-Test315/PPI-Test60.fa Datasets/PPI-site/Train335-Test60-Test315/PPI-Test315.fa"
  ["PPI-site_Train352-Test70"]="Datasets/PPI-site/Train352-Test70/PPI-Test70.fa"
  ["PPI-site_Train9982-Test355"]="Datasets/PPI-site/Train9982-Test355/PPI-Test355.fa"

  ["PepPI-site_Train640-Test639"]="Datasets/PepPI-site/Train640-Test639/PepPI-Test639.fa"
  ["PepPI-site_Train1154-Test125"]="Datasets/PepPI-site/Train1154-Test125/PepPI-Test125.fa"

  ["DPI-site"]="Datasets/DPI-site/DPI-Test129.fa Datasets/DPI-site/DPI-Test181.fa"
  ["RPI-site"]="Datasets/RPI-site/RPI-Test117.fa"
  ["MPI-site_CA"]="Datasets/MPI-site/CA-415_Test.fa"
  ["MPI-site_FE"]="Datasets/MPI-site/FE-70_Test.fa"
  ["MPI-site_MN"]="Datasets/MPI-site/MN-144_Test.fa"

  ["CarbPI-site"]="Datasets/CarbPI-site/Carb-Test162.fa"
  ["SMPI-site"]="Datasets/SMPI-site/SMPI-Test348.fa"
)

declare -A OUTPUT_DIRS=(
  ["PPI-site_Train335-Test60-Test315"]="Datasets/PPI-site/splits"
  ["PPI-site_Train352-Test70"]="Datasets/PPI-site/splits"
  ["PPI-site_Train9982-Test355"]="Datasets/PPI-site/splits"

  ["PepPI-site_Train640-Test639"]="Datasets/PepPI-site/splits"
  ["PepPI-site_Train1154-Test125"]="Datasets/PepPI-site/splits"

  ["DPI-site"]="Datasets/DPI-site/splits"
  ["RPI-site"]="Datasets/RPI-site/splits"
  ["MPI-site_CA"]="Datasets/MPI-site/splits"
  ["MPI-site_FE"]="Datasets/MPI-site/splits"
  ["MPI-site_MN"]="Datasets/MPI-site/splits"

  ["CarbPI-site"]="Datasets/CarbPI-site/splits"
  ["SMPI-site"]="Datasets/SMPI-site/splits"
)

for TASK in "${!TRAIN_FILES[@]}"; do
    echo "===================================="
    echo "Creating splits for: $TASK"
    echo "===================================="

    python ./utils/make_splits.py \
        --train_label_files ${TRAIN_FILES[$TASK]} \
        --test_label_files ${TEST_FILES[$TASK]} \
        --output_dir "${OUTPUT_DIRS[$TASK]}"

    echo ""
done

echo "Done. Splits created for: ${!TRAIN_FILES[@]}"