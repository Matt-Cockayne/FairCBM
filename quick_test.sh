#!/bin/bash

# Quick test script for fairness-aware CBM
# Usage: ./quick_test.sh [model_type] [epochs]

set -e  # Exit on error

MODEL_TYPE=${1:-"fair_curriculum_cbm"}
EPOCHS=${2:-20}
SEED=${3:-42}

echo "========================================"
echo "FAIRNESS CBM QUICK TEST"
echo "========================================"
echo "Model Type: $MODEL_TYPE"
echo "Epochs: $EPOCHS"
echo "Seed: $SEED"
echo "========================================"

# Check if conda environment is active
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "CBM-env" ]]; then
    echo "Activating CBM-env..."
    source ~/.bashrc
    conda activate CBM-env
fi

# Verify Python imports
echo "Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || { echo "PyTorch not found!"; exit 1; }
python -c "from src.data.dataloader import SkinCapDataset; print('Dataloader: OK')" || { echo "Dataloader import failed!"; exit 1; }
python -c "from src.models.fairness_aware_cbm import FairnessAwareCBM; print('FairnessAwareCBM: OK')" || { echo "FairnessAwareCBM import failed!"; exit 1; }
echo "All dependencies OK!"
echo ""

# Create logs directory
mkdir -p logs

# Run training
echo "Starting training..."
python scripts/train_all_models.py \
    --model_type $MODEL_TYPE \
    --backbone swin \
    --exp_name quick_test_$(date +%Y%m%d_%H%M%S) \
    --seed $SEED \
    --epochs $EPOCHS \
    --batch_size 32 \
    --lr 1e-4 \
    --fairness_lambda 0.1 \
    --adversarial_lambda 0.01 \
    --data_root /home/csc29/projects/SkinCAP \
    --raw_csv /home/csc29/projects/SkinCAP/skincap_v240623.csv \
    --train_split 0.8 \
    --val_split 0.1 \
    --test_split 0.1 \
    --concepts_path data/skincap_concepts.txt \
    --save_dir results \
    --eval_every 1 \
    --save_best

echo ""
echo "========================================"
echo "Test complete!"
echo "========================================"
