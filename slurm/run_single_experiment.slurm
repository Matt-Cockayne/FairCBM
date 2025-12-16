#!/bin/bash
#SBATCH --job-name=fair_cbm_single
#SBATCH --output=logs/fair_cbm_single_%j.out
#SBATCH --error=logs/fair_cbm_single_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Single Experiment for Testing
# Use this to verify everything works before launching multi-run batch
#
# Model Types Available:
#   direct                - Direct classifier (no CBM)
#   standard_cbm          - Standard CBM (no fairness, no curriculum)
#   fair_standard_cbm     - Fair Standard CBM (fairness, no curriculum)
#   curriculum_cbm        - Old concept difficulty curriculum (no fairness)
#   fair_curriculum_cbm   - NEW 4-phase fairness-first curriculum
#
# Usage:
#   sbatch slurm/run_single_experiment.slurm [model_type] [backbone] [seed]
#   sbatch slurm/run_single_experiment.slurm fair_curriculum_cbm swin 42

echo "=========================================="
echo "SINGLE MODEL TEST"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/.bashrc
conda activate CBM-env

# Configuration
MODEL_TYPE=${1:-"fair_curriculum_cbm"}  # Default to fair_curriculum_cbm
BACKBONE=${2:-"swin"}
SEED=${3:-42}

DATA_ROOT="/home/csc29/projects/SkinCAP"
RAW_CSV="/home/csc29/projects/SkinCAP/skincap_v240623.csv"
CONCEPTS_PATH="data/skincap_concepts.txt"
RESULTS_DIR="results"
EXP_NAME="test_single_${SLURM_JOB_ID}"
EPOCHS=50  # Shorter for testing
BATCH_SIZE=32
LR=1e-4
TRAIN_SPLIT=0.8
VAL_SPLIT=0.1
TEST_SPLIT=0.1

# Fairness hyperparameters (for Fair CBM only)
FAIRNESS_LAMBDA=0.1
ADVERSARIAL_LAMBDA=0.01
ADVERSARIAL_WARMUP_EPOCHS=15  # 30% of 50 epochs

echo "Model Type: $MODEL_TYPE"
echo "Backbone: $BACKBONE"
echo "Seed: $SEED"
echo "Experiment Name: $EXP_NAME"
echo "Epochs: $EPOCHS"
echo ""

# Create logs directory
mkdir -p logs

# Train model
echo "=========================================="
echo "Training $MODEL_TYPE"
echo "=========================================="

python scripts/train_all_models.py \
    --model_type $MODEL_TYPE \
    --backbone $BACKBONE \
    --fairness_lambda $FAIRNESS_LAMBDA \
    --adversarial_lambda $ADVERSARIAL_LAMBDA \
    --adversarial_warmup_epochs $ADVERSARIAL_WARMUP_EPOCHS \
    --exp_name $EXP_NAME \
    --seed $SEED \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --data_root $DATA_ROOT \
    --raw_csv $RAW_CSV \
    --train_split $TRAIN_SPLIT \
    --val_split $VAL_SPLIT \
    --test_split $TEST_SPLIT \
    --concepts_path $CONCEPTS_PATH \
    --save_dir $RESULTS_DIR \
    --eval_every 1 \
    --save_best

if [ $? -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "ERROR: Training failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Training Complete"
echo "End Time: $(date)"
echo "=========================================="

# Check if model was saved
MODEL_PATH="${RESULTS_DIR}/${EXP_NAME}/${MODEL_TYPE}/best_model.pt"
if [ -f "$MODEL_PATH" ]; then
    echo "Model saved successfully: $MODEL_PATH"
else
    echo "WARNING: Model not found at expected path: $MODEL_PATH"
fi

echo ""
echo "Job complete!"
echo "To view results: ls -lh ${RESULTS_DIR}/${EXP_NAME}/${MODEL_TYPE}/"
