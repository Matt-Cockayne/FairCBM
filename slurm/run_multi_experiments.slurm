#!/bin/bash
#SBATCH --job-name=fair_cbm_%j
#SBATCH --output=logs/fair_cbm_%j_%a.out
#SBATCH --error=logs/fair_cbm_%j_%a.err
#SBATCH --array=1-100
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Fairness-Aware CBM Multi-Run Experiment Script
# This script runs 100 experiments for all 5 model types
# Array job: 1-100 for 100 independent runs

echo "=========================================="
echo "FAIRNESS-AWARE CBM EXPERIMENT"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/.bashrc
conda activate CBM-env

# Configuration
BACKBONE="swin"  # Options: swin, convnext, vit, efficientnet, mobilenet
DATA_ROOT="/home/csc29/projects/SkinCAP"
RAW_CSV="/home/csc29/projects/SkinCAP/skincap_v240623.csv"
CONCEPTS_PATH="data/skincap_concepts.txt"
RESULTS_DIR="results"
EPOCHS=100
BATCH_SIZE=32
LR=1e-4
TRAIN_SPLIT=0.8
VAL_SPLIT=0.1
TEST_SPLIT=0.1

# Fairness hyperparameters (for Fair CBM only)
FAIRNESS_LAMBDA=0.1
ADVERSARIAL_LAMBDA=0.01
# Warmup will auto-compute as 30% of EPOCHS (default behavior)

# Experiment configuration
# Run all 5 models for each seed
RUN_ID=$SLURM_ARRAY_TASK_ID
SEED=$((42 + RUN_ID))
EXP_NAME="multi_run_${SLURM_JOB_ID}"

echo "Backbone: $BACKBONE"
echo "Run ID: $RUN_ID"
echo "Seed: $SEED"
echo "Experiment Name: $EXP_NAME"
echo ""

# Create logs directory
mkdir -p logs

# Function to train a model
train_model() {
    local MODEL_TYPE=$1
    local MODEL_NAME=$2
    
    echo "=========================================="
    echo "Training $MODEL_NAME (Run $RUN_ID)"
    echo "=========================================="
    
    python scripts/train_all_models.py \
        --model_type $MODEL_TYPE \
        --backbone $BACKBONE \
        --fairness_lambda $FAIRNESS_LAMBDA \
        --adversarial_lambda $ADVERSARIAL_LAMBDA \
        --exp_name "${EXP_NAME}/run_${RUN_ID}" \
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
        echo "$MODEL_NAME training completed successfully"
    else
        echo "ERROR: $MODEL_NAME training failed"
        return 1
    fi
    echo ""
}

# Train all 5 models
train_model "direct" "Direct Classifier"
train_model "standard_cbm" "Standard CBM"
train_model "curriculum_cbm" "Curriculum CBM"
train_model "fair_cbm" "Fair CBM (no curriculum)"
train_model "fair_curriculum_cbm" "Fair Curriculum CBM (full)"

echo "=========================================="
echo "All 5 models trained for Run $RUN_ID"
echo "End Time: $(date)"
echo "=========================================="

# Run evaluation if this is the last array job
if [ $SLURM_ARRAY_TASK_ID -eq 100 ]; then
    echo ""
    echo "=========================================="
    echo "Running final evaluation (last array job)"
    echo "=========================================="
    
    python scripts/evaluate_fairness_comparison.py \
        --exp_name "${EXP_NAME}/run_100" \
        --backbone $BACKBONE \
        --results_dir $RESULTS_DIR \
        --data_root $DATA_ROOT \
        --raw_csv $RAW_CSV \
        --train_split $TRAIN_SPLIT \
        --val_split $VAL_SPLIT \
        --test_split $TEST_SPLIT \
        --concepts_path $CONCEPTS_PATH \
        --batch_size $BATCH_SIZE \
        --checkpoint best_model.pt \
        --n_bootstrap 1000
    
    echo "Evaluation complete"
fi

echo "Job complete!"
