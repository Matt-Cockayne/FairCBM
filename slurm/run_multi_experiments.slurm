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
# This script runs N experiments for all 5 model types
# Default: 100 runs. Override with: sbatch --array=1-N run_multi_experiments.slurm
# Example: sbatch --array=1-20 run_multi_experiments.slurm  # Run 20 experiments
#
# IMPORTANT: Before first run, download pretrained weights on head node:
#   conda activate CBM-env
#   python scripts/download_weights.py

# Configuration - Number of runs (matches array size)
# This should match your --array specification
N_RUNS=${N_RUNS:-100}  # Default to 100 if not set

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
        --eval_every 1
        # Note: Not using --save_best, will save only global best models
    
    if [ $? -eq 0 ]; then
        echo "$MODEL_NAME training completed successfully"
        
        # Check if this model is the best so far and update global best
        update_best_model "$MODEL_TYPE" "$MODEL_NAME"
    else
        echo "ERROR: $MODEL_NAME training failed"
        return 1
    fi
    echo ""
}

# Function to update best model if current run is better
update_best_model() {
    local MODEL_TYPE=$1
    local MODEL_NAME=$2
    
    # Create shared directory for best models (across all array jobs)
    BEST_MODELS_DIR="${RESULTS_DIR}/${EXP_NAME}/best_models"
    mkdir -p $BEST_MODELS_DIR
    
    # Get validation F1 from current run
    HISTORY_FILE="${RESULTS_DIR}/${EXP_NAME}/run_${RUN_ID}/${MODEL_TYPE}/history.json"
    if [ ! -f "$HISTORY_FILE" ]; then
        echo "  Warning: No history file found for $MODEL_TYPE"
        return
    fi
    
    CURRENT_F1=$(python -c "
import json
try:
    with open('${HISTORY_FILE}') as f:
        history = json.load(f)
    if 'val' in history and len(history['val']) > 0:
        val_f1s = [v['standard_metrics']['f1'] for v in history['val']]
        print(max(val_f1s))
    else:
        print(-1)
except:
    print(-1)
")
    
    if [ "$CURRENT_F1" == "-1" ]; then
        echo "  Warning: Could not extract validation F1 for $MODEL_TYPE"
        return
    fi
    
    # Lock file for atomic operations
    LOCK_FILE="${BEST_MODELS_DIR}/${MODEL_TYPE}.lock"
    BEST_F1_FILE="${BEST_MODELS_DIR}/${MODEL_TYPE}_best_f1.txt"
    
    # Copy model to temp location first
    TEMP_MODEL="${BEST_MODELS_DIR}/${MODEL_TYPE}_run_${RUN_ID}_temp.pt"
    SOURCE_MODEL="${RESULTS_DIR}/${EXP_NAME}/run_${RUN_ID}/${MODEL_TYPE}/best_model.pt"
    
    if [ ! -f "$SOURCE_MODEL" ]; then
        echo "  Warning: No best_model.pt found for $MODEL_TYPE"
        return
    fi
    
    # Copy to temp location (only place besides shared best_models where model exists)
    cp "$SOURCE_MODEL" "$TEMP_MODEL"
    
    # Use flock for atomic file locking
    (
        flock -x 200
        
        # Read current best F1 (if exists)
        if [ -f "$BEST_F1_FILE" ]; then
            BEST_F1=$(cat "$BEST_F1_FILE")
        else
            BEST_F1=-1
        fi
        
        # Compare and update if current is better
        IS_BETTER=$(python -c "print(1 if float('${CURRENT_F1}') > float('${BEST_F1}') else 0)")
        
        if [ "$IS_BETTER" == "1" ]; then
            echo "  ✓ NEW BEST $MODEL_NAME: Run ${RUN_ID} with Val F1 = ${CURRENT_F1} (previous best: ${BEST_F1})"
            
            # Update best F1 file
            echo "$CURRENT_F1" > "$BEST_F1_FILE"
            
            # Get the best epoch from history
            BEST_EPOCH=$(python -c "
import json
with open('${HISTORY_FILE}') as f:
    history = json.load(f)
val_f1s = [v['standard_metrics']['f1'] for v in history['val']]
best_idx = val_f1s.index(max(val_f1s))
print(history['val'][best_idx]['epoch'])
")
            
            # Move temp to best (atomic replace)
            mv "$TEMP_MODEL" "${BEST_MODELS_DIR}/${MODEL_TYPE}_best.pt"
            
            # Copy history
            cp "${HISTORY_FILE}" \
               "${BEST_MODELS_DIR}/${MODEL_TYPE}_best_history.json"
            
            # Save metadata
            echo "Run: ${RUN_ID}, Seed: ${SEED}, Val F1: ${CURRENT_F1}, Best Epoch: ${BEST_EPOCH}" > \
                 "${BEST_MODELS_DIR}/${MODEL_TYPE}_best_info.txt"
        else
            echo "  Current $MODEL_NAME Val F1 = ${CURRENT_F1} (best so far: ${BEST_F1})"
            # Delete temp file - not the best
            rm -f "$TEMP_MODEL"
        fi
        
    ) 200>"$LOCK_FILE"
    
    # Clean up model checkpoint only - keep history.json for analysis
    rm -f "${RESULTS_DIR}/${EXP_NAME}/run_${RUN_ID}/${MODEL_TYPE}/best_model.pt"
    rm -f "${RESULTS_DIR}/${EXP_NAME}/run_${RUN_ID}/${MODEL_TYPE}/final_model.pt"
    # Keep history.json - needed for analyze_multi_run_results.py
}

# Train all 5 models
train_model "direct" "Direct Classifier"
train_model "standard_cbm" "Standard CBM"
train_model "fair_standard_cbm" "Fair Standard CBM (no curriculum)"
train_model "curriculum_cbm" "Curriculum CBM"
train_model "fair_curriculum_cbm" "Fair Curriculum CBM (full)"

echo "=========================================="
echo "All 5 models trained for Run $RUN_ID"
echo "End Time: $(date)"
echo "=========================================="

# Run evaluation if this is the last array job
if [ $SLURM_ARRAY_TASK_ID -eq $N_RUNS ]; then
    echo ""
    echo "=========================================="
    echo "Running final evaluation (last array job)"
    echo "=========================================="
    
    # Analyze all runs and generate statistics
    echo "Analyzing ${N_RUNS} runs..."
    python scripts/analyze_multi_run_results.py \
        --exp_name $EXP_NAME \
        --backbone $BACKBONE \
        --n_runs $N_RUNS
    
    echo ""
    echo "=========================================="
    echo "Multi-run experiment complete!"
    echo "=========================================="
    echo "Best models (selected during training):"
    
    BEST_MODELS_DIR="${RESULTS_DIR}/${EXP_NAME}/best_models"
    for MODEL_TYPE in direct standard_cbm fair_standard_cbm curriculum_cbm fair_curriculum_cbm; do
        if [ -f "${BEST_MODELS_DIR}/${MODEL_TYPE}_best_info.txt" ]; then
            echo ""
            echo "${MODEL_TYPE}:"
            cat "${BEST_MODELS_DIR}/${MODEL_TYPE}_best_info.txt"
        fi
    done
    
    echo ""
    echo "Best models saved in: ${BEST_MODELS_DIR}"
    echo "Total: Only 5 model checkpoints saved (one per model type)"
    echo "Individual run directories cleaned up automatically during training"
fi

echo "Job complete!"
