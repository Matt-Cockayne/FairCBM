#!/bin/bash
#SBATCH --job-name=ablation_study
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-599  # 6 ablations × 100 runs = 600 jobs

# Ablation Study SLURM Script
# Runs all 6 ablation configurations (full_model + 5 ablated variants) with 100 runs each
# Uses train_all_models.py for consistent training logic with main experiment
#
# Usage:
#   sbatch run_ablation_study.slurm

# Activate conda environment
source ~/.bashrc
conda activate CBM-env

# Experiment configuration
EXP_NAME="ablation_study_$(date +%Y%m%d)"
BACKBONE="swin"
EPOCHS=100
BATCH_SIZE=32
LR=1e-4
FAIRNESS_LAMBDA=0.1   # Match main experiment
ADVERSARIAL_LAMBDA=0.01   # Match main experiment

# Data paths (matching multi_experiments.slurm)
DATA_ROOT="/home/csc29/projects/SkinCAP"
RAW_CSV="/home/csc29/projects/SkinCAP/skincap_v240623.csv"
CONCEPTS_PATH="data/skincap_concepts.txt"
OUTPUT_DIR="results"  # Will create results/<exp_name>/run_<run_id>/fair_curriculum_cbm/
TRAIN_SPLIT=0.8
VAL_SPLIT=0.1
TEST_SPLIT=0.1

# Create logs directory
mkdir -p logs

# Ablation configurations (updated to match new implementation)
ABLATIONS=("full_model" "no_phase1" "no_phase2" "no_phase3" "no_phase4" "no_adversarial")
NUM_ABLATIONS=${#ABLATIONS[@]}
RUNS_PER_ABLATION=100

# Calculate which ablation and run this job corresponds to
ABLATION_IDX=$((SLURM_ARRAY_TASK_ID / RUNS_PER_ABLATION))
RUN_ID=$((SLURM_ARRAY_TASK_ID % RUNS_PER_ABLATION))
ABLATION_KEY=${ABLATIONS[$ABLATION_IDX]}

# Set unique random seed for this run (base seed + task_id for reproducibility)
SEED=$((42 + SLURM_ARRAY_TASK_ID))

# Print job information
echo "============================================"
echo "Ablation Study Job"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Ablation: $ABLATION_KEY ($((ABLATION_IDX + 1))/$NUM_ABLATIONS)"
echo "Run ID: $RUN_ID"
echo "Random Seed: $SEED"
echo "Experiment: $EXP_NAME"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================"
echo ""

# Change to FairCBM directory
cd /home/csc29/projects/SynergyCBM/FairCBM

# Run training using train_all_models.py for consistent training logic
python scripts/train_all_models.py \
    --model_type fair_curriculum_cbm \
    --ablation_key $ABLATION_KEY \
    --exp_name "${EXP_NAME}/${ABLATION_KEY}/run_${RUN_ID}" \
    --backbone $BACKBONE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --fairness_lambda $FAIRNESS_LAMBDA \
    --adversarial_lambda $ADVERSARIAL_LAMBDA \
    --data_root $DATA_ROOT \
    --raw_csv $RAW_CSV \
    --train_split $TRAIN_SPLIT \
    --val_split $VAL_SPLIT \
    --test_split $TEST_SPLIT \
    --concepts_path $CONCEPTS_PATH \
    --save_dir $OUTPUT_DIR \
    --seed $SEED \
    --num_workers 4 \
    --eval_every 1 \
    --save_best

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
    echo "Results saved to: ${OUTPUT_DIR}/${EXP_NAME}/${ABLATION_KEY}/run_${RUN_ID}/fair_curriculum_cbm/"
else
    echo ""
    echo "Training failed with exit code $?"
    exit 1
fi

echo "============================================"
echo "Job complete"
echo "============================================"
