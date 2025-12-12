# Fairness-Aware Curriculum CBM: Usage Guide

Complete guide for running experiments comparing Direct, Standard CBM, Curriculum CBM, and Fair Curriculum CBM.

## Quick Start

### 1. Single Test Experiment

Test a single model to verify everything works:

```bash
# Test Fair Curriculum CBM
sbatch fairness/slurm/run_single_experiment.slurm fair_curriculum_cbm swin 42

# Test Curriculum CBM (baseline)
sbatch fairness/slurm/run_single_experiment.slurm curriculum_cbm swin 42

# Test Standard CBM
sbatch fairness/slurm/run_single_experiment.slurm standard_cbm swin 42

# Test Direct Classifier
sbatch fairness/slurm/run_single_experiment.slurm direct swin 42
```

### 2. Local Training (CPU/GPU)

Train models locally without SLURM:

```bash
# Activate environment
conda activate CBM-env

# Train Fair Curriculum CBM
python fairness/scripts/train_all_models.py \
    --model_type fair_curriculum_cbm \
    --backbone swin \
    --exp_name test_local \
    --seed 42 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --data_root data/skincap \
    --concepts_path data/skincap_concepts.txt \
    --save_dir fairness/results \
    --eval_every 5 \
    --save_best

# Train Curriculum CBM (baseline for comparison)
python fairness/scripts/train_all_models.py \
    --model_type curriculum_cbm \
    --backbone swin \
    --exp_name test_local \
    --seed 42 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --data_root data/skincap \
    --concepts_path data/skincap_concepts.txt \
    --save_dir fairness/results \
    --eval_every 5 \
    --save_best
```

### 3. Evaluate Trained Models

Compare all 4 models on test set:

```bash
python fairness/scripts/evaluate_fairness_comparison.py \
    --exp_name test_local \
    --backbone swin \
    --results_dir fairness/results \
    --data_root data/skincap \
    --concepts_path data/skincap_concepts.txt \
    --batch_size 32 \
    --checkpoint best_model.pt \
    --n_bootstrap 1000
```

**Output:**
- `fairness/results/test_local/comparison/comparison_table.csv`
- `fairness/results/test_local/comparison/per_group_performance.png`
- `fairness/results/test_local/comparison/fairness_metrics_comparison.png`
- `fairness/results/test_local/comparison/performance_fairness_tradeoff.png`

---

## Multi-Run Experiments (100 runs)

For statistical validation with n=100 independent runs:

### 1. Launch Array Job

```bash
# Submit array job (100 runs, 4 models each = 400 total experiments)
sbatch fairness/slurm/run_multi_experiments.slurm
```

This will:
- Run 100 independent experiments (different seeds)
- Train all 4 models in each run
- Save results to `fairness/results/multi_run_<JOB_ID>/run_<N>/`
- Automatically evaluate run_100 when complete

**Estimated time:** ~24 hours per run (adjust `--time` in SLURM script)

### 2. Monitor Progress

```bash
# Check job status
squeue -u $USER

# View logs
tail -f fairness/logs/fair_cbm_<JOB_ID>_<ARRAY_ID>.out
tail -f fairness/logs/fair_cbm_<JOB_ID>_<ARRAY_ID>.err

# Check completed runs
ls fairness/results/multi_run_<JOB_ID>/ | grep run_ | wc -l
```

### 3. Analyze Results

Once all runs complete:

```bash
python fairness/scripts/analyze_multi_run_results.py \
    --exp_name multi_run_<JOB_ID> \
    --backbone swin \
    --results_dir fairness/results \
    --n_runs 100
```

**Output:**
- `fairness/results/multi_run_<JOB_ID>/analysis/summary_table.csv`: Mean ± Std for all metrics
- `fairness/results/multi_run_<JOB_ID>/analysis/summary_table.tex`: LaTeX table
- `fairness/results/multi_run_<JOB_ID>/analysis/statistical_tests.csv`: Pairwise t-tests
- `fairness/results/multi_run_<JOB_ID>/analysis/metric_distributions.png`: Violin plots
- `fairness/results/multi_run_<JOB_ID>/analysis/fairness_performance_scatter.png`: Tradeoff curves
- `fairness/results/multi_run_<JOB_ID>/analysis/analysis_report.json`: Complete report

---

## Advanced Configuration

### Hyperparameter Tuning

Adjust fairness constraints for Fair Curriculum CBM:

```bash
# Stronger fairness constraints
python fairness/scripts/train_all_models.py \
    --model_type fair_curriculum_cbm \
    --fairness_lambda 2.0 \
    --adversarial_lambda 1.0 \
    ...

# Weaker fairness constraints
python fairness/scripts/train_all_models.py \
    --model_type fair_curriculum_cbm \
    --fairness_lambda 0.5 \
    --adversarial_lambda 0.25 \
    ...
```

**Key hyperparameters:**
- `--fairness_lambda`: Weight for demographic parity + equalized odds loss (default: 1.0)
- `--adversarial_lambda`: Weight for adversarial discriminator loss (default: 0.5)
- `--lr`: Learning rate (default: 1e-4)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)

### Different Backbones

Test with different architectures:

```bash
# ConvNeXt
python fairness/scripts/train_all_models.py --model_type fair_curriculum_cbm --backbone convnext ...

# Vision Transformer
python fairness/scripts/train_all_models.py --model_type fair_curriculum_cbm --backbone vit ...

# EfficientNet
python fairness/scripts/train_all_models.py --model_type fair_curriculum_cbm --backbone efficientnet ...

# MobileNet (lightweight)
python fairness/scripts/train_all_models.py --model_type fair_curriculum_cbm --backbone mobilenet ...
```

---

## Understanding Results

### Key Metrics

**Standard Performance:**
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall classification accuracy
- **AUC**: Area under ROC curve

**Fairness Metrics:**
- **Demographic Parity**: Max disparity in positive prediction rates across Fitzpatrick types
  - Target: < 0.10 (fair)
- **Equalized Odds (TPR)**: Max disparity in true positive rates
  - Target: < 0.12 (fair)
- **Performance Gap**: Difference between best and worst group F1 scores
  - Target: < 0.15 (fair)
- **Worst-Group F1**: F1 score of the poorest performing Fitzpatrick type
  - Target: > 0.55 (acceptable)

### Expected Results

Based on REVISED_PLAN.md targets:

| Model | F1 | Performance Gap | Worst-Group F1 | Demographic Parity |
|-------|-----|-----------------|----------------|-------------------|
| **Curriculum CBM** | 0.73 | 0.27 | 0.46 | 0.31 |
| **Fair Curriculum CBM** | ≥0.70 | ≤0.15 | ≥0.55 | ≤0.10 |

**Success criteria:**
- Fair Curriculum CBM achieves < 15% performance gap (vs. 27% for Curriculum CBM)
- Fair Curriculum CBM maintains ≥ 70% of Curriculum CBM's F1 score
- Statistically significant improvement in fairness (p < 0.05)

---

## Troubleshooting

### Common Issues

**1. CUDA out of memory:**
```bash
# Reduce batch size
python fairness/scripts/train_all_models.py --batch_size 16 ...
```

**2. Missing data files:**
```bash
# Check data directory
ls -lh data/skincap/
ls -lh data/skincap_concepts.txt

# Verify dataloader works
python -c "from src.data.dataloader import SkinCapDataset; ds = SkinCapDataset('data/skincap', 'train', 'concept'); print(f'Found {len(ds)} samples')"
```

**3. Import errors:**
```bash
# Verify environment
conda activate CBM-env
python -c "import torch; print(torch.__version__)"
python -c "from src.models.minimal_curriculum_cbm import MinimalCurriculumCBM; print('OK')"
python -c "from fairness.src.models.fairness_aware_cbm import FairnessAwareCBM; print('OK')"
```

**4. Model not saving:**
```bash
# Check disk space
df -h

# Check permissions
ls -ld fairness/results/

# Manually create directory
mkdir -p fairness/results/<exp_name>/<model_type>
```

### Testing Components Individually

**Test FairnessAwareCBM model:**
```bash
python fairness/src/models/fairness_aware_cbm.py
```

**Test fairness metrics:**
```bash
python fairness/src/utils/fairness_metrics.py
```

**Test adversarial discriminator:**
```bash
python fairness/src/models/adversarial_discriminator.py
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{fairness_curriculum_cbm_2025,
  title={Fairness-Aware Curriculum Learning for Concept Bottleneck Models in Dermatological Diagnosis},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

---

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
