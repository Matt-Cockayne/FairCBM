# FairCBM: Fairness-Aware Curriculum for Concept Bottleneck Models

## Overview

Fairness-aware curriculum learning for Concept Bottleneck Models to mitigate performance disparities across Fitzpatrick skin types.

**Key Features:**
- In-training fairness via combined loss + adversarial debiasing
- 3-phase warmup scheduler for training stability
- Group fairness metrics (demographic parity, equalized odds, calibration)
- Interpretable via 23 morphological concepts
- Multi-run validation with statistical tests

## Project Structure

```
FairCBM/
├── src/                      # Core library
│   ├── models/              # Model implementations
│   │   ├── fairness_aware_cbm.py      # Fair Curriculum CBM (main contribution)
│   │   ├── minimal_curriculum_cbm.py  # Base curriculum CBM
│   │   └── adversarial_discriminator.py  # Gradient reversal discriminator
│   ├── utils/               # Utilities
│   │   ├── fairness_metrics.py        # 6 fairness metrics + Aequitas
│   │   └── adversarial_debiasing.py   # Fairness loss functions
│   └── data/                # Data loading
│       └── dataloader.py    # SkinCap dataset with Fitzpatrick labels
├── scripts/                 # Executable scripts
│   ├── train_all_models.py             # Train all 4 model types
│   ├── evaluate_fairness_comparison.py # Compare fairness metrics
│   └── analyze_multi_run_results.py    # Aggregate 100-run statistics
├── slurm/                   # HPC batch scripts
│   ├── run_single_experiment.slurm     # Single test run
│   └── run_multi_experiments.slurm     # 100-run array job
├── docs/                    # Documentation
├── data/                    # Dataset location (user-provided)
├── results/                 # Experiment outputs
├── logs/                    # Training logs
└── [README, requirements, etc.]
```

## Quick Start

### 1. Create Conda Environment

```bash
conda create -n CBM-env python=3.10 -y
conda activate CBM-env
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch ≥ 2.0
- torchvision
- numpy, pandas, scipy
- scikit-learn
- matplotlib, seaborn
- tqdm

### 3. Prepare Data

The SkinCap dataset should be organized as:
```
data/skincap/
├── skincap/              # Image directory
│   ├── <image_id>.png
│   └── ...
├── skincap_train.csv     # Training split with Fitzpatrick labels
├── skincap_val.csv       # Validation split
└── skincap_test.csv      # Test split
```

**CSV Format:** Each CSV must contain:
- `skincap_file_path`: Image filename
- `malignant`: Binary label (0=benign, 1=malignant)
- `fitzpatrick_scale`: Skin type (1-6)
- 23 concept columns (Papule, Plaque, Nodule, etc.)

### 4. Quick Test (20 epochs)

```bash
./quick_test.sh fair_curriculum_cbm 20
```

### 5. Train Single Model

```bash
python scripts/train_all_models.py \
    --model_type fair_curriculum_cbm \
    --backbone swin \
    --exp_name test_run \
    --epochs 100 \
    --batch_size 32 \
    --fairness_lambda 1.0 \
    --adversarial_lambda 0.5 \
    --data_root data/skincap \
    --save_dir results \
    --eval_every 10 \
    --save_best
```

### 6. Train All 5 Models for Comparison

```bash
# Train all 5 models for comparison
for model in direct standard_cbm curriculum_cbm fair_cbm fair_curriculum_cbm; do
    python scripts/train_all_models.py \
        --model_type $model \
        --backbone swin \
        --exp_name comparison \
        --epochs 100 \
        --fairness_lambda 0.1 \
        --adversarial_lambda 0.01 \
        --data_root /home/csc29/projects/SkinCAP \
        --raw_csv /home/csc29/projects/SkinCAP/skincap_v240623.csv \
        --save_dir results
done
```

### 7. Evaluate and Compare

```bash
python scripts/evaluate_fairness_comparison.py \
    --exp_name comparison \
    --backbone swin \
    --results_dir results \
    --data_root data/skincap \
    --n_bootstrap 1000
```

**Output:**
- `results/comparison/comparison/comparison_table.csv` - Metric comparison
- `results/comparison/comparison/per_group_performance.png` - Group F1 scores
- `results/comparison/comparison/fairness_metrics_comparison.png` - Fairness comparison
- `results/comparison/comparison/performance_fairness_tradeoff.png` - Tradeoff scatter

## Model Types

FairCBM provides 4 model variants for comprehensive comparison:

| Model | Interpretable | Curriculum | Fairness | Purpose |
|-------|---------------|------------|----------|---------|
| **Direct** | No | No | No | Baseline (no interpretability) |
| **Standard CBM** | Yes | No | No | Baseline (all concepts jointly) |
| **Curriculum CBM** | Yes | Yes | No | Baseline (3-phase concept curriculum) |
| **Fair CBM** | Yes | No | Yes | Fairness without concept curriculum |
| **Fair Curriculum CBM** | Yes | Yes | Yes | **Full model (curriculum + fairness)** |

## Key Hyperparameters

**Model Selection:**
- `--model_type`: Choose from {direct, standard_cbm, curriculum_cbm, fair_curriculum_cbm}
- `--backbone`: Pretrained encoder {swin, convnext, vit, efficientnet, mobilenet}

**Fairness (Fair Curriculum CBM only):**
- `--fairness_lambda`: Weight for demographic parity + equalized odds loss (default: 0.1)
- `--adversarial_lambda`: Weight for adversarial discriminator loss (default: 0.01)
- `--adversarial_warmup_epochs`: Duration of warmup phase in epochs (default: 30% of total epochs)

**Warmup Scheduling:** 3-phase schedule prevents training instability:
  - **Phase 1 (0-20%):** λ=0.0 (learn task)
  - **Phase 2 (20-50%):** λ: 0.0→target (linear warmup)
  - **Phase 3 (50-100%):** λ=target (full fairness)

## In-Training Fairness

Dual-component approach:

### 1. Combined Fairness Loss
- **Demographic Parity:** `|P(Ŷ=1|A=a) - P(Ŷ=1|A=a')|`
- **Equalized Odds:** `|P(Ŷ=1|Y=y,A=a) - P(Ŷ=1|Y=y,A=a')|` for y ∈ {0,1}

### 2. Adversarial Discriminator
Gradient reversal layer makes concept representations group-invariant:
- Discriminator predicts Fitzpatrick type from concepts
- Reversed gradients prevent encoder from encoding group information

### 3. Warmup Scheduling
3-phase schedule prevents training instability:
- **Phase 1 (0-20%):** λ_adv = 0.0 (learn task)
- **Phase 2 (20-50%):** λ_adv: 0.0 → target (gradual)
- **Phase 3 (50-100%):** λ_adv = target (full fairness)

Prevents adversarial loss explosion and performance collapse.

### Combined Objective
```
Total Loss = L_concept + L_binary + λ_fair × L_fairness + λ_adv(t) × L_adversarial
```
where λ_adv(t) follows the warmup schedule.

## Fairness Evaluation**
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--eval_every`: Evaluate every N epochs (default: 5)

## Fairness Metrics

### Demographic Parity
P(Ŷ=1 | A=a) should be equal across groups
- **Metric:** Statistical Parity Difference (SPD)
- **Target:** SPD < 0.10

### Equalized Odds
P(Ŷ=1 | Y=y, A=a) should be equal across groups for y ∈ {0, 1}
- **Metric:** Equalized Odds Difference (EOD)
- **Target:** EOD < 0.10

### Performance Parity
- **Metric:** Performance Gap (max F1 - min F1 across groups)
- **Target:** Gap < 0.10

### Calibration
- **Metric:** Calibration Disparity (max ECE - min ECE across groups)
- **Target:** Disparity < 0.05

## Multi-Run Experiments (Statistical Validation)

### Single Test Run

```bash
sbatch slurm/run_single_experiment.slurm fair_curriculum_cbm swin 42
```

### Full 100-Run Validation

```bash
# Default: 100 runs
sbatch slurm/run_multi_experiments.slurm

# Custom number of runs (e.g., 20 for faster testing)
sbatch --array=1-20 slurm/run_multi_experiments.slurm

# Set N_RUNS environment variable (alternative method)
N_RUNS=20 sbatch --array=1-20 slurm/run_multi_experiments.slurm
```

Runs N independent experiments with different seeds (default N=100). Trains all 5 models per run (N×5 total experiments). Estimated time: ~24 hours per run.

### Analyze Aggregated Results

```bash
python scripts/analyze_multi_run_results.py \
    --exp_name multi_run_<JOB_ID> \
    --backbone swin \
    --n_runs 100
```

**Output:**
- `results/multi_run_<JOB_ID>/analysis/summary_table.csv` - Mean ± Std for all metrics
- `results/multi_run_<JOB_ID>/analysis/summary_table.tex` - LaTeX table
- `results/multi_run_<JOB_ID>/analysis/statistical_tests.csv` - Pairwise t-tests
- `results/multi_run_<JOB_ID>/analysis/metric_distributions.png` - Violin plots
- `results/multi_run_<JOB_ID>/analysis/*.png` - Various visualizations

## Expected Results

Comparison of all 4 models on test set:

| Model | Overall F1 | Worst-Group F1 | Performance Gap | Demographic Parity |
|-------|-----------|----------------|-----------------|-------------------|
| Direct | 0.75 | 0.42 | 0.33 | 0.28 |
| Standard CBM | 0.72 | 0.45 | 0.27 | 0.25 |
| Curriculum CBM | 0.73 | 0.46 | 0.27 | 0.24 |
| **Fair Curriculum** | **0.71** | **0.56** | **0.15** | **0.10** |

**Key Improvements (Fair Curriculum vs. Curriculum):**
- Worst-group F1: +22% (0.46 → 0.56)
- Performance Gap: -44% (0.27 → 0.15)
- Demographic Parity: -58% (0.24 → 0.10)
- Overall F1: -3% (0.73 → 0.71)

**Success Criteria:**
1. Performance gap < 0.15
2. Worst-group F1 > 0.55
3. Overall F1 ≥ 0.70
4. Demographic parity < 0.10
5. Statistically significant (p < 0.05)

## Documentation

- **[Fairness Methodology](docs/fairness_methodology.md)**: Theoretical foundation
- **[Metrics Guide](docs/metrics_guide.md)**: Fairness metric definitions
- **[Experimental Design](docs/experimental_design.md)**: Protocol & validation

## Citation

If you use this code, please cite:

```bibtex
@software{faircbm2024,
  title={FairCBM: Fairness-Aware Concept Bottleneck Models with Curriculum Learning},
  author={Cockayne, Matthew},
  year={2024},
  url={https://github.com/Matt-Cockayne/FairCBM},
  note={Curriculum learning with adversarial debiasing for fair dermatology diagnosis}
}
```

## Related Work

- Concept Bottleneck Models (Koh et al., 2020)
- Curriculum Learning (Bengio et al., 2009)
- Domain-Adversarial Training (Ganin & Lempitsky, 2015)
- Aequitas Toolkit (Saleiro et al., 2018)

## Contact

**Author:** Matthew J. Cockayne  
**Repository:** https://github.com/Matt-Cockayne/FairCBM
**Issues:** https://github.com/Matt-Cockayne/FairCBM/issues

---

*Last updated: December 12, 2025*
