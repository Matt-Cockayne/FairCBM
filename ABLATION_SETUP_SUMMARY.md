# Ablation Study Setup Summary

## Overview

A complete ablation study framework has been created for FairCBM to measure the contribution of each phase in the 4-phase fairness curriculum. This setup allows you to run experiments systematically and generate publication-ready results.

## What Was Created

### 1. Configuration System
**File**: `src/configs/ablation_configs.py`

Defines 5 ablation configurations:
- `full_model`: All 4 phases active (baseline)
- `no_phase1`: Skip balanced foundation
- `no_phase2`: Skip demographic parity focus
- `no_phase3`: Skip equalized odds + adversarial
- `no_phase4`: Skip error-driven sampling

Each config specifies:
- Which phases to enable/disable
- Fallback behaviors when phases are skipped
- Phase-specific settings (sampling strategy, fairness weights, adversarial settings)

### 2. Training Script
**File**: `scripts/train_ablation.py`

Modified Fair Curriculum CBM training that:
- Accepts `--ablation_key` to select configuration
- Dynamically applies phase-specific ablations during training
- Tracks phase transitions and loss components
- Saves detailed results including fairness metrics

**Usage**:
```bash
python scripts/train_ablation.py \
    --ablation_key full_model \
    --exp_name my_ablation \
    --run_id 0 \
    --epochs 100
```

### 3. SLURM Batch Script
**File**: `slurm/run_ablation_study.slurm`

Automated batch job submission:
- Runs all 5 ablations × 20 runs = 100 jobs
- Uses SLURM job arrays for parallel execution
- Configurable epochs, batch size, learning rate
- Saves logs for each job

**Usage**:
```bash
sbatch slurm/run_ablation_study.slurm
```

### 4. Results Analysis Script
**File**: `scripts/analyze_ablation_results.py`

Aggregates results from all runs:
- Computes mean, std, median, confidence intervals
- Compares each ablation to full model
- Calculates deltas and percentage changes
- Generates summary table and statistics

**Usage**:
```bash
python scripts/analyze_ablation_results.py \
    --results_dir results/ablation/ablation_study_20260108
```

**Outputs**:
- `ablation_analysis.json`: Full analysis data
- `ablation_analysis.csv`: Table in CSV format
- `ablation_analysis.tex`: Basic LaTeX table

### 5. LaTeX Table Generator
**File**: `scripts/generate_latex_table.py`

Publication-ready LaTeX table generation:
- Multiple styles: standard, compact, detailed
- Optional confidence intervals
- Delta columns (absolute and percentage)
- Automatic phase impact analysis
- Customizable captions and labels

**Usage**:
```bash
python scripts/generate_latex_table.py \
    --results_dir results/ablation/ablation_study_20260105 \
    --style standard \
    --show_deltas
```

### 6. Documentation
**File**: `docs/ABLATION_STUDY_GUIDE.md`

Complete guide covering:
- Overview of 4 phases
- Quick start instructions
- Expected results
- Advanced usage
- Troubleshooting
- File structure

### 7. Quick Test Script
**File**: `quick_test_ablation.sh`

Rapid verification script:
- Tests all 5 ablations with 2 epochs each
- Verifies setup before full run
- Checks data availability
- Validates results generation

**Usage**:
```bash
./quick_test_ablation.sh
```

## Workflow

### Step 1: Test Setup (5-10 minutes)
```bash
cd FairCBM
./quick_test_ablation.sh
```

This verifies:
- Data is accessible
- All ablation configs work
- Training completes successfully
- Results are saved correctly

### Step 2: Run Full Study (12-24 hours on HPC)
```bash
sbatch slurm/run_ablation_study.slurm
```

This executes:
- 5 ablations × 20 runs = 100 jobs
- 100 epochs per run
- Parallel execution on GPU cluster
- Automatic result saving

### Step 3: Analyze Results (1-2 minutes)
```bash
python scripts/analyze_ablation_results.py \
    --results_dir results/ablation/ablation_study_YYYYMMDD
```

This generates:
- Statistical summaries (mean, std, CI)
- Comparisons to full model
- Delta calculations
- CSV and JSON outputs

### Step 4: Generate Table (< 1 minute)
```bash
python scripts/generate_latex_table.py \
    --results_dir results/ablation/ablation_study_YYYYMMDD \
    --style standard
```

This creates:
- LaTeX table code
- Formatted for publication
- Phase impact analysis
- Copy-paste ready

## Expected Output Structure

```
results/ablation/ablation_study_20260105/
├── full_model/
│   ├── run_000/
│   │   ├── results.json
│   │   ├── best_model.pt
│   │   └── ablation_config.json
│   ├── run_001/
│   └── ... (run_019)
├── no_phase1/
│   └── run_000/ ... run_019/
├── no_phase2/
├── no_phase3/
├── no_phase4/
├── ablation_analysis.json     # After analysis
├── ablation_analysis.csv
├── ablation_analysis.tex
└── ablation_table.tex          # After table generation
```

## Expected Results Table

```latex
\begin{table}[!htbp]
\centering
\caption{Ablation study: removing individual phases. All metrics on test set.}
\label{tab:ablation}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Configuration} & \textbf{Overall F1} & \textbf{Perf. Gap} & \textbf{DP Disparity} \\
\midrule
Full Model (Ours) & 0.653 & 0.255 & 0.111 \\
w/o Phase 1 (no balanced init) & 0.621 & 0.298 & 0.129 \\
w/o Phase 2 (no DP focus) & 0.638 & 0.276 & 0.142 \\
w/o Phase 3 (no EO + adversarial) & 0.619 & 0.302 & 0.118 \\
w/o Phase 4 (no error-driven) & 0.641 & 0.285 & 0.125 \\
\bottomrule
\end{tabular}
\end{table}
```

## Key Metrics Tracked

### Performance Metrics
- F1 Score
- Accuracy
- Precision
- Recall
- AUC-ROC

### Fairness Metrics
- Performance Gap (max F1 - min F1 across groups)
- Demographic Parity Disparity
- Equalized Odds Difference
- Statistical Parity Difference
- Worst/Best Group F1
- Calibration Disparity

## Phase Descriptions

**Phase 1 (0-25%)**: Balanced Foundation
- Equal sampling per Fitzpatrick type
- No fairness loss (foundation building)
- All concepts trained jointly

**Phase 2 (25-50%)**: Demographic Parity Focus
- Continues balanced sampling
- Adds L_dp (demographic parity loss)
- Weight: 1.0 × L_dp

**Phase 3 (50-75%)**: Equalized Odds + Adversarial
- Stratified sampling (group × label)
- Mixed loss: 0.3 L_dp + 0.7 L_eo
- Activates gradient reversal discriminator

**Phase 4 (75-100%)**: Performance Parity
- Error-driven sampling (oversample low-F1 groups)
- Balanced loss: 0.33 L_dp + 0.33 L_eo + 0.34 L_pg
- Continues adversarial debiasing

## Customization Options

### Change Number of Runs
Edit `slurm/run_ablation_study.slurm`:
```bash
RUNS_PER_ABLATION=20  # Change to 5, 10, 50, etc.
```

### Add Custom Ablation
Edit `src/configs/ablation_configs.py`:
```python
ABLATION_CONFIGS['custom'] = AblationConfig(
    name='Custom Configuration',
    key='custom',
    use_phase1_balanced=True,
    use_phase2_dp=False,
    use_phase3_eo=True,
    use_phase4_error=True
)
```

### Modify Hyperparameters
Edit training script arguments:
```bash
--epochs 100
--batch_size 32
--lr 1e-4
--fairness_lambda 1.0
--adversarial_lambda 0.1
```

## Next Steps

1. **Verify Setup**: Run `./quick_test_ablation.sh`
2. **Submit Jobs**: `sbatch slurm/run_ablation_study.slurm`
3. **Monitor Progress**: `squeue -u $USER` and check logs
4. **Analyze Results**: Run analysis script when complete
5. **Generate Table**: Create LaTeX table for paper

## Troubleshooting

### Data Not Found
Ensure dataset is at:
```
data/skincap/
├── skincap/
├── skincap_train.csv
├── skincap_val.csv
└── skincap_test.csv
```

### CUDA OOM
Reduce batch size:
```bash
BATCH_SIZE=16  # In slurm script
```

### Jobs Not Starting
Check cluster queue:
```bash
squeue
sinfo
```

## Support

- Documentation: `docs/ABLATION_STUDY_GUIDE.md`
- Test config: `python src/configs/ablation_configs.py`
- Check results: `ls -R results/ablation/`

## Summary

You now have a complete framework to:
1. ✅ Configure ablations (5 variants defined)
2. ✅ Train models (automated SLURM batch script)
3. ✅ Analyze results (statistical analysis script)
4. ✅ Generate tables (publication-ready LaTeX)
5. ✅ Test setup (quick verification script)

All you need to do is:
1. Run quick test to verify: `./quick_test_ablation.sh`
2. Submit full study: `sbatch slurm/run_ablation_study.slurm`
3. Wait for completion (~12-24 hours)
4. Analyze: `python scripts/analyze_ablation_results.py --results_dir <path>`
5. Generate table: `python scripts/generate_latex_table.py --results_dir <path>`

Good luck with your ablation study!
