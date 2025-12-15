# Experimental Design and Validation Protocol

## Overview

Experimental protocol for validating FairCBM. Includes statistical tests, multi-run validation, and reproducibility guidelines.

## Table of Contents

1. [Research Questions](#research-questions)
2. [Experimental Design](#experimental-design)
3. [Data Protocol](#data-protocol)
4. [Model Configurations](#model-configurations)
5. [Evaluation Protocol](#evaluation-protocol)
6. [Statistical Validation](#statistical-validation)
7. [Reproducibility](#reproducibility)

---

## Research Questions

### Primary Research Question (RQ1)

**Does Fair Curriculum CBM reduce performance disparities across Fitzpatrick skin types compared to baseline CBM models?**

**Hypothesis:** Fair Curriculum CBM will achieve:
- ≥30% reduction in performance gap vs. Curriculum CBM (baseline)
- Performance gap < 0.15
- Worst-group F1 > 0.55
- Overall F1 ≥ 0.70

**Null Hypothesis:** No significant difference in performance gap between models

### Secondary Research Questions

**RQ2:** Does fairness enforcement reduce overall model performance?
- **Hypothesis:** Overall F1 decrease < 5% compared to Curriculum CBM

**RQ3:** Which fairness component (combined loss vs. adversarial) contributes more?
- **Approach:** Ablation study comparing Fair CBM vs. Fair Curriculum CBM

**RQ4:** Is the fairness-accuracy tradeoff efficient (Pareto optimal)?
- **Approach:** Compare to post-processing fairness methods

**RQ5:** Do results generalize across different backbones?
- **Approach:** Test on 5 architectures: Swin-T, ConvNeXt, ViT, EfficientNet, MobileNet

---

## Experimental Design

### Overall Design: Within-Subject Repeated Measures

**Rationale:** 
- Same data, hyperparameters, and training procedure across all models
- Only difference: fairness components (direct, standard_cbm, curriculum_cbm, fair_cbm, fair_curriculum_cbm)
- Eliminates confounds from data splits or implementation differences

### Model Variants (5 Total)

| Model ID | Type | Interpretable | Curriculum | Fairness | Purpose |
|----------|------|---------------|------------|----------|---------|
| M1 | Direct | No | No | No | Non-interpretable baseline |
| M2 | Standard CBM | Yes | No | No | Interpretable baseline |
| M3 | Curriculum CBM | Yes | Yes | No | Curriculum baseline |
| M4 | Fair CBM | Yes | No | Yes | Fairness w/o curriculum |
| M5 | **Fair Curriculum CBM** | Yes | Yes | Yes | **Full model (ours)** |

**Key Comparisons:**
- **M5 vs. M3:** Effect of adding fairness to curriculum learning
- **M5 vs. M4:** Effect of adding curriculum to fairness
- **M5 vs. M1/M2:** Overall improvement over baselines

### Backbone Architectures (5 Total)

Test generalization across diverse architectures:

1. **Swin Transformer (Swin-T):** Hierarchical vision transformer
2. **ConvNeXt-Tiny:** Modern ConvNet with transformer design principles
3. **ViT-B/16:** Pure transformer (vision)
4. **EfficientNet-B0:** Efficient compound scaling
5. **MobileNet-V3-Large:** Mobile-optimized architecture

**Rationale:** 
- Covers transformers, ConvNets, and efficient architectures
- Tests if fairness approach is architecture-agnostic
- Total: 5 models × 5 backbones = 25 configurations per run

### Replication: 100 Independent Runs

**Why 100 Runs?**
- Statistical power: Detect small effect sizes (Cohen's d ≥ 0.3)
- Confidence: 95% CI width ≈ 0.2 × std (narrow)
- Robustness: Identify outliers and variance sources

**Randomization:**
- Different random seeds (1-100)
- Independent data splits per run (stratified by Fitzpatrick and label)
- Independent weight initialization

**Computational Cost:**
- 100 runs × 5 models = 500 experiments
- ~30 min per model × 500 = ~250 GPU hours
- ~24 hours wall-clock time with parallel execution

---

## Data Protocol

### Dataset: SkinCap

2,848 dermatological images with 23 morphological concepts, binary malignancy label, and Fitzpatrick skin type (1-6).

**Class Distribution:** Malignant ~28% (imbalanced), Fitzpatrick types I/II overrepresented.

### Data Splits

Stratified 80/10/10 split preserving malignancy rate and Fitzpatrick distribution.
- Train: 2,278 images
- Val: 285 images  
- Test: 285 images

Unique seed per run (1-100).

**Verification:**
```python
# Check stratification
for split in ['train', 'val', 'test']:
    for fitz_type in range(1, 7):
        ratio = split_data[fitz_type].mean()
        assert abs(ratio - overall_data[fitz_type].mean()) < 0.02
```

### Data Preprocessing

Resize to 224×224, ImageNet normalization. No augmentation.

**Label Processing:**
- Concepts: Multi-hot vector (23 binary features)
- Binary: Single binary label
- Fitzpatrick: Integer 1-6 (convert to 0-5 for indexing)

### Exclusion Criteria

Remove samples with:
- Missing Fitzpatrick labels (cannot compute fairness)
- Missing malignancy labels
- "Do not consider" flag = 1

**Final:** 2,848 images after exclusions

---

## Model Configurations

### Shared Hyperparameters (All Models)

**Critical:** Identical across all models for fair comparison

```python
# Training
epochs: 100
batch_size: 32
learning_rate: 1e-4
weight_decay: 1e-4
optimizer: Adam(betas=(0.9, 0.999))

# Backbone
pretrained: True (ImageNet-1K)
freeze_backbone: False (full fine-tuning)

# Concept bottleneck
concept_dim: 23
hidden_dim: 512

# Evaluation
eval_every: 5 epochs
save_best: True (based on Val F1)
```

### Fairness Hyperparameters (M4, M5 Only)

**Combined Fairness Loss:**
```python
fairness_lambda: 0.1
demographic_parity_weight: 0.5
equalized_odds_weight: 0.5
```

**Adversarial Debiasing:**
```python
adversarial_lambda: 0.01 (target)
discriminator_hidden: [512, 256, 128]
discriminator_dropout: 0.3
gradient_reversal_alpha: 2.0 (at convergence)
```

**Warmup Schedule:**
```python
warmup_start: 20% of epochs (epoch 20)
warmup_end: 50% of epochs (epoch 50)
warmup_type: 'linear'
```

**Stability Fixes:**
```python
gradient_clipping: 1.0 (max_norm)
label_smoothing: 0.1 (adversarial loss)
loss_capping: 10.0 (adversarial loss max)
```

### Curriculum Schedule (M3, M5 Only)

**3-Phase Curriculum:**
- Phase 1 (0-33%): Easy concepts (visual textures)
- Phase 2 (33-66%): Medium concepts
- Phase 3 (66-100%): Hard concepts
- Binary task: full weight throughout

---

## Evaluation Protocol

### Evaluation Stages

**1. During Training:** Val F1 and fairness metrics every 5 epochs. Save best model. Early stopping patience = 20.

**2. Test Evaluation:** Load best checkpoint, evaluate on test set.

**3. Multi-Run Aggregation:** Compute mean, std, 95% CI across 100 runs.
- Perform statistical tests

### Metrics Computed

**Per Model, Per Run:**

**Performance:**
- Overall F1, Precision, Recall, Accuracy
- F1 per Fitzpatrick group (6 values)
- Worst-group F1, Best-group F1

**Fairness:**
- Performance Gap (primary fairness metric)
- Demographic Parity
- Equalized Odds (TPR & FPR disparity)
- Calibration Disparity (ECE)

**Total:** ~20 metrics per model per run

### Reporting Standards

**Point Estimates:** Mean across 100 runs

**Uncertainty:** 95% confidence intervals (bootstrap)

**Format:**
```
Performance Gap: 0.15 ± 0.03 (95% CI: [0.12, 0.18])
```

**Significance:** p-value and effect size (Cohen's d)

**Format:**
```
Fair Curriculum CBM vs. Curriculum CBM:
  Δ Performance Gap: -0.12 (44% reduction)
  p < 0.001 (paired t-test)
  Cohen's d = 0.82 (large effect)
```

---

## Statistical Validation

### Power Analysis

**Goal:** Detect meaningful differences between models

**Assumptions:**
- Effect size: Cohen's d ≥ 0.3 (small-medium)
- Significance level: α = 0.05
- Power: 1 - β ≥ 0.80

**Result:** N ≥ 90 runs required → Use 100 for safety

### Hypothesis Testing

**Primary Test:** Paired t-test (within-subject design)

**Null Hypothesis:**
```
H0: μ(Gap_FairCurriculum) = μ(Gap_Curriculum)
```

**Alternative:**
```
H1: μ(Gap_FairCurriculum) < μ(Gap_Curriculum)  (one-tailed)
```

**Correction:** Bonferroni for multiple comparisons
- Comparing 5 models → 10 pairwise tests
- Adjusted α = 0.05 / 10 = 0.005

### Effect Size

**Cohen's d:** Standardized mean difference

```
d = (μ1 - μ2) / σ_pooled
```

**Interpretation:**
- |d| ≥ 0.2: Small effect
- |d| ≥ 0.5: Medium effect
- |d| ≥ 0.8: Large effect

**Requirement:** d ≥ 0.5 for practical significance

### Bootstrap Confidence Intervals

Percentile bootstrap with 1,000 iterations. Resample runs, extract 2.5th and 97.5th percentiles for non-parametric 95% CI.

### Robustness Checks

1. **Outlier Detection:** Identify runs >3 SDs from mean
2. **Variance Homogeneity:** Levene's test (use Welch's t-test if violated)
3. **Normality:** Shapiro-Wilk test (use Mann-Whitney U if violated)
4. **Model Selection:** Bootstrap validation set selection

---

## Reproducibility

### Code and Environment

**Version Control:**
```bash
git commit hash: <SHA>
Python: 3.10.x
PyTorch: 2.5.1
CUDA: 11.8
```

**Dependencies:** Pinned in `requirements.txt`
```
torch==2.5.1
torchvision==0.20.1
numpy==1.24.3
scikit-learn==1.3.0
...
```

**Environment:**
```bash
conda create -n CBM-env python=3.10
conda activate CBM-env
pip install -r requirements.txt
```

### Random Seeds

**Seed Hierarchy:**
```python
# Global seed per run (1-100)
run_seed = args.seed

# Python random
random.seed(run_seed)

# NumPy
np.random.seed(run_seed)

# PyTorch
torch.manual_seed(run_seed)
torch.cuda.manual_seed(run_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# DataLoader
generator = torch.Generator().manual_seed(run_seed)
DataLoader(..., generator=generator)
```

### Experiment Tracking

**Save Per Run:**
```
results/{exp_name}/{model_type}_{backbone}_{seed}/
  ├── config.json           # All hyperparameters
  ├── history.json          # Training curves
  ├── test_results.json     # Final metrics
  ├── best_model.pt         # Model checkpoint
  └── logs/                 # Stdout/stderr
```

**Aggregate Results:**
```
results/{exp_name}/analysis/
  ├── summary_table.csv     # Mean ± Std for all metrics
  ├── summary_table.tex     # LaTeX formatted
  ├── statistical_tests.csv # Pairwise comparisons
  ├── plots/                # Visualizations
  └── raw_results.pkl       # All 100 runs
```

### Computational Resources

**Hardware:**
- GPUs: NVIDIA A100 (40GB) or V100 (32GB)
- CPUs: 8-16 cores per job
- RAM: 32GB minimum

**Parallelization:**
- SLURM array jobs (100 parallel tasks)
- Each task trains 5 models sequentially
- Total wall-clock time: ~24 hours

**Storage:**
- Model checkpoints: ~500MB each × 500 = ~250GB
- History files: ~1MB each × 500 = ~500MB
- Total: ~250GB for full experiment

**Disk Management:**
- Save only best model per run (not all epochs)
- Compress old experiments after analysis
- Atomic best-model tracking (only 5 models kept across 100 runs)

---

## Validation Checklist

- [ ] All 100 runs completed successfully
- [ ] No outliers due to bugs (check logs)
- [ ] Metrics pass sanity checks (F1 ∈ [0, 1], etc.)
- [ ] Data splits properly stratified (verify counts)
- [ ] Hyperparameters identical across models (except fairness)
- [ ] Random seeds unique and reproducible
- [ ] Statistical tests show p < 0.05 and d ≥ 0.5
- [ ] Confidence intervals do not include null hypothesis
- [ ] Results consistent across backbones
- [ ] Visualizations checked for errors
- [ ] Code and data available for replication

---

## Experiment Timeline

**Week 1:** Infrastructure setup (data splits, implement models, test single run)
- Implement all 5 models
- Test on single run
- Verify metrics computation

**Week 2:** Pilot runs
- 5 runs per configuration (25 total)
- Check convergence and stability
- Tune hyperparameters if needed
- Validate evaluation pipeline

**Week 3-4:** Full multi-run
- Submit 100-run array job
- Monitor progress (daily)
- Debug any failures
- Backup intermediate results

**Week 5:** Analysis
- Aggregate results
- Statistical tests
- Generate visualizations
- Write up findings

**Week 6:** Validation
- Verify reproducibility
- Cross-validate on held-out set
- Prepare for submission

---

## Failure Modes and Mitigation

### Issue: Training Divergence
**Symptom:** Loss NaN or explodes
**Solution:** Reduce learning rate, increase warmup, check data

### Issue: Poor Fairness
**Symptom:** Demographic parity > 0.20
**Solution:** Increase λ_fair or λ_adv, extend warmup, increase discriminator capacity

### Issue: Performance Drop
**Symptom:** F1 drops > 10%
**Solution:** Reduce λ_fair, use soft constraints, or post-hoc adjustment

### Issue: High Variance
**Symptom:** Std > 0.10
**Solution:** Increase runs, use ensembles, or bootstrap aggregation

---

## References

1. **Statistical Power:** Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
2. **Multiple Testing:** Bonferroni, C. (1936). Teoria statistica delle classi e calcolo delle probabilità
3. **Bootstrap:** Efron, B. (1979). Bootstrap methods: Another look at the jackknife
4. **Reproducibility:** Pineau et al. (2021). Improving Reproducibility in Machine Learning Research
5. **Fairness Auditing:** Saleiro et al. (2018). Aequitas: A Bias and Fairness Audit Toolkit

---

*For implementation, see `scripts/train_all_models.py` and `slurm/run_multi_experiments.slurm`*
