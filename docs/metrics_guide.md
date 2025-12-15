# Fairness Metrics Guide

## Overview

Definitions and implementation for fairness metrics in FairCBM. Covers group fairness metrics and calibration.

## Table of Contents

1. [Standard Performance Metrics](#standard-performance-metrics)
2. [Group Fairness Metrics](#group-fairness-metrics)
3. [Calibration Metrics](#calibration-metrics)
4. [Interpretation Guidelines](#interpretation-guidelines)
5. [Implementation Details](#implementation-details)

---

## Standard Performance Metrics

### Overall Metrics

#### F1 Score
**Definition:** Harmonic mean of precision and recall

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Range:** [0, 1], higher is better

**Interpretation:**
- F1 ≥ 0.75: Excellent performance
- F1 ≥ 0.70: Good performance (acceptable for medical diagnosis)
- F1 < 0.60: Poor performance

#### Accuracy
**Definition:** Proportion of correct predictions

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Note:** Not used as primary metric (dataset imbalanced)

#### Precision (Positive Predictive Value)
**Definition:** Proportion of positive predictions that are correct

```
Precision = TP / (TP + FP)
```

#### Recall (Sensitivity, True Positive Rate)
**Definition:** Proportion of actual positives correctly identified

```
Recall = TP / (TP + FN)
```

#### Specificity (True Negative Rate)
**Definition:** Proportion of actual negatives correctly identified

```
Specificity = TN / (TN + FP)
```



### Group-Stratified Metrics

Standard metrics computed per Fitzpatrick group (1-6).

**Example:**
```python
F1_by_group = {
    1: 0.75,  # Type I (lightest)
    2: 0.73,
    3: 0.72,
    4: 0.68,  # Performance drops
    5: 0.65,
    6: 0.62   # Type VI (darkest) - worst
}
```

**Key Statistics:**
- **Best Group F1:** max(F1_by_group) = 0.75
- **Worst Group F1:** min(F1_by_group) = 0.62
- **Performance Gap:** 0.75 - 0.62 = 0.13

---

## Group Fairness Metrics

### 1. Performance Gap

**Definition:** Difference between best and worst group performance

```
Performance_Gap = max_a F1(a) - min_a F1(a)
```

**Range:** [0, 1], lower is better

**Fairness Criterion:**
- **Target:** < 0.15 (excellent)
- **Acceptable:** < 0.20 (good)
- **Problematic:** ≥ 0.25 (poor)

**Interpretation:**
- Gap = 0.10: Very fair (e.g., all groups between 0.70-0.80)
- Gap = 0.20: Moderate disparity
- Gap = 0.30+: Severe disparity (some groups underserved)

**Example:**
```
Direct Model:
  Type I: F1 = 0.78
  Type VI: F1 = 0.45
  Gap = 0.33 → PROBLEMATIC ❌

Fair Curriculum Model:
  Type I: F1 = 0.73
  Type VI: F1 = 0.58
  Gap = 0.15 → ACCEPTABLE ✓
```

### 2. Demographic Parity

**Definition:** Equality of positive prediction rates across groups

```
Demographic_Parity = max_a,a' |P(Ŷ=1|A=a) - P(Ŷ=1|A=a')|
```

**Alternative Names:** Statistical Parity, Independence

**Range:** [0, 1], lower is better

**Fairness Criterion:**
- **Target:** < 0.10 (excellent fairness)
- **Acceptable:** < 0.15 (moderate fairness)
- **Problematic:** ≥ 0.20 (poor fairness)

**Interpretation:**
- DP = 0.05: Groups have very similar positive rates (fair)
- DP = 0.20: One group receives 20% more positive predictions
- DP = 0.40: Severe disparate impact

**Example:**
```
Type I: 35% predicted malignant
Type IV: 15% predicted malignant
DP = |0.35 - 0.15| = 0.20
```

**Note:** May be inappropriate if base rates genuinely differ across groups

### 3. Equalized Odds

**Definition:** Equality of TPR and FPR across groups

```
Equalized_Odds = max(EOD_TPR, EOD_FPR)

where:
  EOD_TPR = max_a,a' |TPR(a) - TPR(a')|
  EOD_FPR = max_a,a' |FPR(a) - FPR(a')|
```

**Range:** [0, 1], lower is better

**Fairness Criterion:**
- **Target:** < 0.10 (excellent)
- **Acceptable:** < 0.15 (good)
- **Problematic:** ≥ 0.20 (poor)

**Components:**

#### TPR Disparity (Equality of Opportunity)
```
TPR = TP / (TP + FN) = Recall for positive class
```

**Example:**
```
Type I: TPR = 0.85
Type VI: TPR = 0.65
Disparity = 0.20
```

#### FPR Disparity
```
FPR = FP / (FP + TN)
```

**Medical Meaning:** "Are false alarms equally distributed across skin types?"

**Example:**
```
Type I: FPR = 0.10
Type VI: FPR = 0.30
Disparity = 0.20
```

### 4. Equalized Opportunity (TPR Parity Only)

**Definition:** Focuses only on equalizing true positive rates

```
Equalized_Opportunity = max_a,a' |TPR(a) - TPR(a')|
```

**Note:** Only considers TPR (ignores FPR). Appropriate when false negatives are more costly.

### 5. Predictive Parity

**Definition:** Equality of positive predictive value (precision) across groups

```
Predictive_Parity = max_a,a' |PPV(a) - PPV(a')|

where PPV = Precision = TP / (TP + FP)
```

**Range:** [0, 1], lower is better

**Example:**
```
Type I: PPV = 0.80
Type VI: PPV = 0.60
Disparity = 0.20
```

**Note:** Can conflict with equalized odds when base rates differ

### 6. Calibration Disparity

**Definition:** Difference in calibration error across groups

```
Calibration_Disparity = max_a,a' |ECE(a) - ECE(a')|

where ECE = Expected Calibration Error (see below)
```

**Range:** [0, 1], lower is better

**Target:** < 0.05

**Meaning:** Predicted probabilities should be equally reliable across groups

---

## Calibration Metrics

### Expected Calibration Error (ECE)

**Definition:** Average difference between predicted probability and actual frequency

```
ECE = Σ_b (|B_b| / N) × |acc(B_b) - conf(B_b)|

where:
  B_b: predictions in bin b (typically 10 bins)
  acc(B_b): actual accuracy in bin b
  conf(B_b): average predicted confidence in bin b
```

**Range:** [0, 1], lower is better

**Interpretation:**
- ECE < 0.05: Well-calibrated
- ECE < 0.10: Acceptable
- ECE ≥ 0.15: Poorly calibrated

**Example:**
```
Bin: [0.7-0.8]
  Model confidence (avg): 75%
  Actual accuracy: 85%
  Contribution: |0.85 - 0.75| = 0.10
```

### Calibration by Group

ECE computed separately for each Fitzpatrick type:

```python
ECE_by_group = {
    1: 0.04,
    2: 0.06,
    3: 0.08,
    4: 0.12,
    5: 0.15,
    6: 0.18
}
```

Calibration Disparity = 0.18 - 0.04 = 0.14

---

## Interpretation Guidelines

### Fairness vs. Performance Tradeoff

Perfect fairness and perfect accuracy are generally incompatible.

**Example Scenario:**
```
Ground Truth Base Rates:
  Type I: 40% malignant
  Type VI: 25% malignant

Optimal Unfair Model:
  Type I: Predict 40% positive → F1 = 0.85
  Type VI: Predict 25% positive → F1 = 0.82
  Demographic Parity = |0.40 - 0.25| = 0.15
  Performance Gap = 0.03

Fair Model (Enforced DP < 0.10):
  Both groups: Predict ~32.5% positive
  Type I: F1 = 0.81 (slight decrease)
  Type VI: F1 = 0.78 (slight decrease)
  Demographic Parity = 0.05 ✓
  Performance Gap = 0.03

Result: Small accuracy loss (4-5%) for significant fairness gain
```

### Success Criteria

FairCBM considers a model fair if it satisfies **all** of:

1. **Performance Gap < 0.15**
2. **Worst-Group F1 > 0.55** (no group severely underserved)
3. **Overall F1 ≥ 0.70** (maintains clinical utility)
4. **Demographic Parity < 0.10**
5. **Equalized Odds < 0.15**

### Prioritization Hierarchy

When metrics conflict, prioritize:

1. **Equalized Odds** (ensures equal benefit and burden)
2. **Performance Gap** (no group left behind)
3. **Demographic Parity** (equal treatment rates)
4. **Calibration** (trustworthy probabilities)

**Rationale:** Medical context prioritizes detecting disease equally across groups

### Statistical Significance

All metrics should be validated with:
- **Confidence Intervals:** Bootstrap with 1,000+ iterations
- **Hypothesis Tests:** Paired t-tests for model comparisons
- **Effect Sizes:** Cohen's d for practical significance

**Reporting:**
```
Fair Curriculum CBM:
  Performance Gap: 0.15 ± 0.03 (95% CI)
  Significantly better than Curriculum CBM: p < 0.001
  Effect size: d = 0.82 (large)
```

---

## Implementation Details

### Computing Metrics Per Group

```python
def compute_group_metrics(predictions, labels, groups):
    """
    predictions: (N,) array of probabilities
    labels: (N,) array of binary labels
    groups: (N,) array of group IDs (0-5 for Fitzpatrick 1-6)
    """
    metrics_by_group = {}
    
    for group_id in range(6):
        mask = (groups == group_id)
        group_preds = predictions[mask]
        group_labels = labels[mask]
        
        # Standard metrics
        f1 = compute_f1(group_labels, group_preds > 0.5)
        tpr = compute_tpr(group_labels, group_preds > 0.5)
        fpr = compute_fpr(group_labels, group_preds > 0.5)
        
        # Calibration
        ece = compute_ece(group_labels, group_preds)
        
        metrics_by_group[group_id] = {
            'f1': f1,
            'tpr': tpr,
            'fpr': fpr,
            'ece': ece,
            'n_samples': mask.sum()
        }
    
    return metrics_by_group
```

### Computing Fairness Metrics

```python
def compute_fairness_metrics(metrics_by_group):
    """Aggregate group metrics into fairness scores."""
    
    # Performance Gap
    f1_scores = [m['f1'] for m in metrics_by_group.values()]
    performance_gap = max(f1_scores) - min(f1_scores)
    
    # Demographic Parity
    positive_rates = [m['positive_rate'] for m in metrics_by_group.values()]
    demographic_parity = max(positive_rates) - min(positive_rates)
    
    # Equalized Odds (max of TPR and FPR disparities)
    tpr_values = [m['tpr'] for m in metrics_by_group.values()]
    fpr_values = [m['fpr'] for m in metrics_by_group.values()]
    tpr_disparity = max(tpr_values) - min(tpr_values)
    fpr_disparity = max(fpr_values) - min(fpr_values)
    equalized_odds = max(tpr_disparity, fpr_disparity)
    
    # Calibration Disparity
    ece_values = [m['ece'] for m in metrics_by_group.values()]
    calibration_disparity = max(ece_values) - min(ece_values)
    
    return {
        'performance_gap': performance_gap,
        'demographic_parity': demographic_parity,
        'equalized_odds': equalized_odds,
        'tpr_disparity': tpr_disparity,
        'fpr_disparity': fpr_disparity,
        'calibration_disparity': calibration_disparity,
        'worst_group_f1': min(f1_scores),
        'best_group_f1': max(f1_scores)
    }
```

### Bootstrap Confidence Intervals

```python
def bootstrap_fairness_ci(predictions, labels, groups, n_bootstrap=1000):
    """Compute 95% CI for fairness metrics via bootstrap."""
    
    metric_samples = []
    n_samples = len(predictions)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        boot_preds = predictions[indices]
        boot_labels = labels[indices]
        boot_groups = groups[indices]
        
        # Compute metrics
        metrics = compute_all_fairness_metrics(boot_preds, boot_labels, boot_groups)
        metric_samples.append(metrics)
    
    # Compute percentiles
    ci_lower = np.percentile(metric_samples, 2.5, axis=0)
    ci_upper = np.percentile(metric_samples, 97.5, axis=0)
    
    return ci_lower, ci_upper
```

---

## Visualizations

### 1. Group Performance
Bar chart: F1 by Fitzpatrick type with confidence intervals

### 2. Fairness Radar
Axes: Performance Gap, Demographic Parity, Equalized Odds, Calibration

### 3. Performance-Fairness Tradeoff
Scatter: Overall F1 vs. Performance Gap (target: top-left)

### 4. Calibration Curves
Predicted probability vs. actual frequency per group

---

## Common Pitfalls

### 1. Fairness Gerrymandering
**Issue:** Optimizing one fairness metric can worsen others

**Example:** Achieving demographic parity by harming both groups equally

**Solution:** Monitor multiple metrics simultaneously

### 2. Base Rate Fallacy
**Issue:** Ignoring genuine differences in disease prevalence

**Example:** If Type I has 40% malignancy and Type VI has 20%, enforcing equal positive rates may be inappropriate

**Solution:** Consider equalized odds (conditional on true label) instead

### 3. Sample Size Imbalance
**Issue:** Small groups have high variance
**Solution:** Report sample sizes, use stratified bootstrapping

### 4. Multiple Testing
**Issue:** Testing many metrics inflates false positive rate
**Solution:** Bonferroni correction or pre-specify primary metric

---

## References

1. **Hardt et al. (2016):** "Equality of Opportunity in Supervised Learning"
2. **Chouldechova (2017):** "Fair Prediction with Disparate Impact"
3. **Guo et al. (2017):** "On Calibration of Modern Neural Networks"
4. **Aequitas Toolkit (2018):** Comprehensive fairness auditing framework
5. **Groh et al. (2021):** "Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology"

---

*For implementation, see `src/utils/fairness_metrics.py`*
