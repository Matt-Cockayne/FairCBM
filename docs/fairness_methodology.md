# Fairness Methodology

## Overview

FairCBM enforces group fairness **during training** using two components: combined fairness loss and adversarial debiasing. This document covers the theoretical foundation and implementation.

## Table of Contents

1. [Problem Formulation](#problem-formulation)
2. [Fairness Components](#fairness-components)
3. [Adversarial Debiasing](#adversarial-debiasing)
4. [Warmup Scheduling](#warmup-scheduling)
5. [Theoretical Guarantees](#theoretical-guarantees)
6. [Implementation Details](#implementation-details)

---

## Problem Formulation

### Task Definition

Given:
- **Input:** Dermatological image `x ∈ X`
- **Concepts:** Morphological features `c ∈ {0,1}^K` (K=23)
- **Binary Label:** Malignancy `y ∈ {0,1}`
- **Protected Attribute:** Fitzpatrick skin type `a ∈ {1,2,3,4,5,6}`

**Goal:** Learn a model `f: X → {0,1}` that:
1. Achieves high classification accuracy (F1 score)
2. Provides interpretable concept predictions
3. **Ensures fairness across all Fitzpatrick groups**

### Fairness Criteria

1. **Demographic Parity:** Equal positive prediction rates across groups
   ```
   P(Ŷ=1 | A=a) ≈ P(Ŷ=1 | A=a') for all a, a' ∈ {1,...,6}
   ```

2. **Equalized Odds:** Equal TPR and FPR across groups
   ```
   P(Ŷ=1 | Y=y, A=a) ≈ P(Ŷ=1 | Y=y, A=a') for y ∈ {0,1}, all a, a'
   ```

3. **Performance Parity:** Similar F1 scores across groups
   ```
   F1(a) ≈ F1(a') for all groups a, a'
   ```

---

## Fairness Components

### 1. Combined Fairness Loss

We directly optimize for fairness using a composite loss function:

```
L_fairness = L_demographic_parity + L_equalized_odds
```

#### Demographic Parity Loss

Minimizes disparity in positive prediction rates across groups:

```python
# For each group pair (a, a')
L_dp = Σ |P(Ŷ=1|A=a) - P(Ŷ=1|A=a')|²
```

**Implementation:**
- Compute mean prediction probability per group
- Calculate pairwise absolute differences
- Average over all group pairs

**Effect:**
- Directly reduces disparate impact
- Encourages similar positive rates across skin types
- Prevents systematic under/over-prediction for specific groups

#### Equalized Odds Loss

Ensures equal TPR and FPR across groups:

```python
# For y ∈ {0, 1} (positive/negative class)
L_eo = Σ |P(Ŷ=1|Y=y,A=a) - P(Ŷ=1|Y=y,A=a')|²
```

**Implementation:** Stratify by label, minimize prediction rate differences per group for both classes.

### 2. Combined Loss Weight

The fairness loss is weighted by `λ_fair` (default: 0.1):

```
Total_Fairness_Component = λ_fair × (L_dp + L_eo)
```

**Rationale:** Balances fairness with task performance. Empirically tuned.

---

## Adversarial Debiasing

### Architecture

FairCBM uses an adversarial discriminator with gradient reversal:

```
Encoder → Concept Bottleneck → Binary Classifier
              ↓ (Gradient Reversal)
         Discriminator → Fitzpatrick Prediction
```

**Components:**
1. **Discriminator Network:**
   - Input: Concept features (dimension: 512)
   - Architecture: 512 → 256 → 128 → 6 (Fitzpatrick classes)
   - Activation: ReLU + Dropout (0.3)
   - Output: Softmax over 6 groups

2. **Gradient Reversal Layer (GRL):**
   - Forward: Identity
   - Backward: Negate gradients × α

### Theoretical Foundation

Based on Domain-Adversarial Neural Networks (Ganin & Lempitsky, 2015):

**Objective:**
```
min_θ max_φ L_task(θ) - λ_adv × L_disc(θ, φ)
```

where:
- `θ`: Encoder + concept + binary classifier parameters
- `φ`: Discriminator parameters
- `L_task`: Concept + binary classification loss
- `L_disc`: Discriminator loss (cross-entropy)

**Gradient Reversal:**
```
∂L/∂θ_encoder = ∂L_task/∂θ - λ_adv × α × ∂L_disc/∂θ
```

**Effect:**
- Discriminator tries to predict Fitzpatrick type from concepts
- Encoder receives reversed gradients → learns to confuse discriminator
- Result: Concept representations become group-agnostic

### Adversarial Alpha Scheduling

The gradient reversal strength `α` follows a schedule:

```python
# Linear warmup from 0 to 2
α(t) = 2 * t / T

where:
- t: current training iteration
- T: total training iterations
```

**Rationale:**
- Early training: Let encoder learn useful representations (α ≈ 0)
- Late training: Strong debiasing (α → 2)
- Prevents training collapse from aggressive early debiasing

---

## Warmup Scheduling

### The Problem: Training Instability

Static adversarial weights cause:
1. **Adversarial Loss Explosion:** Unbounded cross-entropy → gradients explode
2. **Performance Collapse:** Val F1 drops to near-zero
3. **Gradient Conflicts:** Task and fairness objectives interfere

**Example of Failure (static λ_adv=0.01):**
```
Epoch 15: Adv Loss = 1.78, Val F1 = 0.68
Epoch 30: Adv Loss = 12.4, Val F1 = 0.42
Epoch 50: Adv Loss = 2,028, Val F1 = 0.03  ❌
```

### Solution: 3-Phase Warmup

FairCBM uses a staged warmup schedule:

#### Phase 1: Task Learning (0-20% epochs)
```
λ_adv(t) = 0.0
```
No adversarial training. Model learns concepts and classification.

#### Phase 2: Gradual Fairness (20-50% epochs)
```
λ_adv(t) = λ_target × (t - t_start) / (t_end - t_start)
```
Linear warmup from 0 to target weight.

#### Phase 3: Full Fairness (50-100% epochs)
```
λ_adv(t) = λ_target
```
Full adversarial weight (default: 0.01).

### Example Schedule (100 epochs, λ_target=0.01)

| Epoch Range | Phase | λ_adv | Purpose |
|-------------|-------|-------|---------|
| 1-20 | Task Learning | 0.000 | Stable baseline |
| 21-50 | Warmup | 0.000→0.010 | Gradual fairness |
| 51-100 | Full Fairness | 0.010 | Convergence |

### Stability Fixes

In addition to warmup, we prevent loss explosion:

1. **Gradient Clipping:** `max_norm=1.0` on all parameters
2. **Label Smoothing:** `ε=0.1` in adversarial cross-entropy
3. **Loss Capping:** `clamp(L_adv, max=10.0)`

**Result:**
```
Epoch 15: Adv Loss = 1.45, Val F1 = 0.72
Epoch 30: Adv Loss = 2.31, Val F1 = 0.70
Epoch 50: Adv Loss = 2.87, Val F1 = 0.71  ✓
```

---

## Theoretical Guarantees

### Fairness-Accuracy Tradeoff

**Theorem:** Perfect fairness and perfect accuracy are generally incompatible when base rates differ.

**FairCBM Approach:** Optimize tradeoff via λ_fair and λ_adv. Target <5% accuracy loss for >50% fairness gain.

### Group Invariance

**Proposition:** Gradient reversal with sufficient capacity discriminator yields approximately group-invariant representations.

**Formalization:**
```
H-divergence: d_H(D_a, D_a') → 0 as training progresses
```

where D_a is the distribution of representations for group a.

**Intuition:** Discriminator achieves near-random accuracy (~16.7%), implying concept features contain minimal group information.

### Convergence Properties

**Assumption:** Bounded gradients, convex fairness loss (locally)

**Result:** With proper warmup, loss converges to local minimum satisfying:
```
∇L_total = ∇L_concept + ∇L_binary + λ_fair∇L_fairness + λ_adv∇L_adversarial ≈ 0
```

Loss typically stabilizes by epoch 70-80.

---

## Implementation Details

### Complete Loss Function

```python
# Total training loss
L_total = L_concept + L_binary + λ_fair × L_fairness + λ_adv(t) × L_adversarial

where:
  L_concept = BCE(concept_logits, concept_labels)
  L_binary = BCE(binary_logits, binary_labels)
  L_fairness = L_demographic_parity + L_equalized_odds
  L_adversarial = clamp(CE(discriminator_logits, groups, label_smoothing=0.1), max=10)
  λ_adv(t) = warmup_schedule(epoch, total_epochs)
```

### Hyperparameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `λ_fair` | 0.1 | [0.01, 1.0] | Fairness loss weight |
| `λ_adv` | 0.01 | [0.001, 0.1] | Adversarial weight (target) |
| `warmup_start` | 20% | [10%, 30%] | When warmup begins |
| `warmup_end` | 50% | [40%, 60%] | When full λ_adv reached |
| `α_schedule` | Linear | - | GRL strength schedule |
| `grad_clip` | 1.0 | [0.5, 5.0] | Gradient clipping norm |
| `label_smoothing` | 0.1 | [0.0, 0.2] | Adversarial CE smoothing |

### Training Procedure

```python
for epoch in range(total_epochs):
    # 1. Update adversarial lambda
    λ_adv = compute_warmup_lambda(epoch, total_epochs, λ_target)
    
    # 2. Forward pass
    concepts = encoder(images)
    binary_logits = binary_classifier(concepts)
    group_logits = discriminator(GRL(concepts, α))
    
    # 3. Compute losses
    L_concept = concept_loss(concepts, concept_labels)
    L_binary = binary_loss(binary_logits, binary_labels)
    L_fairness = fairness_loss(binary_logits, binary_labels, groups)
    L_adversarial = adversarial_loss(group_logits, groups)
    
    # 4. Combined loss
    L_total = L_concept + L_binary + λ_fair * L_fairness + λ_adv * L_adversarial
    
    # 5. Backward pass with gradient clipping
    L_total.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

---

## Comparison with Other Approaches

### Post-Processing Methods

Adjust thresholds per group after training. Cannot change representations.

**FairCBM:** Learns fair representations during training.

### Pre-Processing Methods

Reweight or resample training data. May discard useful data.

**FairCBM:** Uses all data, learns group-invariant features.

### In-Training Methods (FairCBM)

**Advantages:**
- Direct optimization of fairness objectives
- Learns fair representations from scratch
- Combines multiple fairness approaches
- Interpretable via concept bottleneck

**Challenges:**
- Requires careful hyperparameter tuning
- Training instability (solved via warmup)
- Fairness-accuracy tradeoff (managed via λ weights)

---

## Fair Curriculum CBM: 4-Phase Fairness-First Curriculum

### Overview

Fair Curriculum CBM extends the static fairness approach with a **dynamic 4-phase curriculum** that prioritizes group fairness before concept complexity.

### Phase Design

#### Phase 1: Balanced Foundation (0-25% of training)
**Goal:** Learn group-invariant representations naturally

- **Sampling:** Balanced per Fitzpatrick type (~equal samples)
- **Fairness Loss:** λ_fair = 0 (no explicit fairness penalty)
- **Adversarial:** λ_adv = 0 (disabled)
- **Concepts:** All 23 concepts (joint training)
- **Rationale:** Balanced data prevents group encoding in initial representations. Unlike Curriculum CBM which orders concepts by difficulty, Fair Curriculum CBM trains all concepts jointly while varying fairness objectives.

#### Phase 2: Demographic Parity Focus (25-50%)
**Goal:** Equalize positive prediction rates P(Ŷ=1|A=a)

- **Sampling:** Continue balanced sampling
- **Fairness Loss:** L_fairness = L_dp only
- **Adversarial:** λ_adv = 0 (not yet active)
- **Concepts:** All 23 concepts (joint training)
- **Rationale:** Focus on single fairness criterion before adding complexity

#### Phase 3: Equalized Odds Focus (50-75%)
**Goal:** Equalize TPR and FPR across groups

- **Sampling:** Stratified by (group × label) for 12 strata
- **Fairness Loss:** L_fairness = 0.3*L_dp + 0.7*L_eo
- **Adversarial:** Linear warmup 0→0.01 (gradient reversal active)
- **Concepts:** All 23 concepts (joint training)
- **Rationale:** Shift emphasis to equalized odds while maintaining DP

#### Phase 4: Performance Parity (75-100%)
**Goal:** Minimize F1 range across groups

- **Sampling:** Error-driven (weight ∝ 1/(F1+ε) for F1≥0.1, minimal weight for F1<0.1)
- **Fairness Loss:** L_fairness = 0.33*L_dp + 0.33*L_eo + 0.34*L_pg
- **Adversarial:** λ_adv = 0.01 (full strength)
- **Concepts:** All 23 concepts (joint training)
- **Rationale:** Balance all fairness criteria, oversample struggling groups

### FairnessAwareSampler

Custom PyTorch Sampler that implements phase-dependent sampling strategies:

```python
class FairnessAwareSampler(Sampler):
    def __iter__(self):
        phase = self._get_phase()
        
        if phase in ['balanced_foundation', 'demographic_parity']:
            return self._balanced_group_sampling()
        elif phase == 'equalized_odds':
            return self._stratified_sampling()
        elif phase == 'performance_parity':
            return self._error_driven_sampling()
```

**Balanced Sampling:** Samples `batch_size // 6` from each Fitzpatrick type

**Stratified Sampling:** Samples equally from 12 strata (6 groups × 2 labels)

**Error-Driven Sampling:** 
- Groups with F1 ≥ 0.1: weight = 1/(F1 + 0.1)
- Groups with F1 < 0.1: weight = 0.5 (minimal, likely missing from validation)
- Prevents oversampling groups absent from validation set

### PhasedFairnessLoss

Dynamic fairness loss that changes emphasis per phase:

```python
class PhasedFairnessLoss(nn.Module):
    def forward(self, predictions, labels, groups, epoch):
        phase = self._get_phase(epoch)
        
        if phase == 'balanced_foundation':
            return 0.0  # No fairness loss
        
        L_dp = self._demographic_parity_loss(...)
        L_eo = self._equalized_odds_loss(...)
        
        if phase == 'demographic_parity':
            return L_dp
        elif phase == 'equalized_odds':
            return 0.3 * L_dp + 0.7 * L_eo
        elif phase == 'performance_parity':
            L_pg = self._performance_gap_loss(...)
            return 0.33*L_dp + 0.33*L_eo + 0.34*L_pg
```

### Key Advantages Over Static Fairness

1. **Progressive Fairness:** Builds foundation before enforcing constraints
2. **Targeted Sampling:** Adapts data distribution to phase objectives
3. **Balanced Multi-Criteria:** Focuses on one criterion before combining
4. **Error-Driven Adaptation:** Responds to group performance in Phase 4
5. **Stability:** Gradual introduction prevents training collapse

### Training Integration

DataLoader must be recreated every epoch to update sampler:

```python
for epoch in range(total_epochs):
    # Recreate DataLoader with updated sampler
    sampler = FairnessAwareSampler(
        groups=train_groups,
        labels=train_labels,
        batch_size=batch_size,
        epoch=epoch,
        total_epochs=total_epochs,
        group_f1_scores=model.group_f1_scores  # For Phase 4
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=4
    )
    
    # Training loop...
```

Group F1 scores are updated every 5 epochs during validation to inform Phase 4 error-driven sampling.

## Future Directions

1. Individual fairness (similar inputs → similar predictions)
2. Intersectional fairness (multiple protected attributes)
3. Causal fairness (distinguish legitimate causal pathways)
4. Dynamic fairness (adapt constraints during deployment)
5. Certified fairness (provable guarantees)
6. Adaptive phase boundaries based on validation metrics

---

## References

1. **Ganin & Lempitsky (2015):** "Domain-Adversarial Training of Neural Networks"
2. **Koh et al. (2020):** "Concept Bottleneck Models"
3. **Hardt et al. (2016):** "Equality of Opportunity in Supervised Learning"
4. **Barocas et al. (2019):** "Fairness and Machine Learning" (textbook)
5. **Groh et al. (2021):** "Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset"

---

*For implementation details, see `src/models/fairness_aware_cbm.py` and `src/utils/adversarial_debiasing.py`*
