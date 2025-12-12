# Fairness-Aware Curriculum Learning Integration Plan

## Executive Summary

This document outlines the systematic integration of fairness-aware curriculum learning into the existing SynergyCBM codebase. The plan leverages production-ready curriculum infrastructure while adding adversarial debiasing and group fairness constraints.

**Timeline:** 8 major tasks  
**Approach:** Copy → Adapt → Extend → Validate

---

## Directory Structure

```
fairness/
├── INTEGRATION_PLAN.md              # This document
├── README.md                         # Fairness project overview
├── requirements.txt                  # Additional dependencies (aequitas, etc.)
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fairness_aware_cbm.py   # NEW: Extends MinimalCurriculumCBM
│   │   └── adversarial_discriminator.py  # NEW: Group prediction network
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── fairness_metrics.py     # NEW: Demographic parity, equalized odds, etc.
│   │   ├── adversarial_debiasing.py  # NEW: Gradient reversal, fairness losses
│   │   ├── group_aware_sampling.py  # NEW: Stratified batch sampling
│   │   ├── metrics.py              # ADAPTED: Add per-group metrics
│   │   ├── calibration.py          # COPY: Per-group calibration
│   │   └── best_model_tracker.py   # COPY: Track best fairness-accuracy tradeoff
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fitzpatrick_dataloader.py  # ADAPTED: Add Fitzpatrick annotations
│   │   └── stratified_sampler.py   # NEW: Group-balanced sampling
│   └── configs/
│       ├── __init__.py
│       └── fairness_curriculum_configs.py  # ADAPTED: Add fairness hyperparameters
├── scripts/
│   ├── train_fairness_cbm.py       # ADAPTED: Main training with fairness
│   ├── evaluate_fairness.py        # NEW: Comprehensive fairness evaluation
│   ├── analyze_fairness_results.py # ADAPTED: Multi-run fairness analysis
│   ├── run_fairness_experiments.py # ADAPTED: Batch experiments with fairness sweep
│   ├── calibrate_fairness_models.py  # ADAPTED: Per-group calibration
│   └── compare_fairness_vs_standard.py  # NEW: Fairness vs. standard CBM comparison
├── data/
│   ├── fitzpatrick_annotations.csv # NEW: Image → Fitzpatrick type mapping
│   └── README.md                    # Data format documentation
├── results/
│   └── .gitkeep
├── slurm/
│   ├── run_fairness_experiment.slurm  # ADAPTED: SLURM job script
│   └── fairness_array_job.slurm      # NEW: Array jobs for fairness sweep
└── docs/
    ├── fairness_methodology.md     # Fairness approach documentation
    ├── metrics_guide.md            # Fairness metrics explanations
    └── experimental_design.md      # Experiment protocols
```

---

## Phase 1: Analysis & Core Infrastructure

### Task 1.1: File Classification

**Files to COPY (No Modifications):**
```
src/utils/best_model_tracker.py      → fairness/src/utils/best_model_tracker.py
src/utils/calibration.py             → fairness/src/utils/calibration.py
src/utils/model_utils.py             → fairness/src/utils/model_utils.py
src/utils/information_metrics.py     → fairness/src/utils/information_metrics.py
```
*Rationale:* These utilities are general-purpose and don't require fairness-specific modifications.

**Files to COPY and ADAPT:**
```
src/models/minimal_curriculum_cbm.py → fairness/src/models/fairness_aware_cbm.py
src/utils/metrics.py                 → fairness/src/utils/metrics.py
src/data/dataloader.py               → fairness/src/data/fitzpatrick_dataloader.py
src/configs/curriculum_configs.py    → fairness/src/configs/fairness_curriculum_configs.py
scripts/test_minimal_curriculum.py   → fairness/scripts/train_fairness_cbm.py
scripts/aggregate_multi_run_results.py → fairness/scripts/analyze_fairness_results.py
```
*Rationale:* Core functionality exists but needs fairness extensions.

**Files to CREATE from Scratch:**
```
fairness/src/models/adversarial_discriminator.py    # Gradient reversal architecture
fairness/src/utils/fairness_metrics.py              # Demographic parity, equalized odds, etc.
fairness/src/utils/adversarial_debiasing.py         # Fairness loss functions
fairness/src/utils/group_aware_sampling.py          # Stratified batch construction
fairness/src/data/stratified_sampler.py             # PyTorch sampler for balanced groups
fairness/scripts/evaluate_fairness.py               # Aequitas-based evaluation
fairness/scripts/compare_fairness_vs_standard.py    # Comparative analysis
fairness/data/fitzpatrick_annotations.csv           # Dataset annotations
```
*Rationale:* Fairness-specific functionality with no existing equivalent.

---

## Phase 2: Model Architecture Extensions

### Task 2.1: Adversarial Discriminator (NEW)

**File:** `fairness/src/models/adversarial_discriminator.py`

**Purpose:** Predicts Fitzpatrick skin type from intermediate representations to enable adversarial debiasing.

**Architecture:**
```python
class AdversarialDiscriminator(nn.Module):
    """
    Adversarial network for group prediction.
    Used with gradient reversal to learn group-invariant representations.
    """
    
    def __init__(self, input_dim: int = 512, hidden_dims: List[int] = [256, 128],
                 num_groups: int = 6, dropout: float = 0.3):
        """
        Args:
            input_dim: Feature dimension from concept layer
            hidden_dims: Hidden layer dimensions
            num_groups: Number of Fitzpatrick types (6: I-VI)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_groups))
        self.network = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Forward pass with optional gradient reversal.
        
        Args:
            features: Concept-layer features [batch, input_dim]
            alpha: Gradient reversal strength (0 = no reversal, 1 = full reversal)
        
        Returns:
            group_logits: [batch, num_groups]
        """
        # Apply gradient reversal layer
        reversed_features = GradientReversalLayer.apply(features, alpha)
        group_logits = self.network(reversed_features)
        return group_logits


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer from Ganin & Lempitsky (2015).
    Forward: identity function
    Backward: multiply gradient by -alpha
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
```

**Integration:** Attached to concept layer of FairnessAwareCBM.

---

### Task 2.2: Fairness-Aware CBM (ADAPTED)

**File:** `fairness/src/models/fairness_aware_cbm.py`

**Base:** `src/models/minimal_curriculum_cbm.py` (883 lines)

**Key Modifications:**

1. **Add adversarial discriminator:**
```python
class FairnessAwareCBM(MinimalCurriculumCBM):
    def __init__(self, *args, num_groups: int = 6, 
                 fairness_lambda: float = 1.0, adversarial_lambda: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add adversarial discriminator for group prediction
        self.adversarial_discriminator = AdversarialDiscriminator(
            input_dim=512,  # Match concept layer hidden dimension
            hidden_dims=[256, 128],
            num_groups=num_groups,
            dropout=0.3
        ).to(self.device)
        
        # Fairness hyperparameters
        self.fairness_lambda = fairness_lambda
        self.adversarial_lambda = adversarial_lambda
        self.num_groups = num_groups
        
        # Track gradient reversal strength (increases during training)
        self.adversarial_alpha = 0.0
```

2. **Modified forward pass to return intermediate features:**
```python
def forward(self, x: torch.Tensor, return_features: bool = False) -> Tuple:
    """
    Forward pass with optional feature extraction for adversarial training.
    
    Returns:
        concept_logits: [batch, num_concepts]
        binary_logits: [batch, 1]
        features: [batch, 512] (if return_features=True)
    """
    features = self.backbone(x)
    
    # ... pooling logic from original ...
    
    # Concept prediction
    concept_features = self.concept_layer[0](features)  # Before final layer
    concept_logits = self.concept_layer[1](concept_features)
    
    # Binary prediction
    concept_probs = torch.sigmoid(concept_logits)
    binary_logits = self.binary_classifier(concept_probs)
    
    if return_features:
        return concept_logits, binary_logits, concept_features
    return concept_logits, binary_logits
```

3. **Fairness-aware loss computation:**
```python
def compute_fairness_loss(self, concept_logits, binary_logits, concept_features,
                         concept_labels, binary_labels, group_labels,
                         active_concepts, phase_name: str = "all") -> Dict[str, torch.Tensor]:
    """
    Compute loss with fairness constraints.
    
    Returns:
        losses: {
            'concept_loss': BCE on concepts,
            'binary_loss': BCE on diagnosis,
            'fairness_loss': Demographic parity + equalized odds,
            'adversarial_loss': Group prediction (for gradient reversal),
            'total_loss': Weighted combination
        }
    """
    # Standard losses from parent class
    base_losses = self.compute_loss(
        concept_logits, binary_logits,
        concept_labels, binary_labels,
        active_concepts,
        new_concepts=self._get_new_concepts_for_phase(phase_name)
    )
    
    # Fairness loss: demographic parity
    binary_probs = torch.sigmoid(binary_logits).squeeze()
    fairness_loss = compute_demographic_parity_loss(binary_probs, group_labels)
    
    # Add equalized odds constraint
    fairness_loss += compute_equalized_odds_loss(binary_probs, binary_labels, group_labels)
    
    # Adversarial loss: predict group from concept features (with gradient reversal)
    group_logits = self.adversarial_discriminator(concept_features, alpha=self.adversarial_alpha)
    adversarial_loss = F.cross_entropy(group_logits, group_labels)
    
    # Total loss
    total_loss = (
        base_losses['total_loss'] +
        self.fairness_lambda * fairness_loss -
        self.adversarial_lambda * adversarial_loss  # Negative to encourage invariance
    )
    
    return {
        **base_losses,
        'fairness_loss': fairness_loss,
        'adversarial_loss': adversarial_loss,
        'total_loss': total_loss
    }
```

4. **Update adversarial alpha during training:**
```python
def update_adversarial_alpha(self, epoch: int, max_epochs: int):
    """Gradually increase gradient reversal strength."""
    progress = epoch / max_epochs
    self.adversarial_alpha = 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0  # 0 → 1
```

---

## Phase 3: Fairness Metrics Implementation

### Task 3.1: Core Fairness Metrics (NEW)

**File:** `fairness/src/utils/fairness_metrics.py`

**Metrics to Implement:**

```python
def compute_demographic_parity(predictions: np.ndarray, 
                              groups: np.ndarray,
                              threshold: float = 0.5) -> Dict[str, float]:
    """
    Demographic Parity: P(Ŷ=1 | A=a) should be equal across groups.
    
    Returns:
        {
            'group_positive_rates': {group: rate},
            'max_disparity': max_rate - min_rate,
            'disparate_impact_ratio': min_rate / max_rate,
            'statistical_parity_difference': max_disparity
        }
    """
    binary_preds = (predictions >= threshold).astype(int)
    
    group_rates = {}
    for group in np.unique(groups):
        group_mask = (groups == group)
        group_rates[int(group)] = binary_preds[group_mask].mean()
    
    rates = list(group_rates.values())
    max_rate = max(rates)
    min_rate = min(rates)
    
    return {
        'group_positive_rates': group_rates,
        'max_disparity': max_rate - min_rate,
        'disparate_impact_ratio': min_rate / max_rate if max_rate > 0 else 0.0,
        'statistical_parity_difference': max_rate - min_rate
    }


def compute_equalized_odds(predictions: np.ndarray,
                          labels: np.ndarray,
                          groups: np.ndarray,
                          threshold: float = 0.5) -> Dict[str, float]:
    """
    Equalized Odds: P(Ŷ=1 | Y=y, A=a) should be equal across groups for both y=0 and y=1.
    
    Measures:
    - TPR disparity: max_group(TPR) - min_group(TPR)
    - FPR disparity: max_group(FPR) - min_group(FPR)
    - Average absolute disparity: mean(|TPR_disparity|, |FPR_disparity|)
    """
    binary_preds = (predictions >= threshold).astype(int)
    
    group_tpr = {}
    group_fpr = {}
    
    for group in np.unique(groups):
        group_mask = (groups == group)
        group_preds = binary_preds[group_mask]
        group_labels = labels[group_mask]
        
        # True Positive Rate (Sensitivity)
        positives = (group_labels == 1)
        if positives.sum() > 0:
            group_tpr[int(group)] = group_preds[positives].mean()
        else:
            group_tpr[int(group)] = 0.0
        
        # False Positive Rate
        negatives = (group_labels == 0)
        if negatives.sum() > 0:
            group_fpr[int(group)] = group_preds[negatives].mean()
        else:
            group_fpr[int(group)] = 0.0
    
    tpr_values = list(group_tpr.values())
    fpr_values = list(group_fpr.values())
    
    tpr_disparity = max(tpr_values) - min(tpr_values)
    fpr_disparity = max(fpr_values) - min(fpr_values)
    
    return {
        'group_tpr': group_tpr,
        'group_fpr': group_fpr,
        'tpr_disparity': tpr_disparity,
        'fpr_disparity': fpr_disparity,
        'equalized_odds_difference': (tpr_disparity + fpr_disparity) / 2,
        'max_disparity': max(tpr_disparity, fpr_disparity)
    }


def compute_calibration_by_group(predictions: np.ndarray,
                                 labels: np.ndarray,
                                 groups: np.ndarray,
                                 n_bins: int = 10) -> Dict[str, Any]:
    """
    Calibration quality per group (reliability diagrams).
    
    Returns:
        {
            'group_ece': {group: expected_calibration_error},
            'group_mce': {group: max_calibration_error},
            'calibration_disparity': max_ece - min_ece
        }
    """
    from src.utils.calibration import compute_calibration_error
    
    group_ece = {}
    group_mce = {}
    
    for group in np.unique(groups):
        group_mask = (groups == group)
        group_ece[int(group)], group_mce[int(group)], _ = compute_calibration_error(
            predictions[group_mask],
            labels[group_mask],
            n_bins=n_bins
        )
    
    ece_values = list(group_ece.values())
    calibration_disparity = max(ece_values) - min(ece_values)
    
    return {
        'group_ece': group_ece,
        'group_mce': group_mce,
        'calibration_disparity': calibration_disparity,
        'mean_ece': np.mean(ece_values)
    }


def compute_worst_group_performance(predictions: np.ndarray,
                                   labels: np.ndarray,
                                   groups: np.ndarray,
                                   threshold: float = 0.5) -> Dict[str, Any]:
    """
    Identify worst-performing group (min F1) and performance gap.
    
    Returns:
        {
            'group_f1': {group: f1_score},
            'worst_group': group_id,
            'worst_group_f1': min_f1,
            'best_group_f1': max_f1,
            'performance_gap': max_f1 - min_f1
        }
    """
    from sklearn.metrics import f1_score
    
    binary_preds = (predictions >= threshold).astype(int)
    
    group_f1 = {}
    for group in np.unique(groups):
        group_mask = (groups == group)
        group_f1[int(group)] = f1_score(
            labels[group_mask],
            binary_preds[group_mask],
            zero_division=0
        )
    
    f1_values = list(group_f1.values())
    worst_group = min(group_f1, key=group_f1.get)
    
    return {
        'group_f1': group_f1,
        'worst_group': worst_group,
        'worst_group_f1': min(f1_values),
        'best_group_f1': max(f1_values),
        'performance_gap': max(f1_values) - min(f1_values)
    }
```

### Task 3.2: Aequitas Integration

**Purpose:** Use industry-standard fairness audit toolkit.

```python
def generate_aequitas_report(predictions: np.ndarray,
                            labels: np.ndarray,
                            groups: np.ndarray,
                            threshold: float = 0.5,
                            output_path: str = None) -> Dict[str, Any]:
    """
    Generate comprehensive fairness audit using Aequitas.
    
    Returns:
        Full Aequitas metrics including:
        - Disparate impact ratios
        - Group confusion matrices
        - Parity metrics across multiple definitions
    """
    from aequitas.group import Group
    from aequitas.bias import Bias
    from aequitas.fairness import Fairness
    
    binary_preds = (predictions >= threshold).astype(int)
    
    # Create Aequitas-compatible DataFrame
    df = pd.DataFrame({
        'entity_id': np.arange(len(predictions)),
        'score': predictions,
        'label_value': labels,
        'pred_value': binary_preds,
        'group': groups
    })
    
    # Compute group metrics
    g = Group()
    xtab, _ = g.get_crosstabs(df, attr_cols=['group'])
    
    # Compute bias metrics
    b = Bias()
    bdf = b.get_disparity(xtab, original_df=df, ref_groups_dict={'group': 1})
    
    # Compute fairness flags
    f = Fairness()
    fdf = f.get_group_value_fairness(bdf)
    
    if output_path:
        fdf.to_csv(output_path, index=False)
    
    return {
        'aequitas_crosstabs': xtab.to_dict(),
        'aequitas_bias': bdf.to_dict(),
        'aequitas_fairness': fdf.to_dict()
    }
```

---

## Phase 4: Adversarial Debiasing

### Task 4.1: Fairness Loss Functions (NEW)

**File:** `fairness/src/utils/adversarial_debiasing.py`

```python
def compute_demographic_parity_loss(predictions: torch.Tensor,
                                   groups: torch.Tensor) -> torch.Tensor:
    """
    Loss that encourages equal positive prediction rates across groups.
    
    L_dp = Σ_g Σ_h |P(Ŷ=1|A=g) - P(Ŷ=1|A=h)|²
    """
    unique_groups = torch.unique(groups)
    group_rates = []
    
    for group in unique_groups:
        group_mask = (groups == group)
        if group_mask.sum() > 0:
            group_rate = predictions[group_mask].mean()
            group_rates.append(group_rate)
    
    # Pairwise disparity
    loss = 0.0
    for i in range(len(group_rates)):
        for j in range(i + 1, len(group_rates)):
            loss += (group_rates[i] - group_rates[j]) ** 2
    
    return loss


def compute_equalized_odds_loss(predictions: torch.Tensor,
                               labels: torch.Tensor,
                               groups: torch.Tensor) -> torch.Tensor:
    """
    Loss that encourages equal TPR and FPR across groups.
    
    L_eo = Σ_g Σ_h [|TPR_g - TPR_h|² + |FPR_g - FPR_h|²]
    """
    unique_groups = torch.unique(groups)
    group_tpr = []
    group_fpr = []
    
    for group in unique_groups:
        group_mask = (groups == group)
        group_preds = predictions[group_mask]
        group_labels = labels[group_mask]
        
        # TPR
        positives = (group_labels == 1)
        if positives.sum() > 0:
            tpr = group_preds[positives].mean()
        else:
            tpr = torch.tensor(0.0, device=predictions.device)
        group_tpr.append(tpr)
        
        # FPR
        negatives = (group_labels == 0)
        if negatives.sum() > 0:
            fpr = group_preds[negatives].mean()
        else:
            fpr = torch.tensor(0.0, device=predictions.device)
        group_fpr.append(fpr)
    
    # Pairwise disparity
    loss = 0.0
    for i in range(len(group_tpr)):
        for j in range(i + 1, len(group_tpr)):
            loss += (group_tpr[i] - group_tpr[j]) ** 2
            loss += (group_fpr[i] - group_fpr[j]) ** 2
    
    return loss


def compute_calibration_fairness_loss(predictions: torch.Tensor,
                                     labels: torch.Tensor,
                                     groups: torch.Tensor,
                                     n_bins: int = 10) -> torch.Tensor:
    """
    Loss that encourages equal calibration across groups.
    
    Penalizes differences in predicted vs. actual positive rates within bins.
    """
    unique_groups = torch.unique(groups)
    group_calibration_errors = []
    
    for group in unique_groups:
        group_mask = (groups == group)
        group_preds = predictions[group_mask]
        group_labels = labels[group_mask]
        
        # Bin predictions
        bins = torch.linspace(0, 1, n_bins + 1, device=predictions.device)
        calibration_error = 0.0
        
        for i in range(n_bins):
            bin_mask = (group_preds >= bins[i]) & (group_preds < bins[i + 1])
            if bin_mask.sum() > 0:
                predicted_rate = group_preds[bin_mask].mean()
                actual_rate = group_labels[bin_mask].float().mean()
                calibration_error += (predicted_rate - actual_rate) ** 2
        
        group_calibration_errors.append(calibration_error)
    
    # Variance of calibration errors across groups
    return torch.var(torch.stack(group_calibration_errors))
```

---

## Phase 5: Data Infrastructure

### Task 5.1: Fitzpatrick Annotations (NEW)

**File:** `fairness/data/fitzpatrick_annotations.csv`

**Format:**
```csv
image_id,fitzpatrick_type,fitzpatrick_label
ISIC_0000001,3,III
ISIC_0000002,2,II
ISIC_0000003,5,V
...
```

**Notes:**
- `fitzpatrick_type`: Integer 1-6 (for indexing)
- `fitzpatrick_label`: String I-VI (for readability)
- Must align with existing SkinCap image IDs

### Task 5.2: Fitzpatrick DataLoader (ADAPTED)

**File:** `fairness/src/data/fitzpatrick_dataloader.py`

**Base:** `src/data/dataloader.py`

**Key Modifications:**

```python
class SkinCapFitzpatrickDataset(SkinCapDataset):
    """
    Extends SkinCapDataset to include Fitzpatrick skin type annotations.
    """
    
    def __init__(self, csv_file: str, img_dir: str, 
                 fitzpatrick_csv: str,  # NEW: Path to Fitzpatrick annotations
                 lesion_cols: List[str],
                 label_type: str = 'concept',
                 transform=None):
        super().__init__(csv_file, img_dir, lesion_cols, label_type, transform)
        
        # Load Fitzpatrick annotations
        fitz_df = pd.read_csv(fitzpatrick_csv)
        
        # Merge with main DataFrame
        self.df = self.df.merge(
            fitz_df[['image_id', 'fitzpatrick_type']],
            left_on='lesion_id',  # Adjust column name as needed
            right_on='image_id',
            how='left'
        )
        
        # Handle missing Fitzpatrick annotations
        if self.df['fitzpatrick_type'].isnull().any():
            logging.warning(f"{self.df['fitzpatrick_type'].isnull().sum()} images missing Fitzpatrick annotations")
            # Option 1: Drop samples without annotations
            # self.df = self.df[self.df['fitzpatrick_type'].notnull()]
            
            # Option 2: Assign default group (least common to avoid bias)
            # self.df['fitzpatrick_type'].fillna(6, inplace=True)
        
        # Convert to integer for PyTorch
        self.df['fitzpatrick_type'] = self.df['fitzpatrick_type'].astype(int)
    
    def __getitem__(self, idx):
        """Returns (image, concept_labels, binary_label, fitzpatrick_type)."""
        img, concept_labels, binary_label = super().__getitem__(idx)
        fitzpatrick_type = int(self.df.iloc[idx]['fitzpatrick_type']) - 1  # 0-indexed
        
        return img, concept_labels, binary_label, fitzpatrick_type
```

### Task 5.3: Stratified Sampler (NEW)

**File:** `fairness/src/data/stratified_sampler.py`

**Purpose:** Ensure balanced Fitzpatrick representation in each batch.

```python
class GroupBalancedBatchSampler(torch.utils.data.Sampler):
    """
    Samples batches with balanced representation from each group.
    
    For batch_size=32 and 6 groups, ensures ~5-6 samples per group per batch.
    """
    
    def __init__(self, group_labels: np.ndarray, batch_size: int, 
                 drop_last: bool = False):
        """
        Args:
            group_labels: Array of group IDs for each sample
            batch_size: Target batch size
            drop_last: Whether to drop last incomplete batch
        """
        self.group_labels = group_labels
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Organize indices by group
        self.group_indices = {}
        for group in np.unique(group_labels):
            self.group_indices[group] = np.where(group_labels == group)[0].tolist()
        
        self.num_groups = len(self.group_indices)
        self.samples_per_group = batch_size // self.num_groups
    
    def __iter__(self):
        # Shuffle indices within each group
        group_iters = {
            group: iter(np.random.permutation(indices))
            for group, indices in self.group_indices.items()
        }
        
        while True:
            batch = []
            
            # Sample approximately equal number from each group
            for group in self.group_indices.keys():
                group_batch = []
                try:
                    for _ in range(self.samples_per_group):
                        group_batch.append(next(group_iters[group]))
                except StopIteration:
                    # Reshuffle and continue
                    group_iters[group] = iter(np.random.permutation(self.group_indices[group]))
                    for _ in range(self.samples_per_group):
                        group_batch.append(next(group_iters[group]))
                
                batch.extend(group_batch)
            
            # Fill remaining slots randomly
            while len(batch) < self.batch_size:
                group = np.random.choice(list(self.group_indices.keys()))
                try:
                    batch.append(next(group_iters[group]))
                except StopIteration:
                    group_iters[group] = iter(np.random.permutation(self.group_indices[group]))
                    batch.append(next(group_iters[group]))
            
            yield batch
            
            # Check if all groups exhausted
            if all(len(list(iter_)) == 0 for iter_ in group_iters.values()):
                break
    
    def __len__(self):
        total_samples = sum(len(indices) for indices in self.group_indices.values())
        if self.drop_last:
            return total_samples // self.batch_size
        return (total_samples + self.batch_size - 1) // self.batch_size
```

---

## Phase 6: Training Infrastructure

### Task 6.1: Fairness Training Script (ADAPTED)

**File:** `fairness/scripts/train_fairness_cbm.py`

**Base:** `scripts/test_minimal_curriculum.py` (1226 lines)

**Key Modifications:**

1. **Add fairness hyperparameters:**
```python
parser.add_argument('--fairness_lambda', type=float, default=1.0,
                   help='Weight for fairness loss')
parser.add_argument('--adversarial_lambda', type=float, default=0.1,
                   help='Weight for adversarial loss')
parser.add_argument('--use_group_balanced_sampling', action='store_true',
                   help='Use stratified batch sampling')
parser.add_argument('--fitzpatrick_csv', type=str, required=True,
                   help='Path to Fitzpatrick annotations CSV')
```

2. **Use Fitzpatrick dataloader:**
```python
from src.data.fitzpatrick_dataloader import SkinCapFitzpatrickDataset

train_dataset = SkinCapFitzpatrickDataset(
    csv_file=args.train_csv,
    img_dir=args.data_dir,
    fitzpatrick_csv=args.fitzpatrick_csv,
    lesion_cols=concept_names,
    label_type='concept'
)
```

3. **Optional group-balanced sampling:**
```python
if args.use_group_balanced_sampling:
    from src.data.stratified_sampler import GroupBalancedBatchSampler
    
    group_labels = train_dataset.df['fitzpatrick_type'].values
    train_sampler = GroupBalancedBatchSampler(
        group_labels, batch_size=args.batch_size
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
```

4. **Modified training loop with fairness losses:**
```python
for epoch in range(num_epochs):
    model.train()
    model.update_adversarial_alpha(epoch, num_epochs)
    
    for batch in train_loader:
        images, concept_labels, binary_labels, group_labels = batch
        
        # Forward pass with features
        concept_logits, binary_logits, concept_features = model(
            images, return_features=True
        )
        
        # Compute fairness-aware loss
        losses = model.compute_fairness_loss(
            concept_logits, binary_logits, concept_features,
            concept_labels, binary_labels, group_labels,
            active_concepts=phase.concepts,
            phase_name=phase.name
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()
        
        # Log fairness metrics
        logging.info(f"Epoch {epoch}, Loss: {losses['total_loss']:.4f}, "
                    f"Fairness: {losses['fairness_loss']:.4f}, "
                    f"Adversarial: {losses['adversarial_loss']:.4f}")
```

5. **Per-group evaluation:**
```python
def evaluate_fairness(model, dataloader, device):
    """Evaluate model with fairness metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_groups = []
    
    with torch.no_grad():
        for images, _, binary_labels, group_labels in dataloader:
            images = images.to(device)
            _, binary_logits = model(images)
            binary_probs = torch.sigmoid(binary_logits).cpu().numpy()
            
            all_preds.extend(binary_probs)
            all_labels.extend(binary_labels.numpy())
            all_groups.extend(group_labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_groups = np.array(all_groups)
    
    # Standard metrics
    from src.utils.metrics import MetricsCalculator
    metrics_calc = MetricsCalculator(concept_names)
    standard_metrics = metrics_calc.calculate_binary_metrics(all_preds, all_labels)
    
    # Fairness metrics
    from src.utils.fairness_metrics import (
        compute_demographic_parity,
        compute_equalized_odds,
        compute_worst_group_performance,
        compute_calibration_by_group
    )
    
    fairness_metrics = {
        'demographic_parity': compute_demographic_parity(all_preds, all_groups),
        'equalized_odds': compute_equalized_odds(all_preds, all_labels, all_groups),
        'worst_group': compute_worst_group_performance(all_preds, all_labels, all_groups),
        'calibration': compute_calibration_by_group(all_preds, all_labels, all_groups)
    }
    
    return {**standard_metrics, **fairness_metrics}
```

---

## Phase 7: Evaluation & Analysis

### Task 7.1: Fairness Evaluation Script (NEW)

**File:** `fairness/scripts/evaluate_fairness.py`

**Purpose:** Comprehensive fairness audit with Aequitas.

```python
#!/usr/bin/env python3
"""
Comprehensive fairness evaluation for trained models.

Generates:
- Per-group performance metrics
- Demographic parity analysis
- Equalized odds analysis
- Calibration by group
- Aequitas fairness audit report
"""

import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.fairness_aware_cbm import FairnessAwareCBM
from src.data.fitzpatrick_dataloader import SkinCapFitzpatrickDataset
from src.utils.fairness_metrics import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--fitzpatrick_csv', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.model_path, map_location=device)
    model.eval()
    
    # Load test data
    test_dataset = SkinCapFitzpatrickDataset(
        csv_file=args.test_csv,
        img_dir=args.data_dir,
        fitzpatrick_csv=args.fitzpatrick_csv,
        lesion_cols=model.concept_names,
        label_type='concept'
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Collect predictions
    all_preds = []
    all_labels = []
    all_groups = []
    
    with torch.no_grad():
        for images, _, binary_labels, group_labels in test_loader:
            images = images.to(device)
            _, binary_logits = model(images)
            binary_probs = torch.sigmoid(binary_logits).cpu().numpy()
            
            all_preds.extend(binary_probs)
            all_labels.extend(binary_labels.numpy())
            all_groups.extend(group_labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_groups = np.array(all_groups)
    
    # Compute all fairness metrics
    results = {
        'overall_metrics': {
            'accuracy': accuracy_score(all_labels, (all_preds >= 0.5).astype(int)),
            'f1': f1_score(all_labels, (all_preds >= 0.5).astype(int)),
            'auc': roc_auc_score(all_labels, all_preds)
        },
        'demographic_parity': compute_demographic_parity(all_preds, all_groups),
        'equalized_odds': compute_equalized_odds(all_preds, all_labels, all_groups),
        'worst_group': compute_worst_group_performance(all_preds, all_labels, all_groups),
        'calibration': compute_calibration_by_group(all_preds, all_labels, all_groups)
    }
    
    # Generate Aequitas report
    aequitas_results = generate_aequitas_report(
        all_preds, all_labels, all_groups,
        output_path=Path(args.output_dir) / 'aequitas_report.csv'
    )
    results['aequitas'] = aequitas_results
    
    # Save results
    output_path = Path(args.output_dir) / 'fairness_evaluation.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Fairness evaluation complete. Results saved to {output_path}")
    
    # Print summary
    print("\n=== Fairness Summary ===")
    print(f"Overall F1: {results['overall_metrics']['f1']:.4f}")
    print(f"Worst Group F1: {results['worst_group']['worst_group_f1']:.4f}")
    print(f"Performance Gap: {results['worst_group']['performance_gap']:.4f}")
    print(f"Demographic Parity Difference: {results['demographic_parity']['max_disparity']:.4f}")
    print(f"Equalized Odds Difference: {results['equalized_odds']['equalized_odds_difference']:.4f}")


if __name__ == '__main__':
    main()
```

### Task 7.2: Multi-Run Fairness Analysis (ADAPTED)

**File:** `fairness/scripts/analyze_fairness_results.py`

**Base:** `scripts/aggregate_multi_run_results.py`

**Additions:**
- Aggregate fairness metrics across 100 runs
- Compute confidence intervals for fairness metrics via stratified bootstrap
- Generate fairness-specific visualizations (group performance gaps, disparity distributions)

---

## Phase 8: Experimental Infrastructure

### Task 8.1: Batch Experiment Runner (ADAPTED)

**File:** `fairness/scripts/run_fairness_experiments.py`

**Purpose:** Sweep over fairness hyperparameters with multi-seed validation.

```python
#!/usr/bin/env python3
"""
Run fairness experiments with hyperparameter sweep.

Sweeps over:
- fairness_lambda: [0.1, 0.5, 1.0, 2.0, 5.0]
- adversarial_lambda: [0.01, 0.05, 0.1, 0.5]
- use_group_balanced_sampling: [True, False]

For each configuration: 100 runs with different seeds.
"""

import subprocess
import itertools
from pathlib import Path

# Hyperparameter grid
fairness_lambdas = [0.1, 0.5, 1.0, 2.0, 5.0]
adversarial_lambdas = [0.01, 0.05, 0.1, 0.5]
group_sampling = [True, False]
seeds = range(1, 101)

# Generate all combinations
configs = list(itertools.product(
    fairness_lambdas,
    adversarial_lambdas,
    group_sampling
))

print(f"Total configurations: {len(configs)}")
print(f"Total experiments: {len(configs) * len(seeds)}")

for (f_lambda, adv_lambda, use_sampling) in configs:
    config_name = f"fair{f_lambda}_adv{adv_lambda}_samp{int(use_sampling)}"
    
    for seed in seeds:
        cmd = [
            'python', 'fairness/scripts/train_fairness_cbm.py',
            '--backbone', 'swin_base_patch4_window7_224',
            '--fairness_lambda', str(f_lambda),
            '--adversarial_lambda', str(adv_lambda),
            '--random_seed', str(seed),
            '--fitzpatrick_csv', 'fairness/data/fitzpatrick_annotations.csv',
            '--output_dir', f'fairness/results/{config_name}/seed_{seed}/'
        ]
        
        if use_sampling:
            cmd.append('--use_group_balanced_sampling')
        
        print(f"Running: {config_name}, seed {seed}")
        subprocess.run(cmd)
```

### Task 8.2: SLURM Array Job (NEW)

**File:** `fairness/slurm/fairness_array_job.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=fairness_cbm
#SBATCH --array=1-10000  # 20 configs × 100 seeds × 5 backbones
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=fairness/logs/fairness_%A_%a.out
#SBATCH --error=fairness/logs/fairness_%A_%a.err

# Activate environment
conda activate CBM-env

# Parse array task ID into config parameters
TOTAL_SEEDS=100
TOTAL_CONFIGS=20

CONFIG_ID=$(( ($SLURM_ARRAY_TASK_ID - 1) / $TOTAL_SEEDS ))
SEED=$(( ($SLURM_ARRAY_TASK_ID - 1) % $TOTAL_SEEDS + 1 ))

# Map CONFIG_ID to hyperparameters
# (use lookup table or hash function)

python fairness/scripts/train_fairness_cbm.py \
    --backbone swin_base_patch4_window7_224 \
    --fairness_lambda $FAIRNESS_LAMBDA \
    --adversarial_lambda $ADVERSARIAL_LAMBDA \
    --random_seed $SEED \
    --fitzpatrick_csv fairness/data/fitzpatrick_annotations.csv \
    --output_dir fairness/results/config_${CONFIG_ID}/seed_${SEED}/
```

---

## Implementation Timeline

### Week 1: Core Infrastructure
- ✅ Create directory structure
- ✅ Copy unchanged files (calibration, model_utils, etc.)
- ✅ Implement adversarial discriminator
- ✅ Implement fairness metrics module

### Week 2: Model Extensions
- ⏳ Adapt MinimalCurriculumCBM → FairnessAwareCBM
- ⏳ Implement adversarial debiasing losses
- ⏳ Add group-aware forward pass

### Week 3: Data Infrastructure
- ⏳ Create Fitzpatrick annotations CSV
- ⏳ Adapt dataloader for Fitzpatrick labels
- ⏳ Implement stratified sampler

### Week 4: Training Scripts
- ⏳ Adapt training script with fairness losses
- ⏳ Implement per-group evaluation
- ⏳ Create fairness evaluation script

### Week 5: Experimental Validation
- ⏳ Run baseline experiments (standard CBM, n=100)
- ⏳ Run fairness experiments (single config, n=100)
- ⏳ Compare fairness vs. standard

### Week 6: Hyperparameter Sweep
- ⏳ Grid search over fairness hyperparameters
- ⏳ Identify optimal fairness-accuracy tradeoff
- ⏳ Multi-run validation (n=100) for best config

### Week 7: Analysis & Visualization
- ⏳ Aggregate results across all experiments
- ⏳ Generate fairness analysis visualizations
- ⏳ Create Aequitas audit reports

### Week 8: Documentation & Refinement
- ⏳ Document all fairness metrics
- ⏳ Create usage guides
- ⏳ Finalize experiment protocols

---

## Success Metrics

**Fairness Improvements:**
- [ ] Performance gap reduced by ≥30% (e.g., 0.15 → 0.10 F1 difference)
- [ ] Worst-group F1 improved by ≥10% (e.g., 0.45 → 0.50)
- [ ] Demographic parity difference < 0.10
- [ ] Equalized odds difference < 0.10
- [ ] Calibration disparity (ECE) < 0.05

**Performance Maintenance:**
- [ ] Overall F1 degradation < 3% (e.g., 0.72 → 0.70)
- [ ] AUC maintained within 2% (e.g., 0.90 → 0.88)

**Statistical Validation:**
- [ ] n=100 runs per configuration
- [ ] Bootstrap confidence intervals (95%) for all metrics
- [ ] Statistically significant fairness improvement (p < 0.05)

**Code Quality:**
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Production-ready (matches SynergyCBM quality)

---

## Dependencies

**New Python Packages:**
```
aequitas==0.42.0          # Fairness audit toolkit
fairlearn==0.8.0          # Additional fairness metrics
```

**Existing Packages (from SynergyCBM):**
- PyTorch 2.0+
- timm
- scikit-learn
- pandas, numpy
- matplotlib, seaborn

---

## Next Steps

To begin implementation:

1. **Review this plan** - Confirm approach aligns with research goals
2. **Create directory structure** - Run initialization script
3. **Implement Phase 1** - Core fairness metrics (can develop independently)
4. **Implement Phase 2** - Adversarial discriminator (requires metrics)
5. **Implement Phase 3** - Model extensions (requires discriminator)
6. **Proceed sequentially** - Each phase builds on previous

**Ready to proceed?** Let me know which phase to start with, or if you'd like to modify the plan.
