# Fairness-Aware CBM: Revised Implementation Plan

## Key Discovery: Dataloader Already Supports Fitzpatrick Labels! ✅

The existing `src/data/dataloader.py` **already returns Fitzpatrick labels** in the 4-tuple:
```python
return image, concept_labels, binary_label, fitzpatrick
```

This means we can **directly use the existing dataloader** without modifications!

---

## Experimental Design: 4 Model Types

Based on your requirements, we'll compare:

### 1. **Direct Classifier** (Baseline - No Interpretability)
- **Architecture:** Image → Backbone → Binary Prediction
- **Training:** Standard supervised learning
- **Purpose:** Upper bound on performance without interpretability

### 2. **Standard CBM** (Baseline - No Curriculum)
- **Architecture:** Image → Backbone → Concepts → Binary Prediction
- **Training:** All 23 concepts trained jointly from the start
- **Purpose:** Test if curriculum learning helps

### 3. **Curriculum CBM** (Production Model)
- **Architecture:** Image → Backbone → Concepts → Binary Prediction
- **Training:** 3-phase curriculum (Primary → Secondary → Color)
- **Differential loss weighting:** New concepts (0.1) vs. existing (1.0)
- **Purpose:** Best non-fairness model (current production)

### 4. **Fair Curriculum CBM** (New - Fairness-Aware) 🎯
- **Architecture:** Image → Backbone → Concepts + Adversarial Discriminator → Binary Prediction
- **Training:** 3-phase curriculum **+ fairness constraints**
- **Fairness mechanisms:**
  - Adversarial debiasing (gradient reversal)
  - Demographic parity loss
  - Equalized odds loss
- **Purpose:** Reduce performance disparities across Fitzpatrick types

---

## Simplified Implementation (Leveraging Existing Code)

### Step 1: Copy Existing Models (No Dataloader Changes Needed!)

**Models already exist in `src/models/`:**
- ✅ `direct_classifier.py` - Direct Classifier
- ✅ `standard_cbm.py` - Standard CBM (no curriculum)
- ✅ `minimal_curriculum_cbm.py` - Curriculum CBM

**Only need to create:**
- ⏳ `fairness/src/models/fairness_aware_cbm.py` - Fair Curriculum CBM

### Step 2: Create Fair Curriculum CBM

**File:** `fairness/src/models/fairness_aware_cbm.py`

**Base:** Extend `MinimalCurriculumCBM` (already has curriculum logic)

**Additions:**
1. Adversarial discriminator for Fitzpatrick prediction
2. Modified loss function with fairness terms
3. Gradient reversal training

**Key difference from parent:** Add fairness losses to existing curriculum framework

### Step 3: Adapt Training Script

**File:** `fairness/scripts/train_all_models.py`

**Purpose:** Train all 4 models in one script for fair comparison

```python
def train_all_models(args):
    """Train all 4 model types for comparison."""
    
    # Load data (already returns Fitzpatrick labels!)
    train_loader = get_dataloader(
        csv_file=args.train_csv,
        img_dir=args.data_dir,
        lesion_cols=concept_names,
        label_type='concept',  # Returns (img, concepts, label, fitzpatrick)
        batch_size=args.batch_size
    )
    
    results = {}
    
    # 1. Train Direct Classifier
    print("Training Direct Classifier...")
    direct_model = DirectClassifier(backbone=args.backbone)
    results['direct'] = train_direct_classifier(direct_model, train_loader, val_loader)
    
    # 2. Train Standard CBM (no curriculum)
    print("Training Standard CBM...")
    standard_model = NoCurriculumCBM(backbone=args.backbone, concept_names=concept_names)
    results['standard_cbm'] = train_cbm(standard_model, train_loader, val_loader)
    
    # 3. Train Curriculum CBM
    print("Training Curriculum CBM...")
    curriculum_model = MinimalCurriculumCBM(backbone=args.backbone, concept_names=concept_names)
    results['curriculum_cbm'] = train_curriculum_cbm(curriculum_model, train_loader, val_loader)
    
    # 4. Train Fair Curriculum CBM
    print("Training Fair Curriculum CBM...")
    fair_model = FairnessAwareCBM(
        backbone=args.backbone,
        concept_names=concept_names,
        num_groups=6,  # Fitzpatrick I-VI
        fairness_lambda=args.fairness_lambda,
        adversarial_lambda=args.adversarial_lambda
    )
    results['fair_curriculum_cbm'] = train_fair_curriculum_cbm(fair_model, train_loader, val_loader)
    
    return results
```

### Step 4: Evaluation with Fairness Metrics

**Key insight:** Evaluate **all 4 models** on fairness metrics, not just Fair Curriculum CBM

```python
def evaluate_all_models_on_fairness(models, test_loader):
    """Evaluate all models on both performance and fairness."""
    
    for model_name, model in models.items():
        print(f"\n=== Evaluating {model_name} ===")
        
        # Collect predictions
        all_preds = []
        all_labels = []
        all_groups = []
        
        for images, concepts, labels, fitzpatrick in test_loader:
            # Forward pass (handle different model types)
            if model_name == 'direct':
                preds = model(images)
            else:  # CBM variants
                _, preds = model(images)
            
            all_preds.extend(torch.sigmoid(preds).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_groups.extend(fitzpatrick.cpu().numpy())
        
        # Standard metrics
        f1 = compute_f1(all_preds, all_labels)
        auc = compute_auc(all_preds, all_labels)
        
        # Fairness metrics (for ALL models)
        fairness_results = compute_all_fairness_metrics(
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_groups)
        )
        
        print(f"  F1: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Worst-group F1: {fairness_results['worst_group']['worst_group_f1']:.4f}")
        print(f"  Performance Gap: {fairness_results['worst_group']['performance_gap']:.4f}")
        print(f"  Demographic Parity: {fairness_results['demographic_parity']['max_disparity']:.4f}")
        print(f"  Equalized Odds: {fairness_results['equalized_odds']['equalized_odds_difference']:.4f}")
```

---

## Revised File Structure

### Files to Create (Minimal Changes!)

```
fairness/
├── src/
│   ├── models/
│   │   └── fairness_aware_cbm.py        # NEW: Only this needs to be created!
│   └── utils/
│       ├── fairness_metrics.py          # ✅ Already created
│       └── adversarial_debiasing.py     # ✅ Already created
├── scripts/
│   ├── train_all_models.py              # NEW: Train all 4 models
│   ├── evaluate_fairness_comparison.py  # NEW: Compare all 4 models
│   └── analyze_fairness_results.py      # NEW: Multi-run analysis
└── [docs, results, etc.]                # ✅ Already created
```

### Files to Copy Directly (No Changes)

```bash
# These already exist and work with Fitzpatrick labels!
cp src/models/direct_classifier.py fairness/src/models/
cp src/models/standard_cbm.py fairness/src/models/
cp src/models/minimal_curriculum_cbm.py fairness/src/models/
cp src/utils/metrics.py fairness/src/utils/
cp src/utils/calibration.py fairness/src/utils/
cp src/utils/model_utils.py fairness/src/utils/
```

---

## Training Loop Modifications

### Standard Training (Direct, Standard CBM, Curriculum CBM)

**No changes needed** - existing code already works!

```python
for images, concepts, labels, fitzpatrick in train_loader:
    # Standard training (ignore fitzpatrick during training)
    if model_type == 'direct':
        preds = model(images)
        loss = criterion(preds, labels)
    else:  # CBM
        concept_preds, binary_preds = model(images)
        loss = compute_cbm_loss(concept_preds, binary_preds, concepts, labels)
    
    loss.backward()
    optimizer.step()
```

### Fair Curriculum CBM Training (NEW)

**Only difference:** Use Fitzpatrick labels in loss computation

```python
for images, concepts, labels, fitzpatrick in train_loader:
    # Get features for adversarial training
    concept_logits, binary_logits, concept_features = model(images, return_features=True)
    
    # Compute fairness-aware loss
    losses = model.compute_fairness_loss(
        concept_logits=concept_logits,
        binary_logits=binary_logits,
        concept_features=concept_features,
        concept_labels=concepts,
        binary_labels=labels,
        group_labels=fitzpatrick,  # ← Use existing Fitzpatrick labels!
        active_concepts=phase.concepts
    )
    
    losses['total_loss'].backward()
    optimizer.step()
```

---

## Expected Results Comparison Table

| Model                | Overall F1 | Worst-Group F1 | Performance Gap | Dem. Parity | Eq. Odds | Interpretable |
|---------------------|-----------|----------------|-----------------|-------------|----------|---------------|
| Direct Classifier   | 0.75      | 0.42           | 0.33           | 0.28        | 0.25     | ❌            |
| Standard CBM        | 0.72      | 0.45           | 0.27           | 0.25        | 0.23     | ✅            |
| Curriculum CBM      | 0.73      | 0.46           | 0.27           | 0.24        | 0.22     | ✅            |
| **Fair Curriculum** | **0.71**  | **0.56**       | **0.15**       | **0.10**    | **0.12** | ✅            |

**Key Findings (Expected):**
- Direct classifier: Highest overall performance, worst fairness
- Standard CBM: Interpretable, similar fairness to direct
- Curriculum CBM: Slight improvement in overall performance
- **Fair Curriculum CBM:** Significant fairness improvement (-44% gap) with minimal performance cost (-3%)

---

## Implementation Priority (Revised)

### Week 1: Core Fair Curriculum CBM
1. ✅ Fairness metrics module (already done)
2. ✅ Adversarial discriminator (already done)
3. ⏳ **Create `FairnessAwareCBM` class** (extend MinimalCurriculumCBM)
4. ⏳ Test forward/backward passes with Fitzpatrick labels

### Week 2: Training Infrastructure
5. ⏳ **Create `train_all_models.py`** (train all 4 models)
6. ⏳ Test single run of each model type
7. ⏳ Validate Fitzpatrick label usage in Fair Curriculum CBM

### Week 3: Evaluation & Comparison
8. ⏳ **Create `evaluate_fairness_comparison.py`** (compare all 4)
9. ⏳ Generate fairness comparison visualizations
10. ⏳ Create fairness report template

### Week 4: Multi-Run Validation
11. ⏳ Run n=100 experiments per model type (400 total)
12. ⏳ Statistical significance testing (Fair vs. Curriculum)
13. ⏳ Aggregate results and confidence intervals

### Week 5: Analysis & Documentation
14. ⏳ Cross-model fairness analysis
15. ⏳ Performance vs. fairness tradeoff curves
16. ⏳ Final documentation and results write-up

---

## Quick Start (Updated)

### Step 1: Copy Existing Models

```bash
cd /home/csc29/projects/SynergyCBM/fairness

# Copy existing models (no changes needed)
cp ../src/models/direct_classifier.py src/models/
cp ../src/models/standard_cbm.py src/models/
cp ../src/models/minimal_curriculum_cbm.py src/models/

# Copy utilities
cp ../src/utils/metrics.py src/utils/
cp ../src/utils/calibration.py src/utils/
cp ../src/utils/model_utils.py src/utils/
```

### Step 2: Create FairnessAwareCBM (Template)

```python
# fairness/src/models/fairness_aware_cbm.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.minimal_curriculum_cbm import MinimalCurriculumCBM
from fairness.src.models.adversarial_discriminator import AdversarialDiscriminator
from fairness.src.utils.adversarial_debiasing import compute_combined_fairness_loss
import torch
import torch.nn as nn
import torch.nn.functional as F


class FairnessAwareCBM(MinimalCurriculumCBM):
    """
    Curriculum CBM with fairness constraints.
    
    Extends MinimalCurriculumCBM with:
    - Adversarial discriminator for Fitzpatrick prediction
    - Fairness-aware loss computation
    - Gradient reversal training
    """
    
    def __init__(self, *args, num_groups=6, fairness_lambda=1.0, 
                 adversarial_lambda=0.1, fairness_type='equalized_odds', **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add adversarial discriminator
        self.adversarial_discriminator = AdversarialDiscriminator(
            input_dim=512,
            hidden_dims=[256, 128],
            num_groups=num_groups,
            dropout=0.3
        ).to(self.device)
        
        # Fairness hyperparameters
        self.fairness_lambda = fairness_lambda
        self.adversarial_lambda = adversarial_lambda
        self.fairness_type = fairness_type
        self.num_groups = num_groups
        
        # Adversarial alpha (increased during training)
        self.adversarial_alpha = 0.0
        
    def forward(self, x, return_features=False):
        """Forward pass with optional feature extraction."""
        # Extract backbone features
        features = self.backbone(x)
        
        # Handle different feature shapes (copied from parent)
        if len(features.shape) == 4:
            if features.shape[-1] > features.shape[1]:
                features = features.permute(0, 3, 1, 2)
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        elif len(features.shape) == 3:
            features = features[:, 0, :]
        
        # Concept prediction (get features before final layer)
        concept_features = features  # Use backbone output as concept features
        concept_logits = self.concept_layer(features)
        
        # Binary prediction
        concept_probs = torch.sigmoid(concept_logits)
        binary_logits = self.binary_classifier(concept_probs)
        
        if return_features:
            return concept_logits, binary_logits, concept_features
        return concept_logits, binary_logits
    
    def compute_fairness_loss(self, concept_logits, binary_logits, concept_features,
                             concept_labels, binary_labels, group_labels,
                             active_concepts, new_concepts=None, 
                             new_concept_weight_multiplier=1.0):
        """
        Compute loss with fairness constraints.
        
        Returns dict with: concept_loss, binary_loss, fairness_loss, 
                          adversarial_loss, total_loss
        """
        # Get base losses from parent class
        base_losses = super().compute_loss(
            concept_logits, binary_logits,
            concept_labels, binary_labels,
            active_concepts,
            new_concepts=new_concepts,
            new_concept_weight_multiplier=new_concept_weight_multiplier
        )
        
        # Compute fairness loss
        binary_probs = torch.sigmoid(binary_logits).squeeze()
        fairness_loss = compute_combined_fairness_loss(
            predictions=binary_probs,
            labels=binary_labels,
            groups=group_labels.long(),
            fairness_type=self.fairness_type
        )
        
        # Compute adversarial loss (group prediction with gradient reversal)
        group_logits = self.adversarial_discriminator(concept_features, alpha=self.adversarial_alpha)
        adversarial_loss = F.cross_entropy(group_logits, group_labels.long())
        
        # Total loss
        total_loss = (
            base_losses['total_loss'] +
            self.fairness_lambda * fairness_loss -
            self.adversarial_lambda * adversarial_loss
        )
        
        return {
            'concept_loss': base_losses['concept_loss'],
            'binary_loss': base_losses['binary_loss'],
            'fairness_loss': fairness_loss,
            'adversarial_loss': adversarial_loss,
            'total_loss': total_loss
        }
    
    def update_adversarial_alpha(self, epoch, max_epochs):
        """Update gradient reversal strength (Ganin schedule)."""
        progress = epoch / max_epochs
        self.adversarial_alpha = 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0
```

### Step 3: Test the Model

```python
# Test script
from fairness.src.models.fairness_aware_cbm import FairnessAwareCBM
import torch

# Create model
model = FairnessAwareCBM(
    backbone='swin_base_patch4_window7_224',
    concept_names=['Papule', 'Plaque', 'Nodule'],
    num_groups=6,
    fairness_lambda=1.0,
    adversarial_lambda=0.1
)

# Test forward pass
batch = {
    'images': torch.randn(4, 3, 224, 224),
    'concepts': torch.randint(0, 2, (4, 3)).float(),
    'labels': torch.randint(0, 2, (4,)).float(),
    'fitzpatrick': torch.randint(1, 7, (4,)).float()
}

# Forward
concept_logits, binary_logits, features = model(batch['images'], return_features=True)

# Compute loss
losses = model.compute_fairness_loss(
    concept_logits, binary_logits, features,
    batch['concepts'], batch['labels'], batch['fitzpatrick'],
    active_concepts=['Papule', 'Plaque', 'Nodule']
)

print("Losses:", {k: v.item() for k, v in losses.items()})
```

---

## Summary of Changes

### What We DON'T Need (Thanks to Existing Dataloader!)
- ❌ Custom Fitzpatrick dataloader
- ❌ Fitzpatrick annotation CSV (already in main dataset)
- ❌ Stratified sampler (can add later if needed)
- ❌ Data preprocessing scripts

### What We DO Need
- ✅ Fairness metrics module (already created)
- ✅ Adversarial discriminator (already created)
- ✅ Adversarial debiasing losses (already created)
- ⏳ FairnessAwareCBM model (extend MinimalCurriculumCBM)
- ⏳ Training script for all 4 models
- ⏳ Evaluation/comparison scripts
- ⏳ Multi-run experiment infrastructure

---

**Ready to implement FairnessAwareCBM?** The template above provides the complete implementation!
