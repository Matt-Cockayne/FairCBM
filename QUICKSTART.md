# Fairness-Aware CBM: Implementation Quick Start

## Current Status

✅ **Completed (Tasks 1-5):**
- Integration plan with complete specifications
- Directory structure created
- Fairness metrics module (580 lines)
- Adversarial discriminator (270 lines)
- Adversarial debiasing losses (380 lines)

⏳ **Next Steps (Tasks 6-8):**
- Fairness-aware CBM model
- Fitzpatrick dataloader
- Training & evaluation scripts

---

## Step 1: Create Fairness-Aware CBM Model

### File: `fairness/src/models/fairness_aware_cbm.py`

**Copy this template and fill in the TODOs:**

```python
#!/usr/bin/env python3
"""
Fairness-Aware Curriculum CBM

Extends MinimalCurriculumCBM with:
- Adversarial discriminator for group prediction
- Fairness-aware loss computation
- Gradient reversal training
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.minimal_curriculum_cbm import MinimalCurriculumCBM
from fairness.src.models.adversarial_discriminator import (
    AdversarialDiscriminator,
    AdversarialAlphaScheduler
)
from fairness.src.utils.adversarial_debiasing import compute_combined_fairness_loss
import torch
import torch.nn as nn


class FairnessAwareCBM(MinimalCurriculumCBM):
    """CBM with fairness constraints and adversarial debiasing."""
    
    def __init__(self, *args, 
                 num_groups: int = 6,
                 fairness_lambda: float = 1.0,
                 adversarial_lambda: float = 0.1,
                 fairness_type: str = 'equalized_odds',
                 **kwargs):
        """
        Initialize fairness-aware CBM.
        
        Args:
            num_groups: Number of demographic groups (6 Fitzpatrick types)
            fairness_lambda: Weight for fairness loss
            adversarial_lambda: Weight for adversarial loss (negative for reversal)
            fairness_type: 'demographic_parity', 'equalized_odds', or 'combined'
        """
        super().__init__(*args, **kwargs)
        
        # TODO: Add adversarial discriminator
        # self.adversarial_discriminator = AdversarialDiscriminator(...)
        
        # TODO: Store fairness hyperparameters
        # self.fairness_lambda = fairness_lambda
        # ...
        
        # TODO: Initialize alpha scheduler
        # self.alpha_scheduler = AdversarialAlphaScheduler(...)
        
        pass
    
    def forward(self, x, return_features=False):
        """
        Forward pass with optional feature extraction.
        
        TODO: Modify to return intermediate concept features for adversarial training
        """
        # TODO: Get features before final concept layer
        # features = self.backbone(x)
        # concept_features = self.concept_layer[0](features)
        # concept_logits = self.concept_layer[1](concept_features)
        
        pass
    
    def compute_fairness_loss(self, concept_logits, binary_logits, concept_features,
                             concept_labels, binary_labels, group_labels,
                             active_concepts, phase_name='all'):
        """
        Compute loss with fairness constraints.
        
        TODO: Implement fairness-aware loss computation
        """
        # TODO: Get base losses from parent class
        # base_losses = self.compute_loss(...)
        
        # TODO: Compute fairness loss
        # binary_probs = torch.sigmoid(binary_logits).squeeze()
        # fairness_loss = compute_combined_fairness_loss(...)
        
        # TODO: Compute adversarial loss
        # group_logits = self.adversarial_discriminator(concept_features, alpha=self.current_alpha)
        # adversarial_loss = F.cross_entropy(group_logits, group_labels)
        
        # TODO: Combine losses
        # total_loss = base_losses['total_loss'] + fairness_lambda * fairness_loss - adversarial_lambda * adversarial_loss
        
        pass
```

**See:** `fairness/INTEGRATION_PLAN.md` Section 2.2 for complete implementation

---

## Step 2: Create Fitzpatrick DataLoader

### File: `fairness/src/data/fitzpatrick_dataloader.py`

```python
#!/usr/bin/env python3
"""
Fitzpatrick-Aware DataLoader

Extends SkinCapDataset to include Fitzpatrick skin type annotations.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.data.dataloader import SkinCapDataset
import pandas as pd
import logging


class SkinCapFitzpatrickDataset(SkinCapDataset):
    """SkinCap dataset with Fitzpatrick annotations."""
    
    def __init__(self, csv_file, img_dir, fitzpatrick_csv, 
                 lesion_cols, label_type='concept', transform=None):
        """
        TODO: Load Fitzpatrick annotations and merge with main dataset
        """
        super().__init__(csv_file, img_dir, lesion_cols, label_type, transform)
        
        # TODO: Load Fitzpatrick CSV
        # fitz_df = pd.read_csv(fitzpatrick_csv)
        
        # TODO: Merge with self.df
        # self.df = self.df.merge(fitz_df, ...)
        
        pass
    
    def __getitem__(self, idx):
        """Returns (image, concept_labels, binary_label, fitzpatrick_type)."""
        # TODO: Call parent __getitem__
        # img, concept_labels, binary_label = super().__getitem__(idx)
        
        # TODO: Get Fitzpatrick type
        # fitzpatrick_type = int(self.df.iloc[idx]['fitzpatrick_type']) - 1  # 0-indexed
        
        # TODO: Return 4-tuple
        # return img, concept_labels, binary_label, fitzpatrick_type
        
        pass
```

**See:** `fairness/INTEGRATION_PLAN.md` Section 5.2 for complete implementation

---

## Step 3: Create Training Script

### File: `fairness/scripts/train_fairness_cbm.py`

**Copy from:** `scripts/test_minimal_curriculum.py` (1226 lines)

**Key modifications:**

1. **Import fairness modules:**
```python
from fairness.src.models.fairness_aware_cbm import FairnessAwareCBM
from fairness.src.data.fitzpatrick_dataloader import SkinCapFitzpatrickDataset
from fairness.src.utils.fairness_metrics import compute_all_fairness_metrics
```

2. **Add arguments:**
```python
parser.add_argument('--fairness_lambda', type=float, default=1.0)
parser.add_argument('--adversarial_lambda', type=float, default=0.1)
parser.add_argument('--fitzpatrick_csv', type=str, required=True)
parser.add_argument('--use_group_balanced_sampling', action='store_true')
```

3. **Use Fitzpatrick dataset:**
```python
train_dataset = SkinCapFitzpatrickDataset(
    csv_file=args.train_csv,
    img_dir=args.data_dir,
    fitzpatrick_csv=args.fitzpatrick_csv,
    lesion_cols=concept_names,
    label_type='concept'
)
```

4. **Modified training loop:**
```python
for epoch in range(num_epochs):
    model.update_adversarial_alpha(epoch, num_epochs)
    
    for images, concept_labels, binary_labels, group_labels in train_loader:
        # Forward with features
        concept_logits, binary_logits, concept_features = model(images, return_features=True)
        
        # Compute fairness-aware loss
        losses = model.compute_fairness_loss(
            concept_logits, binary_logits, concept_features,
            concept_labels, binary_labels, group_labels,
            active_concepts=phase.concepts,
            phase_name=phase.name
        )
        
        # Backward
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()
```

**See:** `fairness/INTEGRATION_PLAN.md` Section 6.1 for complete modifications

---

## Step 4: Create Fitzpatrick Annotations

### File: `fairness/data/fitzpatrick_annotations.csv`

**Required format:**
```csv
image_id,fitzpatrick_type,fitzpatrick_label
ISIC_0000001,3,III
ISIC_0000002,2,II
ISIC_0000003,5,V
...
```

**Options:**
1. **If you have annotations:** Convert to this format
2. **If you need to create annotations:** Use dermatology literature or expert labeling
3. **For testing:** Generate synthetic annotations (random or stratified)

**Test with synthetic data:**
```python
import pandas as pd
import numpy as np

# Get image IDs from existing dataset
df = pd.read_csv('/path/to/skincap_train.csv')
image_ids = df['lesion_id'].values  # Adjust column name

# Generate random Fitzpatrick types (for testing only)
np.random.seed(42)
fitzpatrick_types = np.random.randint(1, 7, len(image_ids))
fitzpatrick_labels = ['I', 'II', 'III', 'IV', 'V', 'VI']

fitz_df = pd.DataFrame({
    'image_id': image_ids,
    'fitzpatrick_type': fitzpatrick_types,
    'fitzpatrick_label': [fitzpatrick_labels[t-1] for t in fitzpatrick_types]
})

fitz_df.to_csv('fairness/data/fitzpatrick_annotations.csv', index=False)
print(f"Created annotations for {len(fitz_df)} images")
```

---

## Step 5: Test Implementation

### Test Fairness Metrics
```bash
cd /home/csc29/projects/SynergyCBM
python fairness/src/utils/fairness_metrics.py
```

Expected output:
```
Fairness Summary:
  demographic_parity_difference: 0.xxx
  equalized_odds_difference: 0.xxx
  performance_gap: 0.xxx
  ...
```

### Test Adversarial Components
```bash
python fairness/src/models/adversarial_discriminator.py
python fairness/src/utils/adversarial_debiasing.py
```

### Test Model Forward Pass (After Step 1)
```python
from fairness.src.models.fairness_aware_cbm import FairnessAwareCBM
import torch

model = FairnessAwareCBM(
    backbone='swin_base_patch4_window7_224',
    concept_names=['Papule', 'Plaque', 'Nodule'],  # Subset for testing
    num_groups=6,
    fairness_lambda=1.0,
    adversarial_lambda=0.1
)

# Test forward pass
x = torch.randn(4, 3, 224, 224)
concept_logits, binary_logits, features = model(x, return_features=True)

print(f"Concept logits: {concept_logits.shape}")
print(f"Binary logits: {binary_logits.shape}")
print(f"Features: {features.shape}")
```

### Test DataLoader (After Step 2)
```python
from fairness.src.data.fitzpatrick_dataloader import SkinCapFitzpatrickDataset
from torch.utils.data import DataLoader

dataset = SkinCapFitzpatrickDataset(
    csv_file='/path/to/train.csv',
    img_dir='/path/to/images/',
    fitzpatrick_csv='fairness/data/fitzpatrick_annotations.csv',
    lesion_cols=['Papule', 'Plaque'],
    label_type='concept'
)

loader = DataLoader(dataset, batch_size=8)
images, concepts, labels, groups = next(iter(loader))

print(f"Images: {images.shape}")
print(f"Concepts: {concepts.shape}")
print(f"Labels: {labels.shape}")
print(f"Groups: {groups.shape}")
print(f"Unique groups: {groups.unique()}")
```

---

## Step 6: Run First Experiment

### Basic Training Command
```bash
cd /home/csc29/projects/SynergyCBM

python fairness/scripts/train_fairness_cbm.py \
    --backbone swin_base_patch4_window7_224 \
    --curriculum \
    --fairness_lambda 1.0 \
    --adversarial_lambda 0.1 \
    --fitzpatrick_csv fairness/data/fitzpatrick_annotations.csv \
    --random_seed 42 \
    --data_dir /path/to/SkinCap/ \
    --train_csv /path/to/skincap_train.csv \
    --val_csv /path/to/skincap_val.csv \
    --test_csv /path/to/skincap_test.csv \
    --output_dir fairness/results/test_run/
```

### Expected Output
```
Epoch 1/50, Phase: primary_lesions
  Concept Loss: 0.xxx
  Binary Loss: 0.xxx
  Fairness Loss: 0.xxx
  Adversarial Loss: 0.xxx
  Total Loss: 0.xxx
  Adversarial Alpha: 0.02
...
```

---

## Step 7: Evaluate Fairness

### Create Evaluation Script: `fairness/scripts/evaluate_fairness.py`

```python
#!/usr/bin/env python3
import torch
import numpy as np
from fairness.src.models.fairness_aware_cbm import FairnessAwareCBM
from fairness.src.data.fitzpatrick_dataloader import SkinCapFitzpatrickDataset
from fairness.src.utils.fairness_metrics import compute_all_fairness_metrics, compute_fairness_summary
from torch.utils.data import DataLoader

# Load model
model = torch.load('fairness/results/test_run/best_model.pth')
model.eval()

# Load test data
test_dataset = SkinCapFitzpatrickDataset(...)
test_loader = DataLoader(test_dataset, batch_size=32)

# Collect predictions
all_preds, all_labels, all_groups = [], [], []
with torch.no_grad():
    for images, _, labels, groups in test_loader:
        _, binary_logits = model(images)
        probs = torch.sigmoid(binary_logits).cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.numpy())
        all_groups.extend(groups.numpy())

# Compute fairness metrics
metrics = compute_all_fairness_metrics(
    np.array(all_preds),
    np.array(all_labels),
    np.array(all_groups)
)

summary = compute_fairness_summary(metrics)
print("\nFairness Summary:")
for key, value in summary.items():
    print(f"  {key}: {value}")
```

---

## Troubleshooting

### Issue: Import errors
**Solution:** Ensure parent directory in `sys.path`:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
```

### Issue: Fitzpatrick CSV not found
**Solution:** Create synthetic annotations (see Step 4) or check path

### Issue: Gradient reversal not working
**Solution:** Test gradient flow:
```python
x = torch.randn(4, 10, requires_grad=True)
y = GradientReversalLayer.apply(x, 1.0)
y.sum().backward()
print(f"Gradient reversed: {(x.grad < 0).all()}")  # Should be True
```

### Issue: OOM during training
**Solution:** Reduce batch size or use gradient accumulation:
```python
# In training loop
loss.backward()
if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

---

## Progress Checklist

- [ ] Step 1: FairnessAwareCBM implemented
- [ ] Step 2: Fitzpatrick dataloader created
- [ ] Step 3: Training script adapted
- [ ] Step 4: Fitzpatrick annotations prepared
- [ ] Step 5: All tests passing
- [ ] Step 6: First experiment running
- [ ] Step 7: Fairness evaluation working
- [ ] Baseline comparison (standard vs. fairness CBM)
- [ ] Multi-run validation (n=100)
- [ ] Hyperparameter sweep
- [ ] Final analysis and documentation

---

## Key Resources

### Documentation
- `INTEGRATION_PLAN.md` - Complete specifications
- `PROJECT_STATUS.md` - Current progress
- `README.md` - Usage guide

### Parent Codebase
- `../src/models/minimal_curriculum_cbm.py` - Base model (883 lines)
- `../src/data/dataloader.py` - Base dataloader (360 lines)
- `../scripts/test_minimal_curriculum.py` - Base training script (1226 lines)

### Fairness Modules (Already Complete)
- `src/utils/fairness_metrics.py` - 6 fairness metrics
- `src/models/adversarial_discriminator.py` - Gradient reversal
- `src/utils/adversarial_debiasing.py` - Fairness losses

---

## Support

**Questions?** Refer to:
1. `INTEGRATION_PLAN.md` for detailed specifications
2. Parent `code-base-description.md` for SynergyCBM architecture
3. Existing module docstrings and tests

**Ready to start?** Begin with Step 1 (FairnessAwareCBM model)!
