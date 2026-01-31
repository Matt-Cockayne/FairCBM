# Ablation Study Integration - Implementation Summary

## Current Status

**File Restoration Complete**: The corrupted `fair_curriculum_cbm.py` has been restored from git.

**Remaining Work**: The ablation parameter support needs to be integrated into the restored file.

## Changes Made Successfully

### 1. Updated Training Script
✅ [train_all_models.py](scripts/train_all_models.py)
- Added `--ablation_key` argument
- Created ablation config mapping in `create_model()`
- Updated model instantiation to pass ablation parameters

### 2. Updated SLURM Script  
✅ [run_ablation_study.slurm](slurm/run_ablation_study.slurm)
- Changed to use `train_all_models.py` instead of `train_ablation.py`
- Fixed hyperparameters: `FAIRNESS_LAMBDA=0.1`, `ADVERSARIAL_LAMBDA=0.01`
- Updated ablation list: `["full_model", "no_phase2", "no_phase3", "no_phase4", "no_adversarial"]`

### 3. Created Analysis Scripts
✅ [analyze_consistent_ablation.py](scripts/analyze_consistent_ablation.py)
- Analyzes results from new ablation structure
- Computes statistics per ablation configuration
- Shows phase contributions

✅ [test_ablation_config.py](scripts/test_ablation_config.py)
- Tests ablation configuration functionality
- Verifies phase disabling works correctly

### 4. Created Documentation
✅ [ABLATION_FIX.md](docs/ABLATION_FIX.md)
- Complete explanation of the problem
- Details of the solution approach
- Usage instructions

## Remaining Implementation Steps

The following changes still need to be made to [src/models/fair_curriculum_cbm.py](src/models/fair_curriculum_cbm.py):

### Step 1: Modify `PhasedFairnessLoss.__init__()` (Line ~267)

```python
def __init__(self, total_epochs: int, num_groups: int = 6, 
             disabled_phases: Optional[List[int]] = None):
    """
    Initialize PhasedFairnessLoss.
    
    Args:
        total_epochs: Total training epochs
        num_groups: Number of demographic groups
        disabled_phases: List of phase numbers (1-4) to disable for ablation
    """
    super().__init__()
    self.total_epochs = total_epochs
    self.num_groups = num_groups
    self.disabled_phases = disabled_phases or []
```

### Step 2: Modify `PhasedFairnessLoss.forward()` (Line ~298)

Add phase number detection and fallback logic at the start of `forward()`:

```python
def forward(self, predictions, labels, groups, epoch):
    phase = self._get_phase(epoch)
    progress = epoch / self.total_epochs
    
    # Determine current phase number
    if progress <= 0.25:
        phase_num = 1
    elif progress <= 0.50:
        phase_num = 2
    elif progress <= 0.75:
        phase_num = 3
    else:
        phase_num = 4
    
    # If current phase is disabled, use previous active phase behavior
    if phase_num in self.disabled_phases:
        # Fallback logic:
        # Phase 2 disabled → use Phase 1 (no fairness)
        # Phase 3 disabled → continue Phase 2 (DP only)
        # Phase 4 disabled → continue Phase 3 (DP + EO)
        if phase_num == 2 and 1 not in self.disabled_phases:
            return torch.tensor(0.0, device=predictions.device)
        elif phase_num == 3 and 2 not in self.disabled_phases:
            phase = 'demographic_parity'
        elif phase_num == 4 and 3 not in self.disabled_phases:
            phase = 'equalized_odds'
        # ... (more fallback cases)
    
    # Rest of forward() method continues...
```

### Step 3: Modify `FairCurriculumCBM.__init__()` (Line ~477)

```python
def __init__(self, num_concepts: int, backbone: str = 'swin',
             num_groups: int = 6, fairness_lambda: float = 0.1,
             adversarial_lambda_target: float = 0.01,
             dropout_rate: float = 0.1, device: torch.device = None,
             concept_names: Optional[List[str]] = None,
             disabled_phases: Optional[List[int]] = None,
             disable_adversarial: bool = False):
    """
    Initialize Fair Curriculum CBM.
    
    Args:
        ...existing args...
        disabled_phases: List of phase numbers (1-4) to disable for ablation
        disable_adversarial: Disable adversarial debiasing for ablation
    """
    super().__init__(...)
    
    self.disabled_phases = disabled_phases or []
    self.disable_adversarial = disable_adversarial
    
    # Pass ablation config to fairness loss
    self.fairness_loss_fn = PhasedFairnessLoss(
        total_epochs=100,
        num_groups=num_groups,
        disabled_phases=disabled_phases  # NEW
    )
```

### Step 4: Modify `FairCurriculumCBM.compute_adversarial_lambda()` (Line ~603)

```python
def compute_adversarial_lambda(self, epoch: int, total_epochs: int) -> float:
    """Compute current adversarial lambda with warmup."""
    # If adversarial is disabled for ablation, return 0
    if self.disable_adversarial:
        return 0.0
    
    progress = epoch / total_epochs
    
    # If Phase 3 is disabled, adversarial never activates
    if 3 in self.disabled_phases:
        return 0.0
    
    # Rest of method continues...
```

## Quick Fix Script

To save time, I can create a Python script that applies all these changes automatically. Would you like me to:

1. **Manual approach**: I'll make each change step-by-step using `replace_string_in_file`
2. **Script approach**: Create `apply_ablation_changes.py` that does all changes at once
3. **Review approach**: Show you the exact diffs and you apply them manually

## Testing After Implementation

Once the changes are applied, run:

```bash
cd /home/csc29/projects/SynergyCBM/FairCBM

# Test 1: Check imports
python -c "from src.models.fair_curriculum_cbm import FairCurriculumCBM; print('Import OK')"

# Test 2: Run ablation config tests
python scripts/test_ablation_config.py

# Test 3: Verify with real training (quick)
python scripts/train_all_models.py \
    --model_type fair_curriculum_cbm \
    --ablation_key full_model \
    --exp_name test_ablation \
    --epochs 2 \
    --batch_size 8
```

## Next Steps After Testing

1. **Submit ablation study**: `sbatch slurm/run_ablation_study.slurm`
2. **Monitor progress**: `squeue -u $USER | grep ablation`
3. **Analyze results**: `python scripts/analyze_consistent_ablation.py --exp_name ablation_study_20260108`
4. **Verify baseline**: Check that `full_model` matches main experiment (F1 ≈ 0.653, Gap ≈ 0.255)

## Summary

- ✅ Training infrastructure updated
- ✅ SLURM script fixed
- ✅ Analysis scripts created
- ✅ Documentation complete
- ⏳ Model class modifications pending (4 changes needed in `fair_curriculum_cbm.py`)

The core architecture is ready, we just need to add the ablation parameters to the model class itself.
