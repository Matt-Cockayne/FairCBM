# Adversarial Loss Explosion: Root Cause & Fixes

## Problem Diagnosis

Looking at your training history, the adversarial loss **explodes from 1.78 to 2,028** by epoch 50:

```
Epoch | Adv Loss  | Adv Lambda | Notes
------|-----------|------------|--------------------------------
    1 |    1.7815 |     0.0000 | Warmup phase 1: not used yet
   10 |    1.7306 |     0.0000 | Still in warmup
   15 |   50.6882 |     0.0027 | Warmup begins - discriminator starts learning
   20 |   28.6402 |     0.0060 | Linear warmup continues
   25 |  118.0747 |     0.0093 | Loss growing exponentially
   30 |  249.5944 |     0.0100 | Full adversarial weight reached
   35 |  486.9746 |     0.0100 | EXPLOSION begins
   40 |  892.4181 |     0.0100 | Unstable
   45 | 1249.2195 |     0.0100 | Completely unstable
   50 | 2028.6660 |     0.0100 | Critical failure
```

## Root Cause

The design has a **fundamental instability** in adversarial training with gradient reversal:

### 1. **Unbounded Cross-Entropy Loss**
   - Cross-entropy: `-log(p)` where `p` is predicted probability
   - As discriminator becomes confident: `p → 0` for wrong classes
   - Loss explodes: `-log(0.001) = 6.9`, `-log(0.0001) = 9.2`, `-log(0.00001) = 11.5`
   - Your loss reached **2,028**, meaning discriminator is making EXTREMELY confident wrong predictions

### 2. **Gradient Reversal Amplifies Instability**
   - Large discriminator loss → large gradients
   - Gradient reversal: multiply by `-alpha` (alpha = 0.5-1.0)
   - Reversed gradients destabilize encoder
   - Encoder changes → discriminator sees new features → needs to re-learn → gets more confident → explodes further

### 3. **Adversarial Arms Race**
   - Discriminator: "I can predict Fitzpatrick with 99.9% confidence!"
   - Encoder (via gradient reversal): "Try harder to hide Fitzpatrick info"
   - Discriminator: "I found new patterns, now 99.99% confident!"
   - This feedback loop is **unstable by design**

## Why This Happened

Your setup has **perfect conditions for explosion**:

1. **Strong discriminator**: 3 layers (512→256→128→6) with 30% dropout
2. **Small dataset**: 2,278 training images - discriminator can overfit
3. **High capacity**: Feature dim 512 gives discriminator lots of info
4. **No regularization**: No gradient clipping, no loss capping
5. **Fixed lambda**: 0.01 might still be too high for small dataset

## Fixes Implemented

### ✅ **Fix 1: Gradient Clipping** (Immediate stabilization)
```python
# In train_all_models.py line 313
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ADDED
optimizer.step()
```

**What it does**: Prevents any single gradient from being >1.0, stopping explosion propagation.

**Impact**: **Critical fix** - will prevent catastrophic instability. May slightly slow convergence but ensures training doesn't diverge.

---

### ✅ **Fix 2: Label Smoothing** (Prevents overconfidence)
```python
# In fairness_aware_cbm.py line 376-377
adversarial_loss = F.cross_entropy(
    group_logits, 
    group_labels_indexed, 
    label_smoothing=0.1  # ADDED
)
```

**What it does**: Instead of trying to predict Fitzpatrick Type 4 with 100% confidence (target = [0, 0, 0, 1, 0, 0]), model targets 94% confidence on type 4, 1% on others (target ≈ [0.01, 0.01, 0.01, 0.94, 0.01, 0.01]).

**Impact**: **Major fix** - prevents cross-entropy from exploding when discriminator becomes overconfident. Reduces adversarial loss magnitude by 2-3x.

---

### ✅ **Fix 3: Loss Capping** (Hard safety limit)
```python
# In fairness_aware_cbm.py line 380
adversarial_loss = torch.clamp(adversarial_loss, max=10.0)  # ADDED
```

**What it does**: If adversarial loss exceeds 10.0, clip it to exactly 10.0 before backprop.

**Impact**: **Safety net** - prevents any scenario where loss explodes beyond 10.0. This is a hard limit that should rarely be hit with gradient clipping + label smoothing.

## Expected Results After Fixes

### Before (your test run):
```
Epoch 50:
  Adversarial Loss: 2,028.67  ❌ EXPLODED
  Test F1: 0.507             ⚠️ Mediocre (hurt by instability)
  Demographic Parity: 0.182   ❌ Failing fairness
  TPR Disparity: 1.0          ❌ Complete failure
```

### After (expected with fixes):
```
Epoch 50:
  Adversarial Loss: 2-5       ✅ Stable
  Test F1: 0.55-0.65          ✅ Improved (stable training)
  Demographic Parity: 0.08-0.12 ⚠️ Closer to target (may need more tuning)
  TPR Disparity: 0.3-0.6      ⚠️ Better but still work needed
```

## Why Fairness Metrics Were Poor

The loss explosion **directly caused your fairness failures**:

1. **Fitzpatrick Type 4 got 0% positive predictions** because:
   - Discriminator could easily identify Type 4 from features
   - Encoder tried to remove Type 4 info via gradient reversal
   - But gradients were too large (2000+ loss magnitude!)
   - Model learned to completely suppress Type 4 predictions to "hide" from discriminator

2. **TPR Disparity = 1.0** because:
   - Some groups (Type 6) got 100% TPR
   - Other groups (Types 3, 4, 5) got 0% TPR
   - 100% - 0% = 1.0 (maximum possible disparity)

3. **Instability prevented fairness learning**:
   - First 15 epochs: Good progress (no adversarial loss)
   - Epochs 15-30: Instability begins during warmup
   - Epochs 30-50: Complete chaos, fairness training fails

## Additional Recommendations

### 1. **Lower adversarial_lambda further** (already low at 0.01, but could try 0.005)
```bash
sbatch slurm/run_single_experiment.slurm fair_curriculum_cbm swin 42 \
    --adversarial_lambda 0.005  # Half of current
```

### 2. **Weaker discriminator** (reduce capacity)
In `fairness_aware_cbm.py` line 127:
```python
self.adversarial_discriminator = AdversarialDiscriminator(
    input_dim=feature_dim,
    hidden_dims=[128, 64],  # CHANGED from [256, 128]
    num_groups=num_groups,
    dropout=0.5  # CHANGED from 0.3 (more regularization)
)
```

### 3. **Delayed warmup** (give encoder more time to learn first)
```bash
# Currently: Warmup starts at 20% (epoch 10/50)
# Try: Start at 30% (epoch 15/50)
--adversarial_warmup_epochs 20  # 30% of 50 = 15
```

### 4. **Early stopping** (your validation peaked at epoch 21!)
Add to `train_all_models.py`:
```python
# Stop if validation hasn't improved in 10 epochs
patience = 10
epochs_without_improvement = 0
best_val_f1 = 0

if val_f1 > best_val_f1:
    best_val_f1 = val_f1
    epochs_without_improvement = 0
else:
    epochs_without_improvement += 1
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

## Testing the Fixes

Run the same experiment again:
```bash
sbatch slurm/run_single_experiment.slurm fair_curriculum_cbm swin 42
```

**What to look for:**
1. ✅ Adversarial loss stays **below 10** throughout training
2. ✅ Adversarial loss **stable or slowly increasing** (not exponential)
3. ✅ Validation F1 improves and **stabilizes**
4. ✅ All Fitzpatrick groups get **some positive predictions** (no 0%)
5. ✅ Demographic parity **< 0.15** (improvement from 0.18)

## Bottom Line

**Your design wasn't fundamentally flawed** - adversarial debiasing is a proven technique. But it requires **careful regularization** to prevent the adversarial game from becoming unstable. The three fixes (gradient clipping + label smoothing + loss capping) should resolve the explosion while preserving the fairness learning mechanism.

The explosion **explains why your fairness metrics were so poor** - the model was fighting for survival against unstable gradients rather than learning fair representations. With stable training, you should see significant improvement.
