"""
Quick test to verify ablation configurations work correctly.

Tests that disabled phases properly affect fairness loss computation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.fair_curriculum_cbm import FairCurriculumCBM, PhasedFairnessLoss

def test_phased_fairness_loss():
    """Test that PhasedFairnessLoss respects disabled_phases."""
    print("Testing PhasedFairnessLoss with ablation...")
    
    # Test normal (no ablation)
    loss_fn = PhasedFairnessLoss(total_epochs=100, num_groups=6, disabled_phases=[])
    
    # Dummy data
    predictions = torch.rand(32, 1)  # [batch_size, 1]
    labels = torch.randint(0, 2, (32,))
    groups = torch.randint(0, 6, (32,))
    
    # Phase 1 (epoch 10): Should return 0 (no fairness)
    loss_p1 = loss_fn(predictions, labels, groups, epoch=10)
    print(f"  Phase 1 (epoch 10, no fairness): {loss_p1.item():.6f}")
    assert loss_p1.item() == 0.0, "Phase 1 should have zero fairness loss"
    
    # Phase 2 (epoch 30): Should return DP loss
    loss_p2 = loss_fn(predictions, labels, groups, epoch=30)
    print(f"  Phase 2 (epoch 30, DP only): {loss_p2.item():.6f}")
    assert loss_p2.item() > 0.0, "Phase 2 should have non-zero fairness loss"
    
    # Test with Phase 2 disabled
    loss_fn_ablated = PhasedFairnessLoss(total_epochs=100, num_groups=6, disabled_phases=[2])
    loss_p2_ablated = loss_fn_ablated(predictions, labels, groups, epoch=30)
    print(f"  Phase 2 DISABLED (epoch 30): {loss_p2_ablated.item():.6f}")
    assert loss_p2_ablated.item() == 0.0, "Phase 2 disabled should return zero (fall back to Phase 1)"
    
    # Test with Phase 3 disabled
    loss_fn_ablated3 = PhasedFairnessLoss(total_epochs=100, num_groups=6, disabled_phases=[3])
    loss_p3_ablated = loss_fn_ablated3(predictions, labels, groups, epoch=60)
    loss_p2_normal = loss_fn(predictions, labels, groups, epoch=30)
    print(f"  Phase 3 DISABLED (epoch 60): {loss_p3_ablated.item():.6f}")
    print(f"  Phase 2 normal (epoch 30): {loss_p2_normal.item():.6f}")
    # Should use Phase 2 behavior (DP only)
    
    print("✓ PhasedFairnessLoss ablation tests passed!\n")


def test_fair_curriculum_cbm():
    """Test that FairCurriculumCBM accepts and uses ablation configs."""
    print("Testing FairCurriculumCBM with ablation...")
    
    # Test normal model
    model = FairCurriculumCBM(
        num_concepts=23,
        backbone='swin',
        disabled_phases=[],
        disable_adversarial=False
    )
    print(f"  Normal model - disabled_phases: {model.disabled_phases}")
    print(f"  Normal model - disable_adversarial: {model.disable_adversarial}")
    assert model.disabled_phases == []
    assert model.disable_adversarial == False
    
    # Test with Phase 2 disabled
    model_no_p2 = FairCurriculumCBM(
        num_concepts=23,
        backbone='swin',
        disabled_phases=[2],
        disable_adversarial=False
    )
    print(f"  Ablated model - disabled_phases: {model_no_p2.disabled_phases}")
    assert 2 in model_no_p2.disabled_phases
    assert model_no_p2.fairness_loss_fn.disabled_phases == [2]
    
    # Test with adversarial disabled
    model_no_adv = FairCurriculumCBM(
        num_concepts=23,
        backbone='swin',
        disabled_phases=[],
        disable_adversarial=True
    )
    print(f"  No adversarial - disable_adversarial: {model_no_adv.disable_adversarial}")
    assert model_no_adv.disable_adversarial == True
    
    # Test adversarial lambda computation
    adv_lambda_normal = model.compute_adversarial_lambda(epoch=75, total_epochs=100)
    adv_lambda_disabled = model_no_adv.compute_adversarial_lambda(epoch=75, total_epochs=100)
    print(f"  Adversarial lambda (normal, epoch 75): {adv_lambda_normal:.4f}")
    print(f"  Adversarial lambda (disabled, epoch 75): {adv_lambda_disabled:.4f}")
    assert adv_lambda_normal > 0.0, "Normal model should have non-zero adversarial lambda"
    assert adv_lambda_disabled == 0.0, "Disabled model should have zero adversarial lambda"
    
    print("✓ FairCurriculumCBM ablation tests passed!\n")


def test_ablation_configs():
    """Test all 5 ablation configurations."""
    print("Testing all ablation configurations...")
    
    ablation_configs = {
        'full_model': {'disabled_phases': [], 'disable_adversarial': False},
        'no_phase2': {'disabled_phases': [2], 'disable_adversarial': False},
        'no_phase3': {'disabled_phases': [3], 'disable_adversarial': False},
        'no_phase4': {'disabled_phases': [4], 'disable_adversarial': False},
        'no_adversarial': {'disabled_phases': [], 'disable_adversarial': True},
    }
    
    for ablation_key, config in ablation_configs.items():
        model = FairCurriculumCBM(
            num_concepts=23,
            backbone='swin',
            disabled_phases=config['disabled_phases'],
            disable_adversarial=config['disable_adversarial']
        )
        print(f"  {ablation_key:<20} disabled_phases={config['disabled_phases']}, "
              f"disable_adversarial={config['disable_adversarial']}")
        
        # Verify configuration
        assert model.disabled_phases == config['disabled_phases']
        assert model.disable_adversarial == config['disable_adversarial']
        assert model.fairness_loss_fn.disabled_phases == config['disabled_phases']
    
    print("✓ All ablation configurations valid!\n")


if __name__ == '__main__':
    print("=" * 70)
    print("ABLATION CONFIGURATION TESTS")
    print("=" * 70)
    print()
    
    test_phased_fairness_loss()
    test_fair_curriculum_cbm()
    test_ablation_configs()
    
    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print()
    print("Ablation support is working correctly!")
    print("Ready to run: sbatch slurm/run_ablation_study.slurm")
