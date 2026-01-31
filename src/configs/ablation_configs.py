"""
Ablation Study Configurations

Defines configurations for ablating individual phases of the Fair Curriculum CBM.
Each configuration disables specific phase components to measure their contribution.

Ablation Strategy:
- Full Model: All 4 phases active (baseline)
- w/o Phase 1: Skip balanced foundation (start with random sampling)
- w/o Phase 2: Skip demographic parity focus (jump from balanced to EO)
- w/o Phase 3: Skip equalized odds + adversarial (no EO loss or adversarial debiasing)
- w/o Phase 4: Skip performance parity (no error-driven sampling)

Author: Matt Cockayne
Date: January 2026
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    
    name: str  # Human-readable name
    key: str   # Short key for filenames (e.g., 'no_phase1')
    
    # Phase controls
    use_phase1_balanced: bool = True  # Balanced sampling in Phase 1
    use_phase2_dp: bool = True        # Demographic parity in Phase 2
    use_phase3_eo: bool = True        # Equalized odds + adversarial in Phase 3
    use_phase4_error: bool = True     # Error-driven sampling in Phase 4
    
    # Additional controls
    use_adversarial: bool = True      # Gradient reversal discriminator
    adversarial_start_phase: int = 3  # When to activate adversarial (phase 3-4)
    
    # Fallback behaviors when phases are disabled
    fallback_sampling: str = 'random'  # 'random' or 'balanced'
    fallback_fairness: str = 'combined'  # 'combined', 'dp', 'eo', or 'none'
    
    def __repr__(self):
        return f"AblationConfig({self.name})"
    
    def get_phase_config(self, epoch: int, total_epochs: int) -> Dict:
        """
        Get configuration for the current epoch based on ablation settings.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total training epochs
            
        Returns:
            Dict with phase-specific settings
        """
        progress = epoch / total_epochs
        
        # Determine base phase
        if progress <= 0.25:
            phase_num = 1
            phase_name = 'balanced_foundation'
        elif progress <= 0.50:
            phase_num = 2
            phase_name = 'demographic_parity'
        elif progress <= 0.75:
            phase_num = 3
            phase_name = 'equalized_odds'
        else:
            phase_num = 4
            phase_name = 'performance_parity'
        
        # Apply ablation modifications
        config = {
            'phase_num': phase_num,
            'phase_name': phase_name,
            'epoch': epoch,
            'progress': progress
        }
        
        # Phase 1: Balanced Foundation
        if phase_num == 1:
            config['use_balanced_sampling'] = self.use_phase1_balanced
            config['use_fairness_loss'] = False  # No fairness loss in Phase 1
            config['fairness_weights'] = {'dp': 0.0, 'eo': 0.0, 'pg': 0.0}
            config['use_adversarial'] = False
            
            if not self.use_phase1_balanced:
                config['sampling_strategy'] = self.fallback_sampling
            else:
                config['sampling_strategy'] = 'balanced'
        
        # Phase 2: Demographic Parity Focus
        elif phase_num == 2:
            if self.use_phase2_dp:
                config['use_balanced_sampling'] = True
                config['use_fairness_loss'] = True
                config['fairness_weights'] = {'dp': 1.0, 'eo': 0.0, 'pg': 0.0}
                config['sampling_strategy'] = 'balanced'
            else:
                # Skip Phase 2: Use Phase 1 settings or jump to Phase 3
                if self.use_phase1_balanced:
                    config['use_balanced_sampling'] = True
                    config['sampling_strategy'] = 'balanced'
                else:
                    config['sampling_strategy'] = self.fallback_sampling
                config['use_fairness_loss'] = False
                config['fairness_weights'] = {'dp': 0.0, 'eo': 0.0, 'pg': 0.0}
            
            config['use_adversarial'] = False
        
        # Phase 3: Equalized Odds + Adversarial
        elif phase_num == 3:
            if self.use_phase3_eo:
                config['use_balanced_sampling'] = False
                config['use_fairness_loss'] = True
                config['fairness_weights'] = {'dp': 0.3, 'eo': 0.7, 'pg': 0.0}
                config['sampling_strategy'] = 'stratified'
                config['use_adversarial'] = self.use_adversarial and (phase_num >= self.adversarial_start_phase)
            else:
                # Skip Phase 3: Use Phase 2 settings or continue without EO
                if self.use_phase2_dp:
                    config['use_fairness_loss'] = True
                    config['fairness_weights'] = {'dp': 1.0, 'eo': 0.0, 'pg': 0.0}
                    config['sampling_strategy'] = 'balanced'
                else:
                    config['use_fairness_loss'] = False
                    config['fairness_weights'] = {'dp': 0.0, 'eo': 0.0, 'pg': 0.0}
                    config['sampling_strategy'] = self.fallback_sampling
                config['use_adversarial'] = False
        
        # Phase 4: Performance Parity
        elif phase_num == 4:
            if self.use_phase4_error:
                config['use_balanced_sampling'] = False
                config['use_fairness_loss'] = True
                config['fairness_weights'] = {'dp': 0.33, 'eo': 0.33, 'pg': 0.34}
                config['sampling_strategy'] = 'error_driven'
                config['use_adversarial'] = self.use_adversarial and (phase_num >= self.adversarial_start_phase)
            else:
                # Skip Phase 4: Continue with Phase 3 settings
                if self.use_phase3_eo:
                    config['use_fairness_loss'] = True
                    config['fairness_weights'] = {'dp': 0.3, 'eo': 0.7, 'pg': 0.0}
                    config['sampling_strategy'] = 'stratified'
                    config['use_adversarial'] = self.use_adversarial and (phase_num >= self.adversarial_start_phase)
                elif self.use_phase2_dp:
                    config['use_fairness_loss'] = True
                    config['fairness_weights'] = {'dp': 1.0, 'eo': 0.0, 'pg': 0.0}
                    config['sampling_strategy'] = 'balanced'
                    config['use_adversarial'] = False
                else:
                    config['use_fairness_loss'] = False
                    config['fairness_weights'] = {'dp': 0.0, 'eo': 0.0, 'pg': 0.0}
                    config['sampling_strategy'] = self.fallback_sampling
                    config['use_adversarial'] = False
        
        return config


# Define all ablation configurations
ABLATION_CONFIGS = {
    'full_model': AblationConfig(
        name='Full Model (All Phases)',
        key='full_model',
        use_phase1_balanced=True,
        use_phase2_dp=True,
        use_phase3_eo=True,
        use_phase4_error=True,
        use_adversarial=True,
        adversarial_start_phase=3
    ),
    
    'no_phase1': AblationConfig(
        name='w/o Phase 1 (No Balanced Init)',
        key='no_phase1',
        use_phase1_balanced=False,  # Skip balanced foundation
        use_phase2_dp=True,
        use_phase3_eo=True,
        use_phase4_error=True,
        use_adversarial=True,
        adversarial_start_phase=3,
        fallback_sampling='random'  # Random sampling instead of balanced
    ),
    
    'no_phase2': AblationConfig(
        name='w/o Phase 2 (No DP Focus)',
        key='no_phase2',
        use_phase1_balanced=True,
        use_phase2_dp=False,  # Skip demographic parity phase
        use_phase3_eo=True,
        use_phase4_error=True,
        use_adversarial=True,
        adversarial_start_phase=3
    ),
    
    'no_phase3': AblationConfig(
        name='w/o Phase 3 (No EO + Adversarial)',
        key='no_phase3',
        use_phase1_balanced=True,
        use_phase2_dp=True,
        use_phase3_eo=False,  # Skip equalized odds phase
        use_phase4_error=True,
        use_adversarial=False,  # No adversarial debiasing
        adversarial_start_phase=5  # Never activates
    ),
    
    'no_phase4': AblationConfig(
        name='w/o Phase 4 (No Error-Driven)',
        key='no_phase4',
        use_phase1_balanced=True,
        use_phase2_dp=True,
        use_phase3_eo=True,
        use_phase4_error=False,  # Skip performance parity phase
        use_adversarial=True,
        adversarial_start_phase=3
    )
}


def get_ablation_config(ablation_key: str) -> AblationConfig:
    """
    Get ablation configuration by key.
    
    Args:
        ablation_key: One of ['full_model', 'no_phase1', 'no_phase2', 'no_phase3', 'no_phase4']
        
    Returns:
        AblationConfig instance
        
    Raises:
        ValueError: If ablation_key is not recognized
    """
    if ablation_key not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation key: {ablation_key}. "
                        f"Valid options: {list(ABLATION_CONFIGS.keys())}")
    
    return ABLATION_CONFIGS[ablation_key]


def list_ablation_configs() -> List[str]:
    """Return list of available ablation configuration keys."""
    return list(ABLATION_CONFIGS.keys())


def print_ablation_summary():
    """Print summary of all ablation configurations."""
    print("\n" + "="*60)
    print("ABLATION STUDY CONFIGURATIONS")
    print("="*60)
    
    for key, config in ABLATION_CONFIGS.items():
        print(f"\n{key}:")
        print(f"  Name: {config.name}")
        print(f"  Phase 1 (Balanced): {config.use_phase1_balanced}")
        print(f"  Phase 2 (DP): {config.use_phase2_dp}")
        print(f"  Phase 3 (EO): {config.use_phase3_eo}")
        print(f"  Phase 4 (Error): {config.use_phase4_error}")
        print(f"  Adversarial: {config.use_adversarial}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    # Test ablation configs
    print_ablation_summary()
    
    # Test phase configuration generation
    print("\n\nTesting Phase Configuration Generation:")
    print("="*60)
    
    for ablation_key in ['full_model', 'no_phase1', 'no_phase3']:
        config = get_ablation_config(ablation_key)
        print(f"\n{config.name}:")
        
        # Test at different epoch milestones
        for epoch in [0, 25, 50, 75]:
            phase_config = config.get_phase_config(epoch, total_epochs=100)
            print(f"  Epoch {epoch} (Phase {phase_config['phase_num']}): "
                  f"Fairness={phase_config['fairness_weights']}, "
                  f"Sampling={phase_config['sampling_strategy']}, "
                  f"Adversarial={phase_config['use_adversarial']}")
