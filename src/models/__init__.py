"""
FairCBM Models
"""

from .direct_classifier import DirectClassifier
from .standard_cbm import StandardCBM
from .minimal_curriculum_cbm import MinimalCurriculumCBM
from .fairness_aware_cbm import FairnessAwareCBM
from .fair_curriculum_cbm import FairCurriculumCBM, FairnessAwareSampler, PhasedFairnessLoss
from .adversarial_discriminator import AdversarialDiscriminator, AdversarialAlphaScheduler

__all__ = [
    'DirectClassifier',
    'StandardCBM',
    'MinimalCurriculumCBM',
    'FairnessAwareCBM',
    'FairCurriculumCBM',
    'FairnessAwareSampler',
    'PhasedFairnessLoss',
    'AdversarialDiscriminator',
    'AdversarialAlphaScheduler',
]
