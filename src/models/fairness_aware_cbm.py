#!/usr/bin/env python3
"""
Fairness-Aware Curriculum CBM

Extends MinimalCurriculumCBM with fairness constraints and adversarial debiasing
to reduce performance disparities across Fitzpatrick skin types.

Author: Matt Cockayne
Date: December 2025
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add FairCBM directory to path for imports
faircbm_dir = Path(__file__).resolve().parent.parent.parent
if str(faircbm_dir) not in sys.path:
    sys.path.insert(0, str(faircbm_dir))

# Import from FairCBM src (self-contained)
from src.models.minimal_curriculum_cbm import MinimalCurriculumCBM
from src.models.adversarial_discriminator import (
    AdversarialDiscriminator,
    AdversarialAlphaScheduler
)
from src.utils.adversarial_debiasing import compute_combined_fairness_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import logging


class FairnessAwareCBM(MinimalCurriculumCBM):
    """
    Curriculum Concept Bottleneck Model with Fairness Constraints.
    
    Extends MinimalCurriculumCBM with:
    - Adversarial discriminator for Fitzpatrick skin type prediction
    - Fairness-aware loss computation (demographic parity + equalized odds)
    - Gradient reversal training for group-invariant representations
    
    The model learns concepts in curriculum phases while simultaneously:
    1. Maintaining diagnostic accuracy (binary classification)
    2. Minimizing performance disparities across Fitzpatrick types
    3. Learning group-invariant concept representations
    
    Reference:
        Ganin & Lempitsky (2015). "Unsupervised Domain Adaptation by Backpropagation"
        Hardt et al. (2016). "Equality of Opportunity in Supervised Learning"
    """
    
    def __init__(self,
                 backbone: str,
                 concept_names: List[str],
                 num_classes: int = 1,
                 num_groups: int = 6,
                 fairness_lambda: float = 1.0,
                 adversarial_lambda: float = 0.1,
                 use_concept_curriculum: bool = False,
                 fairness_type: str = 'equalized_odds',
                 fairness_weights: Optional[Dict[str, float]] = None,
                 dropout_rate: float = 0.1,
                 device: torch.device = None,
                 local_model_path: str = None,
                 classifier_type: str = 'linear',
                 classifier_kwargs: dict = None):
        """
        Initialize Fairness-Aware Curriculum CBM.
        
        Args:
            backbone: Name of the backbone model (e.g., 'swin_base_patch4_window7_224')
            concept_names: List of concept names (23 for SkinCap)
            num_classes: Number of output classes (default: 1 for binary)
            num_groups: Number of demographic groups (default: 6 for Fitzpatrick I-VI)
            fairness_lambda: Weight for fairness loss (default: 1.0)
            adversarial_lambda: Weight for adversarial loss (default: 0.1)
            use_concept_curriculum: If True, use 3-phase concept curriculum. If False, train all concepts jointly (default: False)
            fairness_type: Type of fairness constraint:
                - 'demographic_parity': Equal positive rates
                - 'equalized_odds': Equal TPR and FPR
                - 'equal_opportunity': Equal TPR only
                - 'combined': Weighted combination
            fairness_weights: For 'combined' type: {'dp': 0.5, 'eo': 0.5, 'cal': 0.0}
            dropout_rate: Dropout rate for classifiers
            device: Torch device
            local_model_path: Path to local pretrained weights
            classifier_type: Type of binary classifier ('linear', 'decision_tree', 'attention')
            classifier_kwargs: Additional kwargs for classifier
        """
        # Initialize parent curriculum CBM
        # Note: Simplified MinimalCurriculumCBM only takes num_concepts
        super().__init__(
            num_concepts=len(concept_names) if isinstance(concept_names, list) else concept_names,
            backbone=backbone,
            dropout_rate=dropout_rate,
            device=device
        )
        
        # Store concept names for reference
        self.concept_names = concept_names if isinstance(concept_names, list) else list(range(concept_names))
        
        # Fairness hyperparameters
        self.num_groups = num_groups
        self.fairness_lambda = fairness_lambda
        self.adversarial_lambda = adversarial_lambda
        self.use_concept_curriculum = use_concept_curriculum
        self.fairness_type = fairness_type
        
        # Store concept names and create concept-to-index mapping
        self.concept_names = concept_names
        self.concept_to_idx = {name: idx for idx, name in enumerate(concept_names)}
        
        # Default fairness weights for combined loss
        if fairness_weights is None:
            self.fairness_weights = {'dp': 0.5, 'eo': 0.5, 'cal': 0.0}
        else:
            self.fairness_weights = fairness_weights
        
        # Get feature dimension from backbone
        # Most backbones output features that get pooled to this dimension
        feature_dim = self._get_feature_dim()
        
        # Add adversarial discriminator for Fitzpatrick type prediction
        self.adversarial_discriminator = AdversarialDiscriminator(
            input_dim=feature_dim,
            hidden_dims=[256, 128],
            num_groups=num_groups,
            dropout=0.3
        ).to(self.device)
        
        # Initialize alpha scheduler for gradient reversal
        # Alpha increases from 0 to 1 during training (Ganin schedule)
        self.adversarial_alpha = 0.0
        
        logging.info(f"Initialized FairnessAwareCBM with fairness_lambda={fairness_lambda}, "
                    f"adversarial_lambda={adversarial_lambda}, fairness_type={fairness_type}")
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension from backbone output."""
        # Return the actual backbone output dimension
        # This is stored in the parent class after initialization
        return self.num_features
    
    def get_active_concept_indices(self, active_concepts: List[str]) -> List[int]:
        """
        Convert concept names to indices for masking.
        
        Args:
            active_concepts: List of concept names
            
        Returns:
            List of concept indices
        """
        if isinstance(active_concepts, str) and active_concepts == "all":
            return list(range(self.num_concepts))
        
        indices = []
        for concept_name in active_concepts:
            if concept_name in self.concept_to_idx:
                indices.append(self.concept_to_idx[concept_name])
            else:
                logging.warning(f"Unknown concept: {concept_name}")
        
        return indices
    
    def get_curriculum_phase_info(self, epoch: int, max_epochs: int):
        """
        Get current curriculum phase information based on epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
            max_epochs: Total training epochs
            
        Returns:
            dict with 'phase_idx', 'phase_name', 'active_concepts', 'new_concepts'
        """
        # Divide training into 3 equal phases
        epochs_per_phase = max_epochs // 3
        
        # Determine current phase
        if epoch < epochs_per_phase:
            phase_idx = 0
        elif epoch < 2 * epochs_per_phase:
            phase_idx = 1
        else:
            phase_idx = 2
        
        # Define medical curriculum (same as parent class)
        primary_lesions = ["Nodule", "Ulcer", "Papule", "Plaque", "Pustule", "Bulla", "Patch"]
        secondary_features = ["Scale", "Crust", "Erosion", "Brown(Hyperpigmentation)", "Scar"]
        
        if phase_idx == 0:
            # Phase 1: Primary lesions only
            active_concepts = [c for c in primary_lesions if c in self.concept_names]
            new_concepts = set(active_concepts)
            phase_name = "primary_lesions"
        elif phase_idx == 1:
            # Phase 2: Primary + Secondary
            all_concepts = primary_lesions + secondary_features
            active_concepts = [c for c in all_concepts if c in self.concept_names]
            new_concepts = set([c for c in secondary_features if c in self.concept_names])
            phase_name = "secondary_features"
        else:
            # Phase 3: All concepts
            active_concepts = self.concept_names
            prev_concepts = set([c for c in primary_lesions + secondary_features if c in self.concept_names])
            new_concepts = set([c for c in self.concept_names if c not in prev_concepts])
            phase_name = "all_concepts"
        
        return {
            'phase_idx': phase_idx,
            'phase_name': phase_name,
            'active_concepts': active_concepts,
            'new_concepts': new_concepts
        }
    
    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Forward pass with optional intermediate feature extraction.
        
        Args:
            x: Input images [batch_size, 3, 224, 224]
            return_features: If True, also return concept-layer features
        
        Returns:
            If return_features=False:
                concept_logits: [batch_size, num_concepts]
                binary_logits: [batch_size, num_classes]
            If return_features=True:
                concept_logits: [batch_size, num_concepts]
                binary_logits: [batch_size, num_classes]
                concept_features: [batch_size, feature_dim]
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Handle different feature shapes (copied from parent)
        if len(features.shape) == 4:
            # 4D features [batch, channels, H, W] or [batch, H, W, channels]
            if features.shape[-1] > features.shape[1]:
                # Likely [batch, H, W, channels] (Swin format)
                features = features.permute(0, 3, 1, 2)
            # Pool: [batch, channels, H, W] -> [batch, channels]
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        elif len(features.shape) == 3:
            # 3D features [batch, seq_len, features] - use CLS token
            features = features[:, 0, :]
        
        # Concept prediction
        # For adversarial training, we need features before final activation
        concept_features = features  # Use backbone output as discriminator input
        concept_logits = self.concept_layer(features)
        
        # Binary prediction from concepts
        concept_probs = torch.sigmoid(concept_logits)
        binary_logits = self.binary_classifier(concept_probs)
        
        if return_features:
            return concept_logits, binary_logits, concept_features
        else:
            return concept_logits, binary_logits
    
    def compute_fairness_loss(self,
                             concept_logits: torch.Tensor,
                             binary_logits: torch.Tensor,
                             concept_features: torch.Tensor,
                             concept_labels: torch.Tensor,
                             binary_labels: torch.Tensor,
                             group_labels: torch.Tensor,
                             active_concepts: List[str],
                             new_concepts: Optional[set] = None,
                             new_concept_weight_multiplier: float = 1.0,
                             concept_weight: float = 0.3,
                             binary_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute loss with fairness constraints.
        
        Total loss combines:
        1. Concept loss (from parent curriculum CBM)
        2. Binary classification loss (from parent)
        3. Fairness loss (demographic parity + equalized odds)
        4. Adversarial loss (gradient reversal for group invariance)
        
        Args:
            concept_logits: Predicted concept logits [batch, num_concepts]
            binary_logits: Predicted binary logits [batch, num_classes]
            concept_features: Features for adversarial discriminator [batch, feature_dim]
            concept_labels: True concept labels [batch, num_concepts]
            binary_labels: True binary labels [batch]
            group_labels: Fitzpatrick skin type labels [batch] (1-6, will convert to 0-5)
            active_concepts: List of active concept names in current curriculum phase
            new_concepts: Set of newly introduced concepts in this phase
            new_concept_weight_multiplier: Weight multiplier for new concepts (warmup)
            concept_weight: Global weight for concept loss
            binary_weight: Global weight for binary loss
        
        Returns:
            Dictionary with losses:
                - concept_loss: BCE on concept predictions
                - binary_loss: BCE on binary predictions
                - fairness_loss: Demographic parity + equalized odds
                - adversarial_loss: Group prediction loss
                - total_loss: Weighted combination
        """
        # Compute base losses with optional curriculum masking
        # Get active concept indices
        if self.use_concept_curriculum and active_concepts and len(active_concepts) > 0 and active_concepts != []:
            active_indices = self.get_active_concept_indices(active_concepts)
        else:
            # No curriculum (or curriculum disabled) - train all concepts
            active_indices = list(range(self.num_concepts))
        
        # Concept loss (only for active concepts)
        if len(active_indices) > 0:
            # Apply different weights for new vs existing concepts during warmup (only if curriculum enabled)
            if self.use_concept_curriculum and new_concepts and new_concept_weight_multiplier < 1.0:
                # Get indices of new concepts
                new_concept_indices = self.get_active_concept_indices(list(new_concepts))
                existing_concept_indices = [idx for idx in active_indices if idx not in new_concept_indices]
                
                # Compute weighted loss
                concept_loss = torch.tensor(0.0, device=concept_logits.device)
                
                # Loss for existing concepts (full weight)
                if existing_concept_indices:
                    existing_loss = F.binary_cross_entropy_with_logits(
                        concept_logits[:, existing_concept_indices],
                        concept_labels[:, existing_concept_indices]
                    )
                    concept_loss = concept_loss + existing_loss * (len(existing_concept_indices) / len(active_indices))
                
                # Loss for new concepts (reduced weight during warmup)
                if new_concept_indices:
                    new_loss = F.binary_cross_entropy_with_logits(
                        concept_logits[:, new_concept_indices],
                        concept_labels[:, new_concept_indices]
                    )
                    concept_loss = concept_loss + new_loss * new_concept_weight_multiplier * (len(new_concept_indices) / len(active_indices))
            else:
                # Standard loss (no warmup)
                concept_loss = F.binary_cross_entropy_with_logits(
                    concept_logits[:, active_indices],
                    concept_labels[:, active_indices]
                )
        else:
            concept_loss = torch.tensor(0.0, device=concept_logits.device)
        
        binary_loss = F.binary_cross_entropy_with_logits(binary_logits.squeeze(), binary_labels.float())
        base_loss = concept_weight * concept_loss + binary_weight * binary_loss
        
        # Compute fairness loss on binary predictions
        binary_probs = torch.sigmoid(binary_logits).squeeze()
        
        # Convert Fitzpatrick labels to 0-indexed (1-6 → 0-5)
        group_labels_indexed = (group_labels - 1).long().clamp(0, self.num_groups - 1)
        
        fairness_loss = compute_combined_fairness_loss(
            predictions=binary_probs,
            labels=binary_labels,
            groups=group_labels_indexed,
            fairness_type=self.fairness_type,
            demographic_parity_weight=self.fairness_weights.get('dp', 0.5),
            equalized_odds_weight=self.fairness_weights.get('eo', 0.5),
            calibration_weight=self.fairness_weights.get('cal', 0.0)
        )
        
        # Compute adversarial loss (predict Fitzpatrick type from concept features)
        # Gradient reversal encourages group-invariant representations
        group_logits = self.adversarial_discriminator(concept_features, alpha=self.adversarial_alpha)
        
        # Use label smoothing to prevent overconfident predictions and loss explosion
        # Smoothing=0.1 means 90% confidence on true class, 10% distributed to others
        adversarial_loss = F.cross_entropy(group_logits, group_labels_indexed, label_smoothing=0.1)
        
        # Cap adversarial loss to prevent explosion (critical fix!)
        # When discriminator becomes overconfident, cross-entropy can explode
        adversarial_loss = torch.clamp(adversarial_loss, max=10.0)
        
        # Total loss
        # Note: Adversarial loss has negative weight (maximize via gradient reversal)
        total_loss = (
            base_loss +
            self.fairness_lambda * fairness_loss -
            self.adversarial_lambda * adversarial_loss
        )
        
        return {
            'concept_loss': concept_loss,
            'binary_loss': binary_loss,
            'fairness_loss': fairness_loss,
            'adversarial_loss': adversarial_loss,
            'total_loss': total_loss
        }
    
    def update_adversarial_alpha(self, epoch: int, max_epochs: int):
        """
        Update gradient reversal strength using Ganin schedule.
        
        Alpha increases from 0 to 1 during training, allowing the model to first
        learn task-relevant features, then progressively remove group-specific information.
        
        Schedule: α = 2/(1 + exp(-10p)) - 1, where p = epoch/max_epochs
        
        Args:
            epoch: Current epoch (0-indexed)
            max_epochs: Total number of epochs
        """
        progress = epoch / max_epochs
        self.adversarial_alpha = 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0
        self.adversarial_alpha = float(np.clip(self.adversarial_alpha, 0.0, 1.0))
    
    def get_group_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict Fitzpatrick skin type from images (for evaluation).
        
        Args:
            x: Input images [batch_size, 3, 224, 224]
        
        Returns:
            group_probs: Predicted Fitzpatrick type probabilities [batch_size, num_groups]
        """
        with torch.no_grad():
            _, _, concept_features = self.forward(x, return_features=True)
            group_probs = self.adversarial_discriminator.predict_group(concept_features)
        return group_probs


def create_fairness_aware_cbm(backbone: str,
                              concept_names: List[str],
                              num_groups: int = 6,
                              fairness_lambda: float = 1.0,
                              adversarial_lambda: float = 0.1,
                              fairness_type: str = 'equalized_odds',
                              device: torch.device = None,
                              **kwargs) -> FairnessAwareCBM:
    """
    Factory function to create FairnessAwareCBM with default settings.
    
    Args:
        backbone: Backbone model name
        concept_names: List of concept names
        num_groups: Number of demographic groups (default: 6 for Fitzpatrick)
        fairness_lambda: Fairness loss weight
        adversarial_lambda: Adversarial loss weight
        fairness_type: Type of fairness constraint
        device: Torch device
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Initialized FairnessAwareCBM model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FairnessAwareCBM(
        backbone=backbone,
        concept_names=concept_names,
        num_groups=num_groups,
        fairness_lambda=fairness_lambda,
        adversarial_lambda=adversarial_lambda,
        fairness_type=fairness_type,
        device=device,
        **kwargs
    )
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing FairnessAwareCBM...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    concept_names = ['Papule', 'Plaque', 'Nodule', 'Ulcer', 'Scale']
    model = create_fairness_aware_cbm(
        backbone='swin_base_patch4_window7_224',
        concept_names=concept_names,
        num_groups=6,
        fairness_lambda=1.0,
        adversarial_lambda=0.1,
        fairness_type='equalized_odds',
        device=device
    )
    
    print(f"Model created successfully!")
    print(f"Number of concepts: {len(concept_names)}")
    print(f"Number of groups: 6 (Fitzpatrick I-VI)")
    
    # Test forward pass - move data to device
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    concepts = torch.randint(0, 2, (batch_size, len(concept_names))).float().to(device)
    labels = torch.randint(0, 2, (batch_size,)).float().to(device)
    fitzpatrick = torch.randint(0, 6, (batch_size,)).long().to(device)  # 0-5 (converted from 1-6)
    
    print("\nTesting forward pass...")
    concept_logits, binary_logits, features = model(images, return_features=True)
    print(f"Concept logits shape: {concept_logits.shape}")
    print(f"Binary logits shape: {binary_logits.shape}")
    print(f"Features shape: {features.shape}")
    
    # Test loss computation
    print("\nTesting fairness loss computation...")
    losses = model.compute_fairness_loss(
        concept_logits=concept_logits,
        binary_logits=binary_logits,
        concept_features=features,
        concept_labels=concepts,
        binary_labels=labels,
        group_labels=fitzpatrick,
        active_concepts=concept_names
    )
    
    print("Losses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Test adversarial alpha update
    print("\nTesting adversarial alpha schedule:")
    for epoch in [0, 10, 25, 50, 75, 100]:
        model.update_adversarial_alpha(epoch, 100)
        print(f"  Epoch {epoch:3d}: alpha = {model.adversarial_alpha:.4f}")
    
    # Test group prediction
    print("\nTesting group prediction...")
    group_probs = model.get_group_predictions(images)
    print(f"Group probabilities shape: {group_probs.shape}")
    print(f"Group probabilities (first sample): {group_probs[0].cpu().numpy()}")
    
    print("\n✓ All tests passed!")
