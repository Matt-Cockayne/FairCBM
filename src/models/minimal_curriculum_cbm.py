"""
Minimal Curriculum CBM - Self-contained implementation

This is a self-contained version within FairCBM with no external dependencies.
Provides curriculum learning functionality for concept bottleneck models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class CurriculumPhase:
    """Configuration for a single curriculum phase"""
    name: str
    concepts: List[str]
    epochs: int
    description: str


class MinimalCurriculumCBM(nn.Module):
    """
    Concept Bottleneck Model with fixed curriculum based on medical importance.
    Self-contained version for fairness repository.
    """
    
    def __init__(self, num_concepts: int, backbone: str = 'swin', 
                 dropout_rate: float = 0.1, device: torch.device = None):
        """
        Initialize MinimalCurriculumCBM.
        
        Args:
            num_concepts: Number of concepts
            backbone: Backbone architecture name
            dropout_rate: Dropout rate
            device: Torch device
        """
        super().__init__()
        
        self.num_concepts = num_concepts
        self.device = device or torch.device("cpu")
        self.backbone_name = backbone
        
        # Load backbone - simplified version that uses torchvision
        self.backbone, self.num_features = self._load_backbone(backbone)
        
        # Concept prediction layer
        self.concept_layer = nn.Linear(self.num_features, self.num_concepts)
        
        # Binary classification layer
        self.binary_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_concepts, 1),
        )
        
        self.to(self.device)
        
        logging.info(f"Initialized Minimal Curriculum CBM:")
        logging.info(f"  Concepts: {num_concepts}")
        logging.info(f"  Backbone: {backbone}")
        logging.info(f"  Feature dim: {self.num_features}")
    
    def _load_backbone(self, backbone: str):
        """Load backbone model."""
        import torchvision.models as models
        
        # Normalize backbone name (handle full names like 'swin_base_patch4_window7_224')
        if 'swin' in backbone.lower():
            model = models.swin_b(weights='DEFAULT')
            num_features = model.head.in_features
            model.head = nn.Identity()
        elif 'convnext' in backbone.lower():
            model = models.convnext_base(weights='DEFAULT')
            num_features = model.classifier[2].in_features
            model.classifier = nn.Identity()
        elif 'vit' in backbone.lower():
            model = models.vit_b_16(weights='DEFAULT')
            num_features = model.heads.head.in_features
            model.heads = nn.Identity()
        elif 'efficientnet' in backbone.lower():
            model = models.efficientnet_v2_s(weights='DEFAULT')
            num_features = model.classifier[1].in_features
            model.classifier = nn.Identity()
        elif 'mobilenet' in backbone.lower():
            model = models.mobilenet_v2(weights='DEFAULT')
            num_features = model.classifier[1].in_features
            model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        return model, num_features
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            return_features: If True, also return concept features
        
        Returns:
            concept_logits, binary_logits, [concept_features]
        """
        features = self.backbone(x)
        
        # Handle different feature shapes
        if len(features.shape) == 4:
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        elif len(features.shape) == 3:
            features = features[:, 0, :]
        
        concept_logits = self.concept_layer(features)
        concept_probs = torch.sigmoid(concept_logits)
        binary_logits = self.binary_classifier(concept_probs)
        
        if return_features:
            return concept_logits, binary_logits, features
        return concept_logits, binary_logits
    
    def compute_loss(self, concept_logits: torch.Tensor, binary_logits: torch.Tensor,
                     concept_labels: torch.Tensor, binary_labels: torch.Tensor,
                     epoch: int, max_epochs: int) -> Dict[str, torch.Tensor]:
        """
        Compute curriculum-aware loss.
        
        Args:
            concept_logits: Predicted concept logits
            binary_logits: Predicted binary logits
            concept_labels: Ground truth concept labels
            binary_labels: Ground truth binary labels
            epoch: Current epoch
            max_epochs: Total epochs
        
        Returns:
            Dictionary with loss components
        """
        # Concept loss
        concept_loss = F.binary_cross_entropy_with_logits(concept_logits, concept_labels)
        
        # Binary loss
        binary_loss = F.binary_cross_entropy_with_logits(
            binary_logits.squeeze(), binary_labels.float()
        )
        
        # Combined loss with curriculum weighting
        # Early epochs: focus more on concepts
        # Later epochs: balance concepts and binary
        progress = epoch / max_epochs
        concept_weight = 0.5 - 0.2 * progress  # 0.5 → 0.3
        binary_weight = 0.5 + 0.5 * progress   # 0.5 → 1.0
        
        total_loss = concept_weight * concept_loss + binary_weight * binary_loss
        
        return {
            'total_loss': total_loss,
            'concept_loss': concept_loss,
            'binary_loss': binary_loss
        }
