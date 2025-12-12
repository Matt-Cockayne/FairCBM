"""
Standard Concept Bottleneck Model - Simplified for FairCBM

Standard CBM with joint concept and binary classification training.
Serves as baseline for comparison with curriculum and fairness-aware variants.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Tuple


class StandardCBM(nn.Module):
    """Standard Concept Bottleneck Model with joint training"""
    
    def __init__(self, backbone: str, num_concepts: int, 
                 concept_names: List[str] = None, dropout_rate: float = 0.1):
        super().__init__()
        
        self.num_concepts = num_concepts
        self.concept_names = concept_names or [f'concept_{i}' for i in range(num_concepts)]
        self.dropout_rate = dropout_rate
        
        # Load pretrained backbone
        self.backbone, self.feature_dim = self._load_backbone(backbone)
        
        # Concept predictor
        self.concept_predictor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, num_concepts),
        )
        
        # Binary classifier (uses concepts as input)
        self.binary_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_concepts, 1),
        )
        
    def _load_backbone(self, backbone_name: str) -> Tuple[nn.Module, int]:
        """Load pretrained backbone and return model + feature dimension."""
        if backbone_name == 'swin':
            model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
            feature_dim = model.head.in_features
            model.head = nn.Identity()
        elif backbone_name == 'convnext':
            model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            feature_dim = model.classifier[2].in_features
            model.classifier = nn.Identity()
        elif backbone_name == 'vit':
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            feature_dim = model.heads.head.in_features
            model.heads = nn.Identity()
        elif backbone_name == 'efficientnet':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
        elif backbone_name == 'mobilenet':
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            feature_dim = model.classifier[0].in_features
            model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        return model, feature_dim
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CBM
        
        Args:
            x: Input images [batch_size, 3, 224, 224]
            
        Returns:
            concept_logits: Concept predictions [batch_size, num_concepts]
            binary_logits: Binary classification [batch_size]
        """
        # Extract features
        features = self.backbone(x)
        
        # Predict concepts
        concept_logits = self.concept_predictor(features)
        concept_probs = torch.sigmoid(concept_logits)
        
        # Predict binary classification from concepts
        binary_logits = self.binary_classifier(concept_probs)
        
        return concept_logits, binary_logits.squeeze(-1)
