"""
Direct Classification Model - Simplified for FairCBM

Direct image classification without concept bottleneck.
Serves as a non-interpretable baseline for comparison.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict


class DirectClassifier(nn.Module):
    """
    Direct image classifier without concept bottleneck.
    Simple architecture: backbone → dense → binary output
    """
    
    def __init__(self, backbone: str = 'swin', hidden_dim: int = 512, 
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Load pretrained backbone
        self.backbone, self.feature_dim = self._load_backbone(backbone)
        
        # Direct classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )
        
    def _load_backbone(self, backbone_name: str):
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: image -> features -> classification"""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits.squeeze(-1)
