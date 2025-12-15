#!/usr/bin/env python3
"""
Download pretrained weights to cache before multi-run experiments.
Run this on a node with internet access before submitting batch jobs.
"""

import torch
from torchvision import models

print("Downloading pretrained weights to cache...")
print("This only needs to be done once.")
print()

# Swin Transformer
print("Downloading Swin-T weights...")
models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
print("✓ Swin-T")

# ConvNeXt
print("Downloading ConvNeXt-Tiny weights...")
models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
print("✓ ConvNeXt-Tiny")

# ViT
print("Downloading ViT-B/16 weights...")
models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
print("✓ ViT-B/16")

# EfficientNet
print("Downloading EfficientNet-B0 weights...")
models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
print("✓ EfficientNet-B0")

# MobileNet
print("Downloading MobileNet-V3-Large weights...")
models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
print("✓ MobileNet-V3-Large")

print()
print("All weights downloaded to:", torch.hub.get_dir())
print("Multi-run experiments can now run offline on compute nodes.")
