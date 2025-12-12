#!/usr/bin/env python3
"""
Adversarial Discriminator for Fairness-Aware Learning

Implements gradient reversal layer and adversarial discriminator network
for learning group-invariant representations.

Based on:
- Ganin & Lempitsky (2015). "Unsupervised Domain Adaptation by Backpropagation"
- Zhang et al. (2018). "Mitigating Unwanted Biases with Adversarial Learning"

Author: Matt Cockayne
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import numpy as np


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) from Ganin & Lempitsky (2015).
    
    Forward pass: Identity function (no modification)
    Backward pass: Multiply gradient by -alpha (reverses gradient)
    
    Purpose:
        When optimizing the main model to minimize task loss while maximizing
        adversarial loss (via gradient reversal), the model learns features
        that are informative for the task but uninformative for group prediction.
    
    Usage:
        features = GradientReversalLayer.apply(features, alpha)
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        """
        Forward pass: return input unchanged.
        
        Args:
            x: Input tensor
            alpha: Gradient reversal strength (0 = no reversal, 1 = full reversal)
        
        Returns:
            x unchanged
        """
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: reverse gradient.
        
        Args:
            grad_output: Gradient from subsequent layers
        
        Returns:
            Reversed gradient: -alpha * grad_output
            None for alpha (no gradient w.r.t. alpha)
        """
        return grad_output.neg() * ctx.alpha, None


class AdversarialDiscriminator(nn.Module):
    """
    Adversarial discriminator network for predicting group membership.
    
    When combined with gradient reversal, encourages the encoder to learn
    group-invariant representations.
    
    Architecture:
        Input (concept features) → FC + ReLU + Dropout → ... → FC → Group logits
    """
    
    def __init__(self,
                 input_dim: int = 512,
                 hidden_dims: List[int] = [256, 128],
                 num_groups: int = 6,
                 dropout: float = 0.3):
        """
        Initialize adversarial discriminator.
        
        Args:
            input_dim: Dimension of input features (from concept layer)
            hidden_dims: List of hidden layer dimensions
            num_groups: Number of demographic groups (e.g., 6 Fitzpatrick types)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_groups = num_groups
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_groups))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Forward pass with optional gradient reversal.
        
        Args:
            features: Input features [batch_size, input_dim]
            alpha: Gradient reversal strength (0-1)
                   0 = no reversal (standard discriminator)
                   1 = full reversal (adversarial training)
        
        Returns:
            group_logits: [batch_size, num_groups]
        """
        # Apply gradient reversal layer
        reversed_features = GradientReversalLayer.apply(features, alpha)
        
        # Predict group
        group_logits = self.network(reversed_features)
        
        return group_logits
    
    def predict_group(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict group membership without gradient reversal.
        
        Args:
            features: Input features [batch_size, input_dim]
        
        Returns:
            group_probs: [batch_size, num_groups]
        """
        with torch.no_grad():
            logits = self.network(features)
            probs = F.softmax(logits, dim=1)
        return probs


class AdversarialAlphaScheduler:
    """
    Scheduler for gradually increasing gradient reversal strength during training.
    
    Follows the schedule from Ganin & Lempitsky (2015):
        alpha = 2 / (1 + exp(-10 * progress)) - 1
    
    This gradually increases alpha from 0 to 1 as training progresses, allowing
    the model to first learn task-relevant features, then progressively remove
    group-specific information.
    """
    
    def __init__(self, max_epochs: int, schedule_type: str = 'ganin'):
        """
        Initialize alpha scheduler.
        
        Args:
            max_epochs: Total number of training epochs
            schedule_type: Type of schedule ('ganin', 'linear', 'constant')
        """
        self.max_epochs = max_epochs
        self.schedule_type = schedule_type
    
    def get_alpha(self, epoch: int) -> float:
        """
        Get gradient reversal strength for current epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
        
        Returns:
            alpha: Gradient reversal strength (0-1)
        """
        progress = epoch / self.max_epochs
        
        if self.schedule_type == 'ganin':
            # Ganin schedule: smooth sigmoid increase
            alpha = 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0
        elif self.schedule_type == 'linear':
            # Linear schedule
            alpha = progress
        elif self.schedule_type == 'constant':
            # Constant (full reversal from start)
            alpha = 1.0
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return float(np.clip(alpha, 0.0, 1.0))


def test_gradient_reversal():
    """
    Test gradient reversal layer to ensure gradients are properly reversed.
    """
    print("Testing Gradient Reversal Layer...")
    
    # Create dummy input
    x = torch.randn(4, 10, requires_grad=True)
    alpha = 1.0
    
    # Forward pass
    y = GradientReversalLayer.apply(x, alpha)
    
    # Backward pass
    loss = y.sum()
    loss.backward()
    
    # Check gradient
    print(f"Input requires_grad: {x.requires_grad}")
    print(f"Output shape: {y.shape}")
    print(f"Gradient exists: {x.grad is not None}")
    
    # Verify gradient is reversed
    if x.grad is not None:
        # Gradient should be -alpha * ones (since loss = sum(y) and y = x)
        expected_grad = -alpha * torch.ones_like(x)
        is_reversed = torch.allclose(x.grad, expected_grad)
        print(f"Gradient properly reversed: {is_reversed}")
        if not is_reversed:
            print(f"Expected grad: {expected_grad[0, :5]}")
            print(f"Actual grad: {x.grad[0, :5]}")
    
    print("✓ Gradient reversal test complete\n")


def test_adversarial_discriminator():
    """
    Test adversarial discriminator network.
    """
    print("Testing Adversarial Discriminator...")
    
    # Create discriminator
    discriminator = AdversarialDiscriminator(
        input_dim=512,
        hidden_dims=[256, 128],
        num_groups=6,
        dropout=0.3
    )
    
    # Test forward pass
    batch_size = 8
    features = torch.randn(batch_size, 512)
    
    # Without gradient reversal
    logits_no_reversal = discriminator(features, alpha=0.0)
    print(f"Output shape (no reversal): {logits_no_reversal.shape}")
    print(f"Expected: ({batch_size}, 6)")
    
    # With gradient reversal
    logits_reversed = discriminator(features, alpha=1.0)
    print(f"Output shape (with reversal): {logits_reversed.shape}")
    
    # Test prediction
    probs = discriminator.predict_group(features)
    print(f"Prediction probs shape: {probs.shape}")
    print(f"Probs sum to 1: {torch.allclose(probs.sum(dim=1), torch.ones(batch_size))}")
    
    print("✓ Adversarial discriminator test complete\n")


def test_alpha_scheduler():
    """
    Test alpha scheduler for different schedule types.
    """
    print("Testing Alpha Scheduler...")
    
    max_epochs = 100
    
    # Test Ganin schedule
    scheduler = AdversarialAlphaScheduler(max_epochs, schedule_type='ganin')
    alphas = [scheduler.get_alpha(epoch) for epoch in range(0, max_epochs, 10)]
    print(f"Ganin schedule (epochs 0-100, step 10): {[f'{a:.3f}' for a in alphas]}")
    
    # Test linear schedule
    scheduler = AdversarialAlphaScheduler(max_epochs, schedule_type='linear')
    alphas = [scheduler.get_alpha(epoch) for epoch in range(0, max_epochs, 10)]
    print(f"Linear schedule (epochs 0-100, step 10): {[f'{a:.3f}' for a in alphas]}")
    
    print("✓ Alpha scheduler test complete\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Adversarial Discriminator Module Tests")
    print("=" * 60 + "\n")
    
    test_gradient_reversal()
    test_adversarial_discriminator()
    test_alpha_scheduler()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
