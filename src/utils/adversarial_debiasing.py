#!/usr/bin/env python3
"""
Adversarial Debiasing Loss Functions

Implements fairness-aware loss functions for training CBMs with demographic equity constraints.

Includes:
- Demographic Parity Loss
- Equalized Odds Loss
- Calibration Fairness Loss
- Combined Fairness Loss

Author: Matt Cockayne
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import numpy as np


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training.
    
    Forward: Identity
    Backward: Negate gradients × alpha
    
    Reference:
        Ganin & Lempitsky (2015). "Unsupervised Domain Adaptation by Backpropagation"
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Apply gradient reversal to input tensor.
    
    Args:
        x: Input tensor
        alpha: Gradient reversal strength (default: 1.0)
        
    Returns:
        Output tensor (same as input in forward pass)
    """
    return GradientReversalFunction.apply(x, alpha)


def compute_demographic_parity_loss(predictions: torch.Tensor,
                                   groups: torch.Tensor,
                                   reduction: str = 'mean') -> torch.Tensor:
    """
    Demographic Parity Loss: Encourages equal positive prediction rates across groups.
    
    L_dp = Σ_i Σ_j |P(Ŷ=1|A=i) - P(Ŷ=1|A=j)|²
    
    Args:
        predictions: Binary predictions (probabilities) [batch_size]
        groups: Group membership labels [batch_size] (0-indexed integers)
        reduction: 'mean' or 'sum'
    
    Returns:
        demographic_parity_loss: Scalar loss
    
    Reference:
        Feldman et al. (2015). "Certifying and removing disparate impact"
    """
    unique_groups = torch.unique(groups)
    group_rates = []
    
    for group in unique_groups:
        group_mask = (groups == group)
        if group_mask.sum() > 0:
            # Mean prediction rate for this group
            group_rate = predictions[group_mask].mean()
            group_rates.append(group_rate)
    
    if len(group_rates) < 2:
        # Need at least 2 groups to compute disparity
        return torch.tensor(0.0, device=predictions.device)
    
    # Compute pairwise disparities
    loss = 0.0
    for i in range(len(group_rates)):
        for j in range(i + 1, len(group_rates)):
            loss += (group_rates[i] - group_rates[j]) ** 2
    
    # Normalize by number of pairs
    n_pairs = len(group_rates) * (len(group_rates) - 1) / 2
    
    if reduction == 'mean':
        return loss / n_pairs
    else:
        return loss


def compute_equalized_odds_loss(predictions: torch.Tensor,
                               labels: torch.Tensor,
                               groups: torch.Tensor,
                               reduction: str = 'mean') -> torch.Tensor:
    """
    Equalized Odds Loss: Encourages equal TPR and FPR across groups.
    
    L_eo = Σ_i Σ_j [|TPR_i - TPR_j|² + |FPR_i - FPR_j|²]
    
    Args:
        predictions: Binary predictions (probabilities) [batch_size]
        labels: True binary labels [batch_size]
        groups: Group membership labels [batch_size]
        reduction: 'mean' or 'sum'
    
    Returns:
        equalized_odds_loss: Scalar loss
    
    Reference:
        Hardt et al. (2016). "Equality of Opportunity in Supervised Learning"
    """
    unique_groups = torch.unique(groups)
    group_tpr = []
    group_fpr = []
    
    for group in unique_groups:
        group_mask = (groups == group)
        group_preds = predictions[group_mask]
        group_labels = labels[group_mask]
        
        if len(group_labels) == 0:
            continue
        
        # True Positive Rate
        positives = (group_labels == 1)
        if positives.sum() > 0:
            tpr = group_preds[positives].mean()
        else:
            tpr = torch.tensor(0.0, device=predictions.device)
        group_tpr.append(tpr)
        
        # False Positive Rate
        negatives = (group_labels == 0)
        if negatives.sum() > 0:
            fpr = group_preds[negatives].mean()
        else:
            fpr = torch.tensor(0.0, device=predictions.device)
        group_fpr.append(fpr)
    
    if len(group_tpr) < 2:
        return torch.tensor(0.0, device=predictions.device)
    
    # Compute pairwise disparities for TPR and FPR
    loss = 0.0
    for i in range(len(group_tpr)):
        for j in range(i + 1, len(group_tpr)):
            loss += (group_tpr[i] - group_tpr[j]) ** 2
            loss += (group_fpr[i] - group_fpr[j]) ** 2
    
    # Normalize
    n_pairs = len(group_tpr) * (len(group_tpr) - 1) / 2
    
    if reduction == 'mean':
        return loss / (2 * n_pairs)  # 2 for TPR + FPR
    else:
        return loss


def compute_equal_opportunity_loss(predictions: torch.Tensor,
                                   labels: torch.Tensor,
                                   groups: torch.Tensor,
                                   reduction: str = 'mean') -> torch.Tensor:
    """
    Equal Opportunity Loss: Encourages equal TPR across groups.
    
    L_eo = Σ_i Σ_j |TPR_i - TPR_j|²
    
    Relaxed version of equalized odds (only considers TPR, not FPR).
    
    Args:
        predictions: Binary predictions (probabilities) [batch_size]
        labels: True binary labels [batch_size]
        groups: Group membership labels [batch_size]
        reduction: 'mean' or 'sum'
    
    Returns:
        equal_opportunity_loss: Scalar loss
    """
    unique_groups = torch.unique(groups)
    group_tpr = []
    
    for group in unique_groups:
        group_mask = (groups == group)
        group_preds = predictions[group_mask]
        group_labels = labels[group_mask]
        
        if len(group_labels) == 0:
            continue
        
        # True Positive Rate
        positives = (group_labels == 1)
        if positives.sum() > 0:
            tpr = group_preds[positives].mean()
        else:
            tpr = torch.tensor(0.0, device=predictions.device)
        group_tpr.append(tpr)
    
    if len(group_tpr) < 2:
        return torch.tensor(0.0, device=predictions.device)
    
    # Compute pairwise disparities
    loss = 0.0
    for i in range(len(group_tpr)):
        for j in range(i + 1, len(group_tpr)):
            loss += (group_tpr[i] - group_tpr[j]) ** 2
    
    n_pairs = len(group_tpr) * (len(group_tpr) - 1) / 2
    
    if reduction == 'mean':
        return loss / n_pairs
    else:
        return loss


def compute_calibration_fairness_loss(predictions: torch.Tensor,
                                     labels: torch.Tensor,
                                     groups: torch.Tensor,
                                     n_bins: int = 10,
                                     reduction: str = 'mean') -> torch.Tensor:
    """
    Calibration Fairness Loss: Encourages equal calibration quality across groups.
    
    Penalizes variance in calibration errors across groups.
    A well-calibrated model has predicted probabilities matching empirical frequencies.
    
    Args:
        predictions: Binary predictions (probabilities) [batch_size]
        labels: True binary labels [batch_size]
        groups: Group membership labels [batch_size]
        n_bins: Number of calibration bins
        reduction: 'mean' or 'sum'
    
    Returns:
        calibration_fairness_loss: Scalar loss
    
    Reference:
        Pleiss et al. (2017). "On Fairness and Calibration"
    """
    unique_groups = torch.unique(groups)
    group_calibration_errors = []
    
    # Compute calibration error for each group
    for group in unique_groups:
        group_mask = (groups == group)
        group_preds = predictions[group_mask]
        group_labels = labels[group_mask].float()
        
        if len(group_labels) == 0:
            continue
        
        # Bin predictions
        bins = torch.linspace(0, 1, n_bins + 1, device=predictions.device)
        calibration_error = 0.0
        
        for i in range(n_bins):
            bin_mask = (group_preds >= bins[i]) & (group_preds < bins[i + 1])
            
            if i == n_bins - 1:  # Last bin includes upper boundary
                bin_mask = (group_preds >= bins[i]) & (group_preds <= bins[i + 1])
            
            if bin_mask.sum() > 0:
                # Average predicted probability in bin
                predicted_rate = group_preds[bin_mask].mean()
                # Empirical positive rate in bin
                actual_rate = group_labels[bin_mask].mean()
                # Calibration error for this bin
                calibration_error += (predicted_rate - actual_rate) ** 2
        
        group_calibration_errors.append(calibration_error)
    
    if len(group_calibration_errors) < 2:
        return torch.tensor(0.0, device=predictions.device)
    
    # Variance of calibration errors across groups
    # High variance = some groups poorly calibrated
    group_calibration_errors = torch.stack(group_calibration_errors)
    loss = torch.var(group_calibration_errors)
    
    if reduction == 'mean':
        return loss
    else:
        return loss


def compute_combined_fairness_loss(predictions: torch.Tensor,
                                   labels: torch.Tensor,
                                   groups: torch.Tensor,
                                   fairness_type: str = 'equalized_odds',
                                   demographic_parity_weight: float = 0.5,
                                   equalized_odds_weight: float = 0.5,
                                   calibration_weight: float = 0.0) -> torch.Tensor:
    """
    Combined fairness loss with multiple fairness constraints.
    
    L_fairness = w_dp * L_dp + w_eo * L_eo + w_cal * L_cal
    
    Args:
        predictions: Binary predictions (probabilities) [batch_size]
        labels: True binary labels [batch_size]
        groups: Group membership labels [batch_size]
        fairness_type: 'demographic_parity', 'equalized_odds', 'equal_opportunity', 'combined'
        demographic_parity_weight: Weight for demographic parity loss
        equalized_odds_weight: Weight for equalized odds loss
        calibration_weight: Weight for calibration fairness loss
    
    Returns:
        combined_fairness_loss: Scalar loss
    """
    if fairness_type == 'demographic_parity':
        return compute_demographic_parity_loss(predictions, groups)
    
    elif fairness_type == 'equalized_odds':
        return compute_equalized_odds_loss(predictions, labels, groups)
    
    elif fairness_type == 'equal_opportunity':
        return compute_equal_opportunity_loss(predictions, labels, groups)
    
    elif fairness_type == 'combined':
        loss = 0.0
        
        if demographic_parity_weight > 0:
            loss += demographic_parity_weight * compute_demographic_parity_loss(predictions, groups)
        
        if equalized_odds_weight > 0:
            loss += equalized_odds_weight * compute_equalized_odds_loss(predictions, labels, groups)
        
        if calibration_weight > 0:
            loss += calibration_weight * compute_calibration_fairness_loss(predictions, labels, groups)
        
        return loss
    
    else:
        raise ValueError(f"Unknown fairness_type: {fairness_type}")


class FairnessLoss(nn.Module):
    """
    PyTorch module wrapper for fairness losses.
    
    Usage:
        fairness_loss_fn = FairnessLoss(
            fairness_type='equalized_odds',
            lambda_dp=0.5,
            lambda_eo=0.5
        )
        loss = fairness_loss_fn(predictions, labels, groups)
    """
    
    def __init__(self,
                 fairness_type: str = 'equalized_odds',
                 lambda_dp: float = 0.5,
                 lambda_eo: float = 0.5,
                 lambda_cal: float = 0.0):
        """
        Initialize fairness loss module.
        
        Args:
            fairness_type: Type of fairness constraint
            lambda_dp: Weight for demographic parity
            lambda_eo: Weight for equalized odds
            lambda_cal: Weight for calibration fairness
        """
        super().__init__()
        self.fairness_type = fairness_type
        self.lambda_dp = lambda_dp
        self.lambda_eo = lambda_eo
        self.lambda_cal = lambda_cal
    
    def forward(self,
                predictions: torch.Tensor,
                labels: torch.Tensor,
                groups: torch.Tensor) -> torch.Tensor:
        """
        Compute fairness loss.
        
        Args:
            predictions: Binary predictions (probabilities) [batch_size]
            labels: True binary labels [batch_size]
            groups: Group membership labels [batch_size]
        
        Returns:
            fairness_loss: Scalar loss
        """
        return compute_combined_fairness_loss(
            predictions, labels, groups,
            fairness_type=self.fairness_type,
            demographic_parity_weight=self.lambda_dp,
            equalized_odds_weight=self.lambda_eo,
            calibration_weight=self.lambda_cal
        )


def test_fairness_losses():
    """
    Test fairness loss functions with synthetic data.
    """
    print("Testing Fairness Loss Functions...\n")
    
    # Create synthetic data
    batch_size = 32
    predictions = torch.rand(batch_size)
    labels = torch.randint(0, 2, (batch_size,))
    groups = torch.randint(0, 6, (batch_size,))  # 6 Fitzpatrick types
    
    print(f"Batch size: {batch_size}")
    print(f"Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"Labels: {labels.unique()}")
    print(f"Groups: {groups.unique()}\n")
    
    # Test demographic parity loss
    dp_loss = compute_demographic_parity_loss(predictions, groups)
    print(f"Demographic Parity Loss: {dp_loss:.4f}")
    
    # Test equalized odds loss
    eo_loss = compute_equalized_odds_loss(predictions, labels, groups)
    print(f"Equalized Odds Loss: {eo_loss:.4f}")
    
    # Test equal opportunity loss
    eop_loss = compute_equal_opportunity_loss(predictions, labels, groups)
    print(f"Equal Opportunity Loss: {eop_loss:.4f}")
    
    # Test calibration fairness loss
    cal_loss = compute_calibration_fairness_loss(predictions, labels, groups)
    print(f"Calibration Fairness Loss: {cal_loss:.4f}\n")
    
    # Test combined loss
    combined_loss = compute_combined_fairness_loss(
        predictions, labels, groups,
        fairness_type='combined',
        demographic_parity_weight=0.3,
        equalized_odds_weight=0.5,
        calibration_weight=0.2
    )
    print(f"Combined Fairness Loss: {combined_loss:.4f}\n")
    
    # Test module interface
    fairness_loss_fn = FairnessLoss(fairness_type='equalized_odds')
    module_loss = fairness_loss_fn(predictions, labels, groups)
    print(f"Module-based Loss: {module_loss:.4f}")
    
    # Test gradient flow
    predictions.requires_grad = True
    loss = compute_equalized_odds_loss(predictions, labels, groups)
    loss.backward()
    print(f"Gradient exists: {predictions.grad is not None}")
    print(f"Gradient norm: {predictions.grad.norm():.4f}\n")
    
    print("✓ All fairness loss tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Adversarial Debiasing Loss Functions Tests")
    print("=" * 60 + "\n")
    
    test_fairness_losses()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
