#!/usr/bin/env python3
"""
Fair Curriculum CBM - Fairness-First Curriculum Learning

Implements the 4-phase fairness-aware curriculum that prioritizes group balance
and fairness criteria before concept complexity.

Phases:
1. Group-Balanced Foundation (25%): Equal sampling, basic concepts
2. Demographic Parity Focus (25-50%): DP loss, medium concepts
3. Equalized Odds Focus (50-75%): EO loss + adversarial, all concepts
4. Performance Parity (75-100%): Error-driven sampling, performance gap loss

Author: Matt Cockayne
Date: December 2025
"""

import sys
import os
from pathlib import Path
import numpy as np
from collections import defaultdict

# Add FairCBM directory to path for imports
faircbm_dir = Path(__file__).resolve().parent.parent.parent
if str(faircbm_dir) not in sys.path:
    sys.path.insert(0, str(faircbm_dir))

from src.models.minimal_curriculum_cbm import MinimalCurriculumCBM
from src.models.adversarial_discriminator import AdversarialDiscriminator
from src.utils.adversarial_debiasing import (
    compute_combined_fairness_loss,
    compute_demographic_parity_loss,
    compute_equalized_odds_loss
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from typing import Dict, List, Optional, Tuple
import logging


class FairnessAwareSampler(Sampler):
    """
    Custom sampler that ensures balanced group representation per phase.
    
    Phases:
    1. Balanced Foundation: Equal samples per Fitzpatrick type
    2. Demographic Parity: Continue balanced sampling
    3. Equalized Odds: Stratified sampling (equal per group × label)
    4. Performance Parity: Error-driven sampling (oversample low-F1 groups)
    """
    
    def __init__(self, 
                 groups: torch.Tensor,
                 labels: torch.Tensor,
                 batch_size: int,
                 epoch: int,
                 total_epochs: int,
                 group_f1_scores: Optional[Dict[int, float]] = None):
        """
        Initialize FairnessAwareSampler.
        
        Args:
            groups: Fitzpatrick skin type labels [dataset_size]
            labels: Binary labels [dataset_size]
            batch_size: Batch size
            epoch: Current epoch (0-indexed)
            total_epochs: Total training epochs
            group_f1_scores: F1 scores per group (for Phase 4 error-driven sampling)
        """
        self.groups = groups.cpu().numpy()
        self.labels = labels.cpu().numpy()
        self.batch_size = batch_size
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.group_f1_scores = group_f1_scores or {}
        
        self.num_groups = 6  # Fitzpatrick types 1-6 (stored as 0-5)
        self.dataset_size = len(groups)
        
        # Precompute indices per group
        self.indices_per_group = {g: [] for g in range(self.num_groups)}
        for idx, g in enumerate(self.groups):
            self.indices_per_group[g].append(idx)
        
        # Precompute indices per (group, label) stratum
        self.indices_per_stratum = {}
        for g in range(self.num_groups):
            for y in [0, 1]:
                stratum_key = (g, y)
                mask = (self.groups == g) & (self.labels == y)
                self.indices_per_stratum[stratum_key] = np.where(mask)[0].tolist()
        
        self.phase = self._get_phase()
        
    def _get_phase(self) -> str:
        """Determine current curriculum phase."""
        progress = self.epoch / self.total_epochs
        
        if progress <= 0.25:
            return 'balanced_foundation'
        elif progress <= 0.50:
            return 'demographic_parity'
        elif progress <= 0.75:
            return 'equalized_odds'
        else:
            return 'performance_parity'
    
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return self.dataset_size // self.batch_size
    
    def __iter__(self):
        """Generate batch indices based on current phase."""
        if self.phase in ['balanced_foundation', 'demographic_parity']:
            return self._balanced_group_sampling()
        elif self.phase == 'equalized_odds':
            return self._stratified_sampling()
        elif self.phase == 'performance_parity':
            return self._error_driven_sampling()
        else:
            raise ValueError(f"Unknown phase: {self.phase}")
    
    def _balanced_group_sampling(self):
        """Sample equally from each Fitzpatrick type."""
        samples_per_group = self.batch_size // self.num_groups
        
        all_batch_indices = []
        num_batches = len(self)
        
        for _ in range(num_batches):
            batch_indices = []
            
            for g in range(self.num_groups):
                group_indices = self.indices_per_group[g]
                if len(group_indices) == 0:
                    continue
                
                # Sample with replacement to handle minority groups
                sampled = np.random.choice(
                    group_indices,
                    size=samples_per_group,
                    replace=True
                )
                batch_indices.extend(sampled)
            
            # Shuffle within batch
            np.random.shuffle(batch_indices)
            all_batch_indices.append(batch_indices)  # Append as batch, not extend
        
        return iter(all_batch_indices)  # Returns iterator of batches
    
    def _stratified_sampling(self):
        """Sample equally from each (Fitzpatrick × Label) stratum."""
        num_strata = self.num_groups * 2  # 6 groups × 2 labels = 12 strata
        samples_per_stratum = max(1, self.batch_size // num_strata)
        
        all_batch_indices = []
        num_batches = len(self)
        
        for _ in range(num_batches):
            batch_indices = []
            
            for g in range(self.num_groups):
                for y in [0, 1]:
                    stratum_key = (g, y)
                    stratum_indices = self.indices_per_stratum[stratum_key]
                    
                    if len(stratum_indices) == 0:
                        continue
                    
                    # Sample with replacement for small strata
                    sampled = np.random.choice(
                        stratum_indices,
                        size=samples_per_stratum,
                        replace=True
                    )
                    batch_indices.extend(sampled)
            
            # Trim or pad to exact batch_size
            if len(batch_indices) > self.batch_size:
                batch_indices = np.random.choice(batch_indices, self.batch_size, replace=False).tolist()
            
            np.random.shuffle(batch_indices)
            all_batch_indices.append(batch_indices)  # Append as batch, not extend
        
        return iter(all_batch_indices)  # Returns iterator of batches
    
    def _error_driven_sampling(self):
        """Oversample groups with lower F1 scores."""
        # Compute sampling weights: lower F1 → higher weight
        if not self.group_f1_scores:
            # Fall back to balanced sampling if no F1 scores available
            return self._balanced_group_sampling()
        
        weights = {}
        epsilon = 0.1  # Prevent division by zero
        min_f1_threshold = 0.1  # Ignore groups with F1 < 0.1 (likely missing from val set)
        
        # Filter out groups with very low F1 (missing or insufficient data)
        valid_groups = {g: f1 for g, f1 in self.group_f1_scores.items() if f1 >= min_f1_threshold}
        
        if not valid_groups:
            # No valid groups, fall back to balanced sampling
            return self._balanced_group_sampling()
        
        for g in range(self.num_groups):
            if g in valid_groups:
                f1 = valid_groups[g]
                weights[g] = 1.0 / (f1 + epsilon)
            else:
                # Groups with F1 < threshold get minimal weight (not excluded completely)
                weights[g] = 0.5  # Lower than even high-performing groups
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {g: w / total_weight for g, w in weights.items()}
        
        all_batch_indices = []
        num_batches = len(self)
        
        for _ in range(num_batches):
            batch_indices = []
            
            for g in range(self.num_groups):
                group_indices = self.indices_per_group[g]
                if len(group_indices) == 0:
                    continue
                
                # Number of samples proportional to weight
                num_samples = int(self.batch_size * weights[g])
                if num_samples == 0:
                    num_samples = 1  # Ensure at least one sample
                
                sampled = np.random.choice(
                    group_indices,
                    size=num_samples,
                    replace=True
                )
                batch_indices.extend(sampled)
            
            # Trim to exact batch_size
            if len(batch_indices) > self.batch_size:
                batch_indices = np.random.choice(batch_indices, self.batch_size, replace=False).tolist()
            
            np.random.shuffle(batch_indices)
            all_batch_indices.append(batch_indices)  # Append as batch, not extend
        
        return iter(all_batch_indices)  # Returns iterator of batches


class PhasedFairnessLoss(nn.Module):
    """
    Phased fairness loss that emphasizes different criteria per phase.
    
    Phases:
    1. Balanced Foundation: No fairness loss (rely on balanced sampling)
    2. Demographic Parity: Focus on DP loss only
    3. Equalized Odds: Shift to EO loss (0.3 DP + 0.7 EO)
    4. Performance Parity: Balance all three (0.33 DP + 0.33 EO + 0.34 PG)
    """
    
    def __init__(self, total_epochs: int, num_groups: int = 6,
                 disabled_phases: Optional[List[int]] = None):
        """
        Initialize PhasedFairnessLoss.
        
        Args:
            total_epochs: Total training epochs
            num_groups: Number of demographic groups
            disabled_phases: List of phase numbers (1-4) to disable for ablation
        """
        super().__init__()
        self.total_epochs = total_epochs
        self.num_groups = num_groups
        self.disabled_phases = disabled_phases or []
    
    def _get_phase(self, epoch: int) -> str:
        """Determine current phase based on epoch."""
        progress = epoch / self.total_epochs
        
        if progress <= 0.25:
            return 'balanced_foundation'
        elif progress <= 0.50:
            return 'demographic_parity'
        elif progress <= 0.75:
            return 'equalized_odds'
        else:
            return 'performance_parity'
    
    def forward(self,
                predictions: torch.Tensor,
                labels: torch.Tensor,
                groups: torch.Tensor,
                epoch: int) -> torch.Tensor:
        """
        Compute phased fairness loss.
        
        Args:
            predictions: Binary predictions [batch_size, 1] (logits or probabilities)
            labels: Binary labels [batch_size]
            groups: Group labels [batch_size] (0-5 for Fitzpatrick I-VI)
            epoch: Current epoch
            
        Returns:
            Weighted fairness loss
        """
        phase = self._get_phase(epoch)
        progress = epoch / self.total_epochs
        
        # Determine current phase number for ablation
        if progress <= 0.25:
            phase_num = 1
        elif progress <= 0.50:
            phase_num = 2
        elif progress <= 0.75:
            phase_num = 3
        else:
            phase_num = 4
        
        # If current phase is disabled, use fallback logic
        if phase_num in self.disabled_phases:
            # Phase 1 disabled → skip to Phase 2 behavior (DP from start)
            if phase_num == 1 and 2 not in self.disabled_phases:
                phase = 'demographic_parity'
            # Phase 2 disabled → use Phase 1 behavior (no fairness)
            elif phase_num == 2:
                return torch.tensor(0.0, device=predictions.device)
            # Phase 3 disabled → continue Phase 2 behavior (DP only)
            elif phase_num == 3 and 2 not in self.disabled_phases:
                phase = 'demographic_parity'
            # Phase 4 disabled → continue Phase 3 behavior (DP + EO)
            elif phase_num == 4 and 3 not in self.disabled_phases:
                phase = 'equalized_odds'
        
        # Ensure predictions are probabilities
        if predictions.min() < 0 or predictions.max() > 1:
            probs = torch.sigmoid(predictions)
        else:
            probs = predictions
        
        # Phase 1: No fairness loss
        if phase == 'balanced_foundation':
            return torch.tensor(0.0, device=predictions.device)
        
        # Compute individual fairness losses
        L_dp = self._demographic_parity_loss(probs, groups)
        L_eo = self._equalized_odds_loss(probs, labels, groups)
        
        # Phase 2: Focus on demographic parity
        if phase == 'demographic_parity':
            return L_dp
        
        # Phase 3: Shift to equalized odds
        elif phase == 'equalized_odds':
            return 0.3 * L_dp + 0.7 * L_eo
        
        # Phase 4: Balance all criteria including performance gap
        elif phase == 'performance_parity':
            L_pg = self._performance_gap_loss(probs, labels, groups)
            return 0.33 * L_dp + 0.33 * L_eo + 0.34 * L_pg
        
        else:
            return torch.tensor(0.0, device=predictions.device)
    
    def _demographic_parity_loss(self,
                                  predictions: torch.Tensor,
                                  groups: torch.Tensor) -> torch.Tensor:
        """
        Minimize disparity in positive prediction rates.
        
        L_dp = Σ |P(Ŷ=1|A=a) - P(Ŷ=1|A=a')|²
        """
        # Compute mean prediction per group
        group_rates = []
        for g in range(self.num_groups):
            mask = (groups == g)
            if mask.sum() > 0:
                rate = predictions[mask].mean()
                group_rates.append(rate)
        
        if len(group_rates) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Pairwise absolute differences
        loss = 0.0
        count = 0
        for i in range(len(group_rates)):
            for j in range(i + 1, len(group_rates)):
                loss += (group_rates[i] - group_rates[j]) ** 2
                count += 1
        
        return loss / count if count > 0 else torch.tensor(0.0, device=predictions.device)
    
    def _equalized_odds_loss(self,
                             predictions: torch.Tensor,
                             labels: torch.Tensor,
                             groups: torch.Tensor) -> torch.Tensor:
        """
        Minimize disparity in TPR and FPR.
        
        L_eo = Σ |P(Ŷ=1|Y=y,A=a) - P(Ŷ=1|Y=y,A=a')|²  for y ∈ {0,1}
        """
        loss_tpr = 0.0
        loss_fpr = 0.0
        
        # TPR disparity (Y=1)
        group_tpr = []
        for g in range(self.num_groups):
            mask = (groups == g) & (labels == 1)
            if mask.sum() > 0:
                tpr = predictions[mask].mean()
                group_tpr.append(tpr)
        
        if len(group_tpr) >= 2:
            for i in range(len(group_tpr)):
                for j in range(i + 1, len(group_tpr)):
                    loss_tpr += (group_tpr[i] - group_tpr[j]) ** 2
            loss_tpr /= (len(group_tpr) * (len(group_tpr) - 1) / 2)
        
        # FPR disparity (Y=0)
        group_fpr = []
        for g in range(self.num_groups):
            mask = (groups == g) & (labels == 0)
            if mask.sum() > 0:
                fpr = predictions[mask].mean()
                group_fpr.append(fpr)
        
        if len(group_fpr) >= 2:
            for i in range(len(group_fpr)):
                for j in range(i + 1, len(group_fpr)):
                    loss_fpr += (group_fpr[i] - group_fpr[j]) ** 2
            loss_fpr /= (len(group_fpr) * (len(group_fpr) - 1) / 2)
        
        return loss_tpr + loss_fpr
    
    def _performance_gap_loss(self,
                              predictions: torch.Tensor,
                              labels: torch.Tensor,
                              groups: torch.Tensor) -> torch.Tensor:
        """
        Minimize the range of F1 scores across groups.
        
        L_pg = (max_g F1_g - min_g F1_g)²
        """
        group_f1 = []
        
        for g in range(self.num_groups):
            mask = (groups == g)
            if mask.sum() > 0:
                preds_g = (predictions[mask] > 0.5).float()
                labels_g = labels[mask].float()
                
                # Compute F1 for this group
                tp = (preds_g * labels_g).sum()
                fp = (preds_g * (1 - labels_g)).sum()
                fn = ((1 - preds_g) * labels_g).sum()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                
                group_f1.append(f1)
        
        if len(group_f1) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Performance gap: max - min
        gap = max(group_f1) - min(group_f1)
        return gap ** 2


class FairCurriculumCBM(MinimalCurriculumCBM):
    """
    Fair Curriculum CBM with Fairness-First Learning.
    
    Implements 4-phase fairness-aware curriculum:
    1. Group-Balanced Foundation
    2. Demographic Parity Focus
    3. Equalized Odds Focus
    4. Performance Parity
    
    Each phase coordinates:
    - Data sampling strategy
    - Fairness loss emphasis
    - Concept curriculum progression
    - Adversarial debiasing schedule
    """
    
    def __init__(self,
                 num_concepts: int,
                 backbone: str = 'swin',
                 num_groups: int = 6,
                 fairness_lambda: float = 0.1,
                 adversarial_lambda_target: float = 0.01,
                 dropout_rate: float = 0.1,
                 device: torch.device = None,
                 concept_names: Optional[List[str]] = None,
                 disabled_phases: Optional[List[int]] = None,
                 disable_adversarial: bool = False):
        """
        Initialize Fair Curriculum CBM.
        
        Args:
            num_concepts: Number of concepts (23 for SkinCap)
            backbone: Backbone architecture name
            num_groups: Number of demographic groups (6 for Fitzpatrick I-VI)
            fairness_lambda: Weight for fairness loss (applied in Phase 2+)
            adversarial_lambda_target: Target weight for adversarial loss (Phase 3+)
            dropout_rate: Dropout rate for classifiers
            device: Torch device
            concept_names: List of concept names (optional, for interpretability)
            disabled_phases: List of phase numbers (1-4) to disable for ablation
            disable_adversarial: Disable adversarial debiasing for ablation
        """
        # Initialize parent curriculum CBM
        super().__init__(
            num_concepts=num_concepts,
            backbone=backbone,
            dropout_rate=dropout_rate,
            device=device
        )
        
        self.num_groups = num_groups
        self.fairness_lambda = fairness_lambda
        self.adversarial_lambda_target = adversarial_lambda_target
        self.concept_names = concept_names or [f"concept_{i}" for i in range(num_concepts)]
        self.disabled_phases = disabled_phases or []
        self.disable_adversarial = disable_adversarial
        
        # Phased fairness loss
        self.fairness_loss_fn = PhasedFairnessLoss(
            total_epochs=100,  # Default, will be updated during training
            num_groups=num_groups,
            disabled_phases=disabled_phases
        )
        
        # Adversarial discriminator for Fitzpatrick type prediction
        self.adversarial_discriminator = AdversarialDiscriminator(
            input_dim=self.num_features,
            hidden_dims=[512, 256, 128],
            num_groups=num_groups,
            dropout=0.3
        ).to(self.device)
        
        # Track per-group F1 scores for Phase 4 error-driven sampling
        self.group_f1_scores = {}
        
        # Current adversarial lambda (warmup from 0 to target)
        self.current_adversarial_lambda = 0.0
        
        logging.info(f"Initialized Fair Curriculum CBM:")
        logging.info(f"  Concepts: {num_concepts}")
        logging.info(f"  Groups: {num_groups}")
        logging.info(f"  Fairness λ: {fairness_lambda}")
        logging.info(f"  Adversarial λ target: {adversarial_lambda_target}")
    
    def get_phase_info(self, epoch: int, total_epochs: int) -> Dict:
        """
        Get current phase information.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total training epochs
            
        Returns:
            dict with phase name, fairness focus, adversarial status
        """
        progress = epoch / total_epochs
        
        if progress <= 0.25:
            phase_name = 'balanced_foundation'
            fairness_focus = 'None (balanced sampling only)'
            adversarial_active = False
            
        elif progress <= 0.50:
            phase_name = 'demographic_parity'
            fairness_focus = 'Demographic Parity'
            adversarial_active = False
            
        elif progress <= 0.75:
            phase_name = 'equalized_odds'
            fairness_focus = 'Equalized Odds (+ DP)'
            adversarial_active = True
            
        else:
            phase_name = 'performance_parity'
            fairness_focus = 'Performance Gap (+ DP + EO)'
            adversarial_active = True
        
        return {
            'phase_name': phase_name,
            'fairness_focus': fairness_focus,
            'adversarial_active': adversarial_active,
            'progress': progress,
            'num_concepts': self.num_concepts  # All concepts used throughout
        }
    
    def compute_adversarial_lambda(self, epoch: int, total_epochs: int) -> float:
        """
        Compute current adversarial lambda with warmup.
        
        Schedule:
        - Phase 1-2 (0-50%): λ_adv = 0
        - Phase 3 (50-75%): Linear warmup from 0 to target
        - Phase 4 (75-100%): Full target weight
        """
        # If adversarial is disabled for ablation, return 0
        if self.disable_adversarial:
            return 0.0
        
        # If Phase 3 is disabled, adversarial never activates
        if 3 in self.disabled_phases:
            return 0.0
        
        progress = epoch / total_epochs
        
        if progress <= 0.50:
            return 0.0
        elif progress <= 0.75:
            # Linear warmup during Phase 3
            phase_progress = (progress - 0.50) / 0.25  # 0 to 1 within phase
            return self.adversarial_lambda_target * phase_progress
        else:
            return self.adversarial_lambda_target
    
    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Forward pass with optional feature extraction.
        
        Args:
            x: Input images [batch_size, 3, 224, 224]
            return_features: If True, return intermediate features for adversarial
            
        Returns:
            concept_logits: [batch_size, num_concepts]
            binary_logits: [batch_size, 1]
            features: [batch_size, num_features] (if return_features=True)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Handle different feature shapes
        if len(features.shape) == 4:
            if features.shape[-1] > features.shape[1]:
                features = features.permute(0, 3, 1, 2)
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        elif len(features.shape) == 3:
            features = features[:, 0, :]  # CLS token
        
        # Concept prediction
        concept_logits = self.concept_layer(features)
        
        # Binary classification from concepts
        binary_logits = self.binary_classifier(torch.sigmoid(concept_logits))
        
        if return_features:
            return concept_logits, binary_logits, features
        else:
            return concept_logits, binary_logits
    
    def compute_loss(self,
                     concept_logits: torch.Tensor,
                     binary_logits: torch.Tensor,
                     concept_labels: torch.Tensor,
                     binary_labels: torch.Tensor,
                     groups: torch.Tensor,
                     features: Optional[torch.Tensor] = None,
                     epoch: int = 0,
                     total_epochs: int = 100) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with phased fairness and adversarial components.
        
        Fair Curriculum CBM uses joint concept training (all 23 concepts throughout)
        while varying fairness objectives by phase.
        
        Args:
            concept_logits: Concept predictions [batch_size, num_concepts]
            binary_logits: Binary predictions [batch_size, 1]
            concept_labels: Concept ground truth [batch_size, num_concepts]
            binary_labels: Binary ground truth [batch_size]
            groups: Fitzpatrick group labels [batch_size]
            features: Backbone features [batch_size, num_features] (for adversarial)
            epoch: Current epoch
            total_epochs: Total training epochs
            
        Returns:
            Dictionary with 'total', 'concept', 'binary', 'fairness', 'adversarial' losses
        """
        # 1. Concept loss (all concepts trained jointly)
        concept_loss = F.binary_cross_entropy_with_logits(
            concept_logits,
            concept_labels
        )
        
        # 2. Binary classification loss
        binary_labels_reshaped = binary_labels.view(-1, 1).float()
        binary_loss = F.binary_cross_entropy_with_logits(
            binary_logits,
            binary_labels_reshaped
        )
        
        # 3. Phased fairness loss
        binary_probs = torch.sigmoid(binary_logits).squeeze(-1)
        fairness_loss = self.fairness_loss_fn(
            binary_probs,
            binary_labels,
            groups,
            epoch
        )
        
        # 4. Adversarial debiasing loss (Phase 3+)
        adversarial_lambda = self.compute_adversarial_lambda(epoch, total_epochs)
        
        if adversarial_lambda > 0 and features is not None:
            # Gradient reversal: discriminator tries to predict group from features
            from src.utils.adversarial_debiasing import gradient_reversal
            
            reversed_features = gradient_reversal(features, alpha=1.0)
            group_logits = self.adversarial_discriminator(reversed_features)
            
            # Adversarial loss with label smoothing
            adversarial_loss = F.cross_entropy(
                group_logits,
                groups,
                label_smoothing=0.1
            )
            
            # Clamp to prevent explosion
            adversarial_loss = torch.clamp(adversarial_loss, max=10.0)
        else:
            adversarial_loss = torch.tensor(0.0, device=concept_logits.device)
        
        # 5. Total loss
        total_loss = (
            concept_loss +
            binary_loss +
            self.fairness_lambda * fairness_loss +
            adversarial_lambda * adversarial_loss
        )
        
        # Store current lambda for tracking
        self.current_adversarial_lambda = adversarial_lambda
        
        return {
            'total': total_loss,
            'concept': concept_loss,
            'binary': binary_loss,
            'fairness': fairness_loss,
            'adversarial': adversarial_loss
        }
    
    def update_group_f1_scores(self, group_f1_dict: Dict[int, float]):
        """
        Update per-group F1 scores for error-driven sampling in Phase 4.
        
        Args:
            group_f1_dict: Dictionary mapping group ID (0-5) to F1 score
        """
        self.group_f1_scores = group_f1_dict.copy()
        
        logging.info(f"Updated group F1 scores for error-driven sampling:")
        for g, f1 in sorted(group_f1_dict.items()):
            logging.info(f"  Group {g}: F1 = {f1:.3f}")
