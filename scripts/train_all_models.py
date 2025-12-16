"""
Training script for all 4 model types: Direct, Standard CBM, Curriculum CBM, Fair Curriculum CBM.

This script provides fair comparison by training all models with identical:
- Data splits
- Backbones
- Hyperparameters
- Training procedures

Only difference: Fair Curriculum CBM uses Fitzpatrick labels for fairness constraints.

Usage:
    python train_all_models.py --model_type direct --backbone swin --exp_name exp_001
    python train_all_models.py --model_type standard_cbm --backbone swin --exp_name exp_001
    python train_all_models.py --model_type curriculum_cbm --backbone swin --exp_name exp_001
    python train_all_models.py --model_type fair_curriculum_cbm --backbone swin --exp_name exp_001
"""

import sys
import os
from pathlib import Path

# Add FairCBM to path
faircbm_root = Path(__file__).parent.parent
sys.path.insert(0, str(faircbm_root))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import fcntl
import tempfile
from datetime import datetime
from tqdm import tqdm

# Import FairCBM components (all local)
from src.data.dataloader import SkinCapDataset
from src.models.direct_classifier import DirectClassifier
from src.models.standard_cbm import StandardCBM
from src.models.minimal_curriculum_cbm import MinimalCurriculumCBM
from src.models.fairness_aware_cbm import FairnessAwareCBM
from src.models.fair_curriculum_cbm import FairCurriculumCBM, FairnessAwareSampler
from src.utils.metrics import compute_metrics
from src.utils.fairness_metrics import compute_all_fairness_metrics


def check_and_save_global_best(model_type, val_f1, model, optimizer, epoch, config, save_dir, global_best_path, performance_gap=None):
    """
    Check if current model is globally best for its type and save if so.
    Uses file locking for atomic updates across all parallel runs.
    
    For Fair Curriculum CBM and Fair Standard CBM, uses composite metric:
        composite_score = F1 * (1 - performance_gap)
    This balances high F1 with low fairness disparity.
    
    For other models, uses F1 only.
    
    Args:
        model_type: Type of model (fair_curriculum_cbm, etc.)
        val_f1: Validation F1 score
        model: Model instance
        optimizer: Optimizer instance
        epoch: Current epoch
        config: Training config dict
        save_dir: Directory to save model
        global_best_path: Path to global best tracking JSON
        performance_gap: Performance gap between best and worst group (optional, for fairness-aware saving)
    
    Returns:
        bool: True if model was saved (is global best), False otherwise
    """
    global_best_path = Path(global_best_path)
    global_best_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine if this is a fairness-focused model
    is_fairness_model = model_type in ['fair_curriculum_cbm', 'fair_standard_cbm']
    
    # Compute composite metric for fairness models
    if is_fairness_model and performance_gap is not None:
        # Composite score balances F1 and fairness
        # F1 * (1 - gap) penalizes high disparity
        # Example: F1=0.8, gap=0.5 → score=0.4 (bad fairness hurts)
        #          F1=0.7, gap=0.2 → score=0.56 (better fairness wins)
        composite_score = val_f1 * (1.0 - performance_gap)
        metric_name = "composite (F1 * (1-gap))"
        metric_value = composite_score
    else:
        # Non-fairness models: use F1 only
        composite_score = val_f1
        metric_name = "F1"
        metric_value = val_f1
    
    # Acquire exclusive lock on best-models.json
    lock_file = global_best_path.parent / '.best-models.lock'
    lock_file.touch(exist_ok=True)
    
    with open(lock_file, 'r') as lock_fd:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        
        try:
            # Load current global best scores
            if global_best_path.exists():
                with open(global_best_path, 'r') as f:
                    global_best = json.load(f)
            else:
                global_best = {}
            
            # Check if current model beats global best
            # For fairness models, compare composite scores; otherwise compare F1
            current_best_score = global_best.get(model_type, {}).get('composite_score', 0.0)
            if not current_best_score:  # Backwards compatibility with old tracking files
                current_best_score = global_best.get(model_type, {}).get('val_f1', 0.0)
            
            is_global_best = composite_score > current_best_score
            
            if is_global_best:
                # Remove old best model for this type
                old_model_path = global_best.get(model_type, {}).get('model_path', None)
                if old_model_path and Path(old_model_path).exists():
                    try:
                        Path(old_model_path).unlink()
                        print(f"Removed old global best: {old_model_path}")
                    except Exception as e:
                        print(f"Warning: Could not remove old model: {e}")
                
                # Save new best model
                model_filename = f'global_best_{model_type}.pt'
                model_path = save_dir / model_filename
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_f1,
                    'performance_gap': performance_gap,
                    'composite_score': composite_score,
                    'config': config
                }, model_path)
                
                # Update global best tracking
                global_best[model_type] = {
                    'val_f1': val_f1,
                    'performance_gap': performance_gap if performance_gap is not None else 'N/A',
                    'composite_score': composite_score,
                    'epoch': epoch,
                    'model_path': str(model_path),
                    'timestamp': datetime.now().isoformat(),
                    'exp_name': config.get('exp_name', 'unknown')
                }
                
                # Atomic write of updated best-models.json
                with tempfile.NamedTemporaryFile('w', dir=global_best_path.parent, delete=False) as tmp:
                    json.dump(global_best, tmp, indent=2)
                    tmp_path = tmp.name
                
                os.replace(tmp_path, global_best_path)
                
                if is_fairness_model and performance_gap is not None:
                    print(f"✓ New global best {model_type}! {metric_name}: {metric_value:.4f} (F1: {val_f1:.4f}, gap: {performance_gap:.4f}) (prev: {current_best_score:.4f})")
                else:
                    print(f"✓ New global best {model_type}! {metric_name}: {metric_value:.4f} (prev: {current_best_score:.4f})")
                return True
            else:
                if is_fairness_model and performance_gap is not None:
                    print(f"Not global best for {model_type} (current {metric_name}: {metric_value:.4f} < best: {current_best_score:.4f})")
                else:
                    print(f"Not global best for {model_type} (current {metric_name}: {metric_value:.4f} < best: {current_best_score:.4f})")
                return False
                
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)


def load_concept_names(concepts_path='data/skincap_concepts.txt'):
    """Load concept names from file."""
    concepts_path = Path(concepts_path)
    if not concepts_path.exists():
        # Fallback to celeba if skincap not found
        concepts_path = Path('data/concepts.txt')
    
    with open(concepts_path, 'r') as f:
        concepts = [line.strip() for line in f if line.strip()]
    return concepts


def create_model(model_type, num_concepts, backbone='swin', fairness_lambda=1.0, adversarial_lambda=0.5, concept_names=None):
    """
    Create model based on type.
    
    Args:
        model_type: 'direct', 'standard_cbm', 'fair_standard_cbm', 'curriculum_cbm', 'fair_curriculum_cbm'
        num_concepts: Number of concepts (integer)
        backbone: Backbone architecture
        fairness_lambda: Weight for fairness loss (Fair CBM only)
        adversarial_lambda: Weight for adversarial loss (Fair CBM only)
        concept_names: List of concept names (for Fair CBM)
    
    Returns:
        model: PyTorch model
    """
    if model_type == 'direct':
        model = DirectClassifier(backbone=backbone)
    
    elif model_type == 'standard_cbm':
        model = StandardCBM(
            num_concepts=num_concepts,
            backbone=backbone
        )
    
    elif model_type == 'curriculum_cbm':
        model = MinimalCurriculumCBM(
            num_concepts=num_concepts,
            backbone=backbone
        )
    
    elif model_type == 'fair_standard_cbm':
        # Fairness-aware Standard CBM WITHOUT concept curriculum (only adversarial warmup)
        if concept_names is None:
            concept_names = [f'concept_{i}' for i in range(num_concepts)]
        
        model = FairnessAwareCBM(
            concept_names=concept_names,
            backbone=backbone,
            num_groups=6,  # Fitzpatrick types I-VI
            fairness_lambda=fairness_lambda,
            adversarial_lambda=adversarial_lambda,
            use_concept_curriculum=False  # No concept staging
        )
    
    elif model_type == 'fair_curriculum_cbm':
        # NEW: 4-phase fairness-first curriculum (FairCurriculumCBM)
        if concept_names is None:
            concept_names = [f'concept_{i}' for i in range(num_concepts)]
        
        model = FairCurriculumCBM(
            num_concepts=num_concepts,
            backbone=backbone,
            num_groups=6,  # Fitzpatrick types I-VI
            fairness_lambda=fairness_lambda,
            adversarial_lambda_target=adversarial_lambda,
            concept_names=concept_names
        )
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model


def train_epoch_direct(model, dataloader, optimizer, device):
    """Training epoch for Direct classifier."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        images, binary_labels, _ = batch  # label_type='binary' returns 3 values
        images = images.to(device)
        binary_labels = binary_labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = nn.BCEWithLogitsLoss()(logits.squeeze(), binary_labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.sigmoid(logits).cpu().detach().numpy()
        all_preds.extend(preds)
        all_labels.extend(binary_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    
    return avg_loss, metrics


def train_epoch_standard_cbm(model, dataloader, optimizer, device):
    """Training epoch for Standard CBM (no curriculum)."""
    model.train()
    total_loss = 0
    total_concept_loss = 0
    total_binary_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        images, concept_labels, binary_labels, _ = batch  # Ignore fitzpatrick
        images = images.to(device)
        concept_labels = concept_labels.to(device)
        binary_labels = binary_labels.to(device)
        
        optimizer.zero_grad()
        concept_logits, binary_logits = model(images)
        
        # Standard CBM loss: concepts + binary
        concept_loss = nn.BCEWithLogitsLoss()(concept_logits, concept_labels)
        binary_loss = nn.BCEWithLogitsLoss()(binary_logits.squeeze(), binary_labels)
        loss = concept_loss + binary_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_concept_loss += concept_loss.item()
        total_binary_loss += binary_loss.item()
        
        preds = torch.sigmoid(binary_logits).cpu().detach().numpy()
        all_preds.extend(preds)
        all_labels.extend(binary_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_concept_loss = total_concept_loss / len(dataloader)
    avg_binary_loss = total_binary_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    
    return {
        'total_loss': avg_loss,
        'concept_loss': avg_concept_loss,
        'binary_loss': avg_binary_loss,
        'metrics': metrics
    }


def train_epoch_curriculum_cbm(model, dataloader, optimizer, device, epoch, max_epochs):
    """Training epoch for Curriculum CBM."""
    model.train()
    total_loss = 0
    total_concept_loss = 0
    total_binary_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        images, concept_labels, binary_labels, _ = batch  # Ignore fitzpatrick
        images = images.to(device)
        concept_labels = concept_labels.to(device)
        binary_labels = binary_labels.to(device)
        
        optimizer.zero_grad()
        concept_logits, binary_logits = model(images)
        
        # Curriculum loss with phase scheduling
        loss_dict = model.compute_loss(
            concept_logits, binary_logits,
            concept_labels, binary_labels.squeeze(),  # Squeeze to [batch_size]
            epoch, max_epochs
        )
        loss = loss_dict['total_loss']
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_concept_loss += loss_dict['concept_loss'].item()
        total_binary_loss += loss_dict['binary_loss'].item()
        
        preds = torch.sigmoid(binary_logits).cpu().detach().numpy()
        all_preds.extend(preds)
        all_labels.extend(binary_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_concept_loss = total_concept_loss / len(dataloader)
    avg_binary_loss = total_binary_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    
    return {
        'total_loss': avg_loss,
        'concept_loss': avg_concept_loss,
        'binary_loss': avg_binary_loss,
        'metrics': metrics
    }


def compute_per_group_f1(all_preds, all_labels, all_groups, num_groups=6):
    """Compute F1 score per demographic group."""
    group_f1_dict = {}
    
    for g in range(num_groups):
        mask = (all_groups == g)
        if mask.sum() > 0:
            g_metrics = compute_metrics(all_labels[mask], all_preds[mask])
            group_f1_dict[g] = g_metrics['f1']
        else:
            group_f1_dict[g] = 0.0
    
    return group_f1_dict


def train_epoch_fair_curriculum_cbm(model, dataloader, optimizer, device, epoch, max_epochs):
    """Training epoch for Fair Curriculum CBM (new 4-phase implementation)."""
    model.train()
    total_loss = 0
    total_concept_loss = 0
    total_binary_loss = 0
    total_fairness_loss = 0
    total_adversarial_loss = 0
    all_preds = []
    all_labels = []
    all_groups = []
    
    # Get phase info from new FairCurriculumCBM (only if using new model)
    is_new_model = hasattr(model, 'get_phase_info')
    
    if is_new_model:
        # New FairCurriculumCBM API
        phase_info = model.get_phase_info(epoch, max_epochs)
        
        # Log phase transitions
        if epoch == 0 or (epoch > 0 and model.get_phase_info(epoch-1, max_epochs)['phase_name'] != phase_info['phase_name']):
            print(f"\n=== Phase: {phase_info['phase_name']} ===")
            print(f"  Fairness focus: {phase_info['fairness_focus']}")
            print(f"  Adversarial active: {phase_info['adversarial_active']}")
            print(f"  Concepts: All {phase_info['num_concepts']} (joint training)")
        
        # Fair Curriculum CBM uses joint concept training (all concepts throughout)
        # No concept masking needed
        active_concept_indices = list(range(model.num_concepts))
    elif model.use_concept_curriculum:
        # Old FairnessAwareCBM API
        curriculum_info = model.get_curriculum_phase_info(epoch, max_epochs)
        active_concepts = curriculum_info['active_concepts']
        new_concepts = curriculum_info['new_concepts']
        
        # Log phase transitions
        if epoch == 0 or (epoch > 0 and model.get_curriculum_phase_info(epoch-1, max_epochs)['phase_idx'] != curriculum_info['phase_idx']):
            print(f"\n=== Curriculum Phase {curriculum_info['phase_idx']+1}: {curriculum_info['phase_name']} ===")
            print(f"Active concepts ({len(active_concepts)}): {active_concepts}")
            if new_concepts:
                print(f"New concepts ({len(new_concepts)}): {list(new_concepts)}")
        
        # Update adversarial alpha (gradient reversal strength)
        model.update_adversarial_alpha(epoch, max_epochs)
    else:
        # No curriculum - use all concepts
        active_concepts = []
        new_concepts = set()
    
    for batch in tqdm(dataloader, desc="Training"):
        images, concept_labels, binary_labels, fitzpatrick = batch
        images = images.to(device)
        concept_labels = concept_labels.to(device)
        binary_labels = binary_labels.to(device)
        
        # Convert Fitzpatrick from 1-6 to 0-5 for indexing
        group_labels = (fitzpatrick - 1).long().to(device)
        
        optimizer.zero_grad()
        concept_logits, binary_logits, features = model(images, return_features=True)
        
        # Compute loss based on model type
        if is_new_model:
            # New FairCurriculumCBM API
            # Fair Curriculum CBM uses joint concept training (all concepts)
            
            loss_dict = model.compute_loss(
                concept_logits=concept_logits,
                binary_logits=binary_logits,
                concept_labels=concept_labels,
                binary_labels=binary_labels,
                groups=group_labels,
                features=features,
                epoch=epoch,
                total_epochs=max_epochs
            )
            # New API returns 'total' not 'total_loss'
            loss = loss_dict['total']
            # Normalize keys for consistent handling
            loss_dict = {
                'total_loss': loss_dict['total'],
                'concept_loss': loss_dict['concept'],
                'binary_loss': loss_dict['binary'],
                'fairness_loss': loss_dict['fairness'],
                'adversarial_loss': loss_dict['adversarial']
            }
        else:
            # Old FairnessAwareCBM API
            # Compute warmup weight for new concepts (first 5 epochs of phase) if curriculum enabled
            if model.use_concept_curriculum:
                phase_epoch = epoch % (max_epochs // 3)
                new_concept_weight_multiplier = min(1.0, 0.1 + 0.18 * phase_epoch) if phase_epoch < 5 else 1.0
            else:
                new_concept_weight_multiplier = 1.0
            
            loss_dict = model.compute_fairness_loss(
                concept_logits=concept_logits,
                binary_logits=binary_logits,
                concept_features=features,
                concept_labels=concept_labels,
                binary_labels=binary_labels,
                group_labels=group_labels,
                active_concepts=active_concepts,
                new_concepts=new_concepts,
                new_concept_weight_multiplier=new_concept_weight_multiplier
            )
            loss = loss_dict['total_loss']
        
        loss.backward()
        
        # Clip gradients to prevent adversarial explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_concept_loss += loss_dict['concept_loss'].item()
        total_binary_loss += loss_dict['binary_loss'].item()
        total_fairness_loss += loss_dict.get('fairness_loss', torch.tensor(0.0)).item()
        total_adversarial_loss += loss_dict.get('adversarial_loss', torch.tensor(0.0)).item()
        
        preds = torch.sigmoid(binary_logits).cpu().detach().numpy()
        all_preds.extend(preds)
        all_labels.extend(binary_labels.cpu().numpy())
        all_groups.extend(group_labels.cpu().numpy())
    
    # Verify group distribution in epoch (debug checkpoint)
    if is_new_model and epoch % 10 == 0:
        all_groups_array = np.array(all_groups)
        group_counts = np.bincount(all_groups_array, minlength=6)
        print(f"  Epoch {epoch} group distribution: {group_counts.tolist()}")
    
    avg_loss = total_loss / len(dataloader)
    avg_concept_loss = total_concept_loss / len(dataloader)
    avg_binary_loss = total_binary_loss / len(dataloader)
    avg_fairness_loss = total_fairness_loss / len(dataloader)
    avg_adversarial_loss = total_adversarial_loss / len(dataloader)
    
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    
    # Get adversarial alpha (different API for new vs old model)
    if is_new_model:
        adv_alpha = model.current_adversarial_lambda
    else:
        adv_alpha = model.adversarial_alpha
    
    return {
        'total_loss': avg_loss,
        'concept_loss': avg_concept_loss,
        'binary_loss': avg_binary_loss,
        'fairness_loss': avg_fairness_loss,
        'adversarial_loss': avg_adversarial_loss,
        'adversarial_alpha': adv_alpha,
        'metrics': metrics
    }


def evaluate(model, dataloader, device, model_type, compute_fairness=True):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: torch.device
        model_type: Type of model
        compute_fairness: Whether to compute fairness metrics
    
    Returns:
        dict: Evaluation results with binary and concept metrics
    """
    model.eval()
    all_binary_preds = []
    all_binary_labels = []
    all_groups = []
    all_concept_preds = []
    all_concept_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Handle different batch formats based on label_type
            if model_type == 'direct':
                # label_type='binary': (image, label, fitzpatrick)
                images, binary_labels, fitzpatrick = batch
                concept_labels = None
            else:
                # label_type='concept': (image, concepts, label, fitzpatrick)
                images, concept_labels, binary_labels, fitzpatrick = batch
            
            images = images.to(device)
            
            if model_type == 'direct':
                logits = model(images)
                concept_logits = None
            else:
                concept_logits, binary_logits = model(images)
                logits = binary_logits
                
                # Collect concept predictions
                concept_probs = torch.sigmoid(concept_logits).cpu().numpy()
                all_concept_preds.append(concept_probs)
                all_concept_labels.append(concept_labels.numpy())
            
            preds = torch.sigmoid(logits).cpu().numpy().flatten()
            all_binary_preds.extend(preds)
            all_binary_labels.extend(binary_labels.numpy().flatten())
            all_groups.extend(fitzpatrick.numpy().flatten())
    
    all_binary_preds = np.array(all_binary_preds).flatten()
    all_binary_labels = np.array(all_binary_labels).flatten()
    all_groups = np.array(all_groups).flatten()
    
    # Binary classification metrics
    binary_metrics = compute_metrics(all_binary_labels, all_binary_preds)
    results = {'binary_metrics': binary_metrics}
    
    # Concept metrics (for CBM models only)
    if model_type != 'direct' and len(all_concept_preds) > 0:
        all_concept_preds = np.vstack(all_concept_preds)  # [n_samples, n_concepts]
        all_concept_labels = np.vstack(all_concept_labels)  # [n_samples, n_concepts]
        
        # Per-concept metrics
        num_concepts = all_concept_preds.shape[1]
        concept_metrics = {
            'per_concept_accuracy': [],
            'per_concept_f1': [],
            'per_concept_precision': [],
            'per_concept_recall': []
        }
        
        for c in range(num_concepts):
            c_preds = all_concept_preds[:, c]
            c_labels = all_concept_labels[:, c]
            c_metrics = compute_metrics(c_labels, c_preds)
            
            concept_metrics['per_concept_accuracy'].append(c_metrics['accuracy'])
            concept_metrics['per_concept_f1'].append(c_metrics['f1'])
            concept_metrics['per_concept_precision'].append(c_metrics['precision'])
            concept_metrics['per_concept_recall'].append(c_metrics['recall'])
        
        # Average concept metrics
        concept_metrics['avg_concept_accuracy'] = np.mean(concept_metrics['per_concept_accuracy'])
        concept_metrics['avg_concept_f1'] = np.mean(concept_metrics['per_concept_f1'])
        concept_metrics['avg_concept_precision'] = np.mean(concept_metrics['per_concept_precision'])
        concept_metrics['avg_concept_recall'] = np.mean(concept_metrics['per_concept_recall'])
        
        results['concept_metrics'] = concept_metrics
        
        # Concept-level fairness metrics (optional)
        if compute_fairness:
            group_indices = (all_groups - 1).astype(int)
            
            # Compute fairness for each concept
            concept_fairness = {
                'per_concept_demographic_parity': [],
                'per_concept_equalized_odds': []
            }
            
            for c in range(num_concepts):
                c_preds = all_concept_preds[:, c]
                c_labels = all_concept_labels[:, c]
                
                try:
                    c_fairness = compute_all_fairness_metrics(
                        predictions=c_preds,
                        labels=c_labels.astype(int),
                        groups=group_indices,
                        threshold=0.5
                    )
                    concept_fairness['per_concept_demographic_parity'].append(
                        c_fairness['demographic_parity']['max_disparity']
                    )
                    concept_fairness['per_concept_equalized_odds'].append(
                        c_fairness['equalized_odds']['max_disparity']
                    )
                except:
                    # Handle cases where fairness computation fails
                    concept_fairness['per_concept_demographic_parity'].append(np.nan)
                    concept_fairness['per_concept_equalized_odds'].append(np.nan)
            
            # Average concept fairness disparities
            concept_fairness['avg_concept_demographic_parity'] = np.nanmean(
                concept_fairness['per_concept_demographic_parity']
            )
            concept_fairness['avg_concept_equalized_odds'] = np.nanmean(
                concept_fairness['per_concept_equalized_odds']
            )
            
            results['concept_fairness'] = concept_fairness
    
    # Binary task fairness metrics
    if compute_fairness:
        # Convert Fitzpatrick from 1-6 to 0-5 for indexing
        group_indices = (all_groups - 1).astype(int)
        
        # compute_all_fairness_metrics expects probabilities as 'predictions' argument
        binary_fairness = compute_all_fairness_metrics(
            predictions=all_binary_preds,  # Pass probabilities
            labels=all_binary_labels.astype(int),
            groups=group_indices,
            threshold=0.5
        )
        results['binary_fairness'] = binary_fairness
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train models for fairness comparison')
    
    # Model configuration
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['direct', 'standard_cbm', 'fair_standard_cbm', 'curriculum_cbm', 'fair_curriculum_cbm'],
                        help='Type of model to train')
    parser.add_argument('--backbone', type=str, default='swin',
                        choices=['swin', 'convnext', 'vit', 'efficientnet', 'mobilenet'],
                        help='Backbone architecture')
    
    # Fairness hyperparameters (Fair CBM only)
    parser.add_argument('--fairness_lambda', type=float, default=0.1,
                        help='Weight for fairness loss (default: 0.1)')
    parser.add_argument('--adversarial_lambda', type=float, default=0.01,
                        help='Target weight for adversarial loss (default: 0.01)')
    parser.add_argument('--adversarial_warmup_epochs', type=int, default=None,
                        help='Epochs to warmup adversarial lambda (default: 30%% of total epochs)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Data configuration
    parser.add_argument('--data_root', type=str, default='data/skincap',
                        help='Root directory for dataset (or image directory)')
    parser.add_argument('--raw_csv', type=str, default=None,
                        help='Path to raw CSV file (will be split automatically)')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Training set proportion')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation set proportion')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Test set proportion')
    parser.add_argument('--concepts_path', type=str, default='data/skincap_concepts.txt',
                        help='Path to concept names file')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    # Experiment configuration
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    
    # Evaluation
    parser.add_argument('--eval_every', type=int, default=5,
                        help='Evaluate every N epochs')
    parser.add_argument('--save_best', action='store_true',
                        help='Save best model based on validation F1')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir) / args.exp_name / args.model_type
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load concept names
    concepts = load_concept_names(args.concepts_path)
    num_concepts = len(concepts)
    print(f"Loaded {num_concepts} concepts")
    
    # Create datasets
    print("Creating datasets...")
    if args.model_type == 'direct':
        label_type = 'binary'  # Direct classifier doesn't need concepts
    else:
        label_type = 'concept'  # CBM models need concepts
    
    # Common dataset kwargs
    dataset_kwargs = {
        'root_dir': args.data_root,
        'label_type': label_type,
        'seed': args.seed
    }
    
    # Add raw_csv if provided
    if args.raw_csv:
        dataset_kwargs.update({
            'raw_csv': args.raw_csv,
            'train_split': args.train_split,
            'val_split': args.val_split,
            'test_split': args.test_split
        })
    
    train_dataset = SkinCapDataset(split='train', **dataset_kwargs)
    val_dataset = SkinCapDataset(split='val', **dataset_kwargs)
    test_dataset = SkinCapDataset(split='test', **dataset_kwargs)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    print(f"Creating {args.model_type} model with {args.backbone} backbone...")
    model = create_model(
        model_type=args.model_type,
        num_concepts=num_concepts,
        backbone=args.backbone,
        fairness_lambda=args.fairness_lambda,
        adversarial_lambda=args.adversarial_lambda,
        concept_names=concepts
    )
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    # Setup adversarial lambda warmup schedule (for both fair_standard_cbm and fair_curriculum_cbm)
    if args.model_type in ['fair_standard_cbm', 'fair_curriculum_cbm']:
        warmup_epochs = args.adversarial_warmup_epochs or int(0.3 * args.epochs)
        warmup_start = int(0.2 * args.epochs)  # Start warmup after 20% of training
        print(f"Adversarial warmup: epochs {warmup_start+1}-{warmup_start + warmup_epochs} (0 → {args.adversarial_lambda})")
        if args.model_type == 'fair_curriculum_cbm':
            print(f"Concept curriculum: 3-phase staging enabled")
        else:
            print(f"Concept curriculum: disabled (all concepts trained jointly)")
    
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_f1 = 0.0
    best_epoch = -1
    global_best_saved = False
    # Global best path at results/ level (shared across ALL experiments)
    # save_dir = results/multi_run_XXX/run_Y/model_type
    # -> parent.parent.parent = results/
    global_best_path = save_dir.parent.parent.parent / 'best-models.json'
    history = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for epoch in range(args.epochs):
        # Compute current adversarial lambda (warmup schedule)
        if args.model_type in ['fair_standard_cbm', 'fair_curriculum_cbm']:
            if epoch < warmup_start:
                # Phase 1: No fairness (learn primary task)
                current_adv_lambda = 0.0
            elif epoch < warmup_start + warmup_epochs:
                # Phase 2: Linear warmup
                progress = (epoch - warmup_start) / warmup_epochs
                current_adv_lambda = args.adversarial_lambda * progress
            else:
                # Phase 3: Full fairness
                current_adv_lambda = args.adversarial_lambda
            
            # Update model's adversarial lambda
            model.adversarial_lambda = current_adv_lambda
        else:
            current_adv_lambda = 0.0  # Not used for other models
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training
        if args.model_type == 'direct':
            train_loss, train_metrics = train_epoch_direct(model, train_loader, optimizer, device)
            print(f"Train Loss: {train_loss:.4f}")
            history['train'].append({'epoch': epoch+1, 'loss': train_loss, 'metrics': train_metrics})
        
        elif args.model_type == 'standard_cbm':
            train_results = train_epoch_standard_cbm(model, train_loader, optimizer, device)
            print(f"Train Loss: {train_results['total_loss']:.4f} "
                  f"(Concept: {train_results['concept_loss']:.4f}, "
                  f"Binary: {train_results['binary_loss']:.4f})")
            history['train'].append({'epoch': epoch+1, **train_results})
        
        elif args.model_type == 'curriculum_cbm':
            train_results = train_epoch_curriculum_cbm(model, train_loader, optimizer, device, epoch, args.epochs)
            print(f"Train Loss: {train_results['total_loss']:.4f} "
                  f"(Concept: {train_results['concept_loss']:.4f}, "
                  f"Binary: {train_results['binary_loss']:.4f})")
            history['train'].append({'epoch': epoch+1, **train_results})
        
        elif args.model_type in ['fair_standard_cbm', 'fair_curriculum_cbm']:
            # Recreate DataLoader with phase-aware sampler for fair_curriculum_cbm
            if args.model_type == 'fair_curriculum_cbm' and hasattr(model, 'get_phase_info'):
                # Get group labels and binary labels from dataset
                # Access directly from dataset attributes if available, otherwise extract
                if hasattr(train_dataset, 'fitzpatrick') and hasattr(train_dataset, 'labels'):
                    # Direct access to dataset arrays
                    all_groups = torch.tensor(train_dataset.fitzpatrick - 1, dtype=torch.long)  # Convert 1-6 to 0-5
                    all_labels = torch.tensor(train_dataset.labels, dtype=torch.float)
                else:
                    # Fallback: extract from samples (slower)
                    all_groups = []
                    all_labels = []
                    for i in range(len(train_dataset)):
                        sample = train_dataset[i]
                        all_labels.append(sample[2].item() if torch.is_tensor(sample[2]) else sample[2])
                        all_groups.append((sample[3].item() if torch.is_tensor(sample[3]) else sample[3]) - 1)
                    all_groups = torch.tensor(all_groups, dtype=torch.long)
                    all_labels = torch.tensor(all_labels, dtype=torch.float)
                
                # Create phase-aware sampler
                sampler = FairnessAwareSampler(
                    groups=all_groups,
                    labels=all_labels,
                    batch_size=args.batch_size,
                    epoch=epoch,
                    total_epochs=args.epochs,
                    group_f1_scores=model.group_f1_scores
                )
                
                # Recreate DataLoader with new sampler
                train_loader_epoch = DataLoader(
                    train_dataset,
                    batch_sampler=sampler,
                    num_workers=args.num_workers,
                    pin_memory=True
                )
            else:
                train_loader_epoch = train_loader
            
            train_results = train_epoch_fair_curriculum_cbm(model, train_loader_epoch, optimizer, device, epoch, args.epochs)
            print(f"Train Loss: {train_results['total_loss']:.4f} "
                  f"(Concept: {train_results['concept_loss']:.4f}, "
                  f"Binary: {train_results['binary_loss']:.4f}, "
                  f"Fairness: {train_results['fairness_loss']:.4f}, "
                  f"Adversarial: {train_results['adversarial_loss']:.4f}, "
                  f"Lambda: {current_adv_lambda:.4f}, "
                  f"Alpha: {train_results['adversarial_alpha']:.4f})")
            history['train'].append({'epoch': epoch+1, 'adversarial_lambda': current_adv_lambda, **train_results})
        
        # Validation
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            print("\nValidation:")
            val_results = evaluate(model, val_loader, device, args.model_type, compute_fairness=True)
            val_f1 = val_results['binary_metrics']['f1']
            print(f"Val Binary F1: {val_f1:.4f}")
            
            # Concept metrics (if available)
            if 'concept_metrics' in val_results:
                cm = val_results['concept_metrics']
                print(f"Val Concept Avg Accuracy: {cm['avg_concept_accuracy']:.4f}, "
                      f"Avg F1: {cm['avg_concept_f1']:.4f}")
            
            # Binary fairness
            if 'binary_fairness' in val_results:
                fm = val_results['binary_fairness']
                print(f"Val Binary Fairness - Performance Gap: {fm['worst_group']['performance_gap']:.4f}, "
                      f"Demographic Parity: {fm['demographic_parity']['max_disparity']:.4f}")
            
            # Concept fairness (if available)
            if 'concept_fairness' in val_results:
                cf = val_results['concept_fairness']
                print(f"Val Concept Fairness - Avg DP: {cf['avg_concept_demographic_parity']:.4f}, "
                      f"Avg EO: {cf['avg_concept_equalized_odds']:.4f}")
            
            history['val'].append({'epoch': epoch+1, **val_results})
            
            # Update group F1 scores every 5 epochs for error-driven sampling (Phase 4)
            if args.model_type == 'fair_curriculum_cbm' and hasattr(model, 'update_group_f1_scores') and (epoch + 1) % 5 == 0:
                # Compute per-group F1 from validation results
                # Re-evaluate to get per-sample predictions with groups
                model.eval()
                val_preds_list = []
                val_labels_list = []
                val_groups_list = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        images, _, binary_labels, fitzpatrick = batch
                        images = images.to(device)
                        
                        concept_logits, binary_logits = model(images)
                        preds = torch.sigmoid(binary_logits).cpu().numpy().flatten()
                        
                        val_preds_list.extend(preds)
                        val_labels_list.extend(binary_labels.numpy().flatten())
                        val_groups_list.extend((fitzpatrick - 1).numpy().flatten())  # Convert to 0-5
                
                val_preds_array = np.array(val_preds_list)
                val_labels_array = np.array(val_labels_list)
                val_groups_array = np.array(val_groups_list, dtype=int)
                
                group_f1_dict = compute_per_group_f1(val_preds_array, val_labels_array, val_groups_array)
                model.update_group_f1_scores(group_f1_dict)
                
                print(f"  Updated group F1 scores:")
                for g, f1 in sorted(group_f1_dict.items()):
                    print(f"    Fitzpatrick Type {g+1}: F1 = {f1:.4f}")
            
            # Check if this is globally best for this model type
            # For fairness models, extract performance gap
            performance_gap = None
            if 'binary_fairness' in val_results and 'worst_group' in val_results['binary_fairness']:
                performance_gap = val_results['binary_fairness']['worst_group']['performance_gap']
            
            # Compute composite metric for fair models
            is_fairness_model = args.model_type in ['fair_curriculum_cbm', 'fair_standard_cbm']
            if is_fairness_model and performance_gap is not None:
                composite_metric = val_f1 * (1.0 - performance_gap)
            else:
                composite_metric = val_f1
            
            # Track best based on composite metric
            if composite_metric > best_val_f1:
                best_val_f1 = composite_metric  # Store composite for fair models
                best_epoch = epoch
                
                # Only save if globally best across all experiments
                is_saved = check_and_save_global_best(
                    model_type=args.model_type,
                    val_f1=val_f1,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    save_dir=save_dir.parent.parent.parent,  # Save to results/ level
                    global_best_path=global_best_path,
                    performance_gap=performance_gap
                )
                
                if is_saved:
                    global_best_saved = True
                    if is_fairness_model and performance_gap is not None:
                        print(f"Saved global best model (composite: {composite_metric:.4f}, F1: {val_f1:.4f}, gap: {performance_gap:.4f})")
                    else:
                        print(f"Saved global best model (F1: {val_f1:.4f})")
    
    # Final test evaluation
    print("\nFinal Test Evaluation:")
    # Load global best model if this run produced it
    global_best_model_path = save_dir.parent.parent.parent / f'global_best_{args.model_type}.pt'
    if global_best_saved and global_best_model_path.exists():
        checkpoint = torch.load(global_best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Display composite metric for fairness models
        is_fairness_model = args.model_type in ['fair_curriculum_cbm', 'fair_standard_cbm']
        if is_fairness_model and 'composite_score' in checkpoint and 'performance_gap' in checkpoint:
            print(f"Loaded global best model from epoch {checkpoint['epoch']+1} "
                  f"(composite: {checkpoint['composite_score']:.4f}, "
                  f"F1: {checkpoint['val_f1']:.4f}, gap: {checkpoint['performance_gap']:.4f})")
        else:
            print(f"Loaded global best model from epoch {checkpoint['epoch']+1} (F1: {checkpoint['val_f1']:.4f})")
    elif best_epoch >= 0:
        # Display what metric was used for selection
        is_fairness_model = args.model_type in ['fair_curriculum_cbm', 'fair_standard_cbm']
        if is_fairness_model:
            print(f"Using model from best epoch {best_epoch+1} (composite: {best_val_f1:.4f}) - not globally best")
        else:
            print(f"Using model from best epoch {best_epoch+1} (F1: {best_val_f1:.4f}) - not globally best")
    
    test_results = evaluate(model, test_loader, device, args.model_type, compute_fairness=True)
    test_f1 = test_results['binary_metrics']['f1']
    print(f"\nTest Binary Classification Metrics:")
    print(f"  F1: {test_f1:.4f}")
    print(f"  Accuracy: {test_results['binary_metrics']['accuracy']:.4f}")
    print(f"  Precision: {test_results['binary_metrics']['precision']:.4f}")
    print(f"  Recall: {test_results['binary_metrics']['recall']:.4f}")
    
    # Concept metrics
    if 'concept_metrics' in test_results:
        cm = test_results['concept_metrics']
        print(f"\nTest Concept Prediction Metrics:")
        print(f"  Avg Accuracy: {cm['avg_concept_accuracy']:.4f}")
        print(f"  Avg F1: {cm['avg_concept_f1']:.4f}")
        print(f"  Avg Precision: {cm['avg_concept_precision']:.4f}")
        print(f"  Avg Recall: {cm['avg_concept_recall']:.4f}")
    
    # Binary fairness metrics
    if 'binary_fairness' in test_results:
        fm = test_results['binary_fairness']
        print(f"\nTest Binary Fairness Metrics:")
        
        # Performance gap
        if 'performance_gap' in fm:
            print(f"  Performance Gap: {fm['performance_gap']['gap']:.4f}")
            if 'group_f1' in fm['performance_gap']:
                group_f1s = list(fm['performance_gap']['group_f1'].values())
                print(f"  Worst-group F1: {min(group_f1s):.4f}, Best-group F1: {max(group_f1s):.4f}")
        
        # Demographic parity
        if 'demographic_parity' in fm:
            print(f"  Demographic Parity Disparity: {fm['demographic_parity']['max_disparity']:.4f}")
        
        # Equalized odds
        if 'equalized_odds' in fm:
            print(f"  Equalized Odds Disparity: {fm['equalized_odds']['max_disparity']:.4f}")
    
    # Concept fairness metrics
    if 'concept_fairness' in test_results:
        cf = test_results['concept_fairness']
        print(f"\nTest Concept Fairness Metrics:")
        print(f"  Avg Demographic Parity Disparity: {cf['avg_concept_demographic_parity']:.4f}")
        print(f"  Avg Equalized Odds Disparity: {cf['avg_concept_equalized_odds']:.4f}")
    
    history['test'].append({'epoch': args.epochs, **test_results})
    
    # Save history (but not final model - only global best is saved)
    history['global_best_saved'] = global_best_saved
    history['best_val_f1'] = best_val_f1
    history['best_epoch'] = best_epoch
    
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    if global_best_saved:
        print(f"\nTraining complete! Global best model saved (F1: {best_val_f1:.4f})")
    else:
        print(f"\nTraining complete! Best F1: {best_val_f1:.4f} (not globally best)")
    print(f"History saved to {save_dir / 'history.json'}")


if __name__ == '__main__':
    main()
