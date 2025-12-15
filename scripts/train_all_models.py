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
from datetime import datetime
from tqdm import tqdm

# Import FairCBM components (all local)
from src.data.dataloader import SkinCapDataset
from src.models.direct_classifier import DirectClassifier
from src.models.standard_cbm import StandardCBM
from src.models.minimal_curriculum_cbm import MinimalCurriculumCBM
from src.models.fairness_aware_cbm import FairnessAwareCBM
from src.utils.metrics import compute_metrics
from src.utils.fairness_metrics import compute_all_fairness_metrics


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
        model_type: 'direct', 'standard_cbm', 'curriculum_cbm', 'fair_curriculum_cbm'
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
    
    elif model_type == 'fair_cbm':
        # Fairness-aware CBM WITHOUT concept curriculum (only adversarial warmup)
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
        # Fairness-aware CBM WITH concept curriculum (3-phase concept staging + adversarial warmup)
        if concept_names is None:
            concept_names = [f'concept_{i}' for i in range(num_concepts)]
        
        model = FairnessAwareCBM(
            concept_names=concept_names,
            backbone=backbone,
            num_groups=6,  # Fitzpatrick types I-VI
            fairness_lambda=fairness_lambda,
            adversarial_lambda=adversarial_lambda,
            use_concept_curriculum=True  # Enable 3-phase concept curriculum
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
            concept_logits, concept_labels,
            binary_logits, binary_labels,
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


def train_epoch_fair_curriculum_cbm(model, dataloader, optimizer, device, epoch, max_epochs):
    """Training epoch for Fair Curriculum CBM."""
    model.train()
    total_loss = 0
    total_concept_loss = 0
    total_binary_loss = 0
    total_fairness_loss = 0
    total_adversarial_loss = 0
    all_preds = []
    all_labels = []
    all_groups = []
    
    # Get curriculum phase info if curriculum is enabled
    if model.use_concept_curriculum:
        curriculum_info = model.get_curriculum_phase_info(epoch, max_epochs)
        active_concepts = curriculum_info['active_concepts']
        new_concepts = curriculum_info['new_concepts']
        
        # Log phase transitions
        if epoch == 0 or (epoch > 0 and model.get_curriculum_phase_info(epoch-1, max_epochs)['phase_idx'] != curriculum_info['phase_idx']):
            print(f"\n=== Curriculum Phase {curriculum_info['phase_idx']+1}: {curriculum_info['phase_name']} ===")
            print(f"Active concepts ({len(active_concepts)}): {active_concepts}")
            if new_concepts:
                print(f"New concepts ({len(new_concepts)}): {list(new_concepts)}")
    else:
        # No curriculum - use all concepts
        active_concepts = []
        new_concepts = set()
    
    # Update adversarial alpha (gradient reversal strength)
    model.update_adversarial_alpha(epoch, max_epochs)
    
    for batch in tqdm(dataloader, desc="Training"):
        images, concept_labels, binary_labels, fitzpatrick = batch
        images = images.to(device)
        concept_labels = concept_labels.to(device)
        binary_labels = binary_labels.to(device)
        
        # Convert Fitzpatrick from 1-6 to 0-5 for indexing
        group_labels = (fitzpatrick - 1).long().to(device)
        
        optimizer.zero_grad()
        concept_logits, binary_logits, concept_features = model(images, return_features=True)
        
        # Fairness-aware loss with optional concept curriculum masking
        # Compute warmup weight for new concepts (first 5 epochs of phase) if curriculum enabled
        if model.use_concept_curriculum:
            phase_epoch = epoch % (max_epochs // 3)
            new_concept_weight_multiplier = min(1.0, 0.1 + 0.18 * phase_epoch) if phase_epoch < 5 else 1.0
        else:
            new_concept_weight_multiplier = 1.0
        
        loss_dict = model.compute_fairness_loss(
            concept_logits=concept_logits,
            binary_logits=binary_logits,
            concept_features=concept_features,
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
    
    avg_loss = total_loss / len(dataloader)
    avg_concept_loss = total_concept_loss / len(dataloader)
    avg_binary_loss = total_binary_loss / len(dataloader)
    avg_fairness_loss = total_fairness_loss / len(dataloader)
    avg_adversarial_loss = total_adversarial_loss / len(dataloader)
    
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    
    return {
        'total_loss': avg_loss,
        'concept_loss': avg_concept_loss,
        'binary_loss': avg_binary_loss,
        'fairness_loss': avg_fairness_loss,
        'adversarial_loss': avg_adversarial_loss,
        'adversarial_alpha': model.adversarial_alpha,
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
        dict: Evaluation results
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_groups = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Handle different batch formats based on label_type
            if model_type == 'direct':
                # label_type='binary': (image, label, fitzpatrick)
                images, binary_labels, fitzpatrick = batch
            else:
                # label_type='concept': (image, concepts, label, fitzpatrick)
                images, _, binary_labels, fitzpatrick = batch
            images = images.to(device)
            
            if model_type == 'direct':
                logits = model(images)
            else:
                concept_logits, binary_logits = model(images)
                logits = binary_logits
            
            preds = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(binary_labels.numpy().flatten())
            all_groups.extend(fitzpatrick.numpy().flatten())
    
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    all_groups = np.array(all_groups).flatten()
    
    # Standard metrics
    metrics = compute_metrics(all_labels, all_preds)
    results = {'standard_metrics': metrics}
    
    # Fairness metrics
    if compute_fairness:
        # Convert Fitzpatrick from 1-6 to 0-5 for indexing
        group_indices = (all_groups - 1).astype(int)
        
        # compute_all_fairness_metrics expects probabilities as 'predictions' argument
        fairness_metrics = compute_all_fairness_metrics(
            predictions=all_preds,  # Pass probabilities
            labels=all_labels.astype(int),
            groups=group_indices,
            threshold=0.5
        )
        results['fairness_metrics'] = fairness_metrics
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train models for fairness comparison')
    
    # Model configuration
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['direct', 'standard_cbm', 'curriculum_cbm', 'fair_cbm', 'fair_curriculum_cbm'],
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
    # Setup adversarial lambda warmup schedule (for both fair_cbm and fair_curriculum_cbm)
    if args.model_type in ['fair_cbm', 'fair_curriculum_cbm']:
        warmup_epochs = args.adversarial_warmup_epochs or int(0.3 * args.epochs)
        warmup_start = int(0.2 * args.epochs)  # Start warmup after 20% of training
        print(f"Adversarial warmup: epochs {warmup_start+1}-{warmup_start + warmup_epochs} (0 → {args.adversarial_lambda})")
        if args.model_type == 'fair_curriculum_cbm':
            print(f"Concept curriculum: 3-phase staging enabled")
        else:
            print(f"Concept curriculum: disabled (all concepts trained jointly)")
    
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_f1 = 0.0
    history = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for epoch in range(args.epochs):
        # Compute current adversarial lambda (warmup schedule)
        if args.model_type in ['fair_cbm', 'fair_curriculum_cbm']:
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
        
        elif args.model_type in ['fair_cbm', 'fair_curriculum_cbm']:
            train_results = train_epoch_fair_curriculum_cbm(model, train_loader, optimizer, device, epoch, args.epochs)
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
            val_f1 = val_results['standard_metrics']['f1']
            print(f"Val F1: {val_f1:.4f}")
            
            if 'fairness_metrics' in val_results:
                fm = val_results['fairness_metrics']
                print(f"Val Fairness - Performance Gap: {fm['worst_group']['performance_gap']:.4f}, "
                      f"Demographic Parity: {fm['demographic_parity']['max_disparity']:.4f}")
            
            history['val'].append({'epoch': epoch+1, **val_results})
            
            # Save best model (always save best for multi-run selection)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_f1,
                    'config': config
                }, save_dir / 'best_model.pt')
                if args.save_best:
                    print(f"Saved best model (F1: {val_f1:.4f})")
    
    # Final test evaluation
    print("\nFinal Test Evaluation:")
    # Always load best model for evaluation (best_model.pt always saved now)
    if (save_dir / 'best_model.pt').exists():
        checkpoint = torch.load(save_dir / 'best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    test_results = evaluate(model, test_loader, device, args.model_type, compute_fairness=True)
    test_f1 = test_results['standard_metrics']['f1']
    print(f"Test F1: {test_f1:.4f}")
    
    if 'fairness_metrics' in test_results:
        fm = test_results['fairness_metrics']
        print(f"Test Fairness Metrics:")
        
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
    
    history['test'].append({'epoch': args.epochs, **test_results})
    
    # Save final model and history
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_f1': test_f1,
        'config': config
    }, save_dir / 'final_model.pt')
    
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    print(f"\nTraining complete! Results saved to {save_dir}")


if __name__ == '__main__':
    main()
