"""
Ablation Study Training Script

Trains Fair Curriculum CBM with specific phases ablated to measure their contributions.

Usage:
    python train_ablation.py --ablation_key full_model --exp_name ablation_001
    python train_ablation.py --ablation_key no_phase1 --exp_name ablation_001
    python train_ablation.py --ablation_key no_phase2 --exp_name ablation_001
    python train_ablation.py --ablation_key no_phase3 --exp_name ablation_001
    python train_ablation.py --ablation_key no_phase4 --exp_name ablation_001

Author: Matt Cockayne
Date: January 2026
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
import shutil
from datetime import datetime
from tqdm import tqdm

# Import FairCBM components
from src.data.dataloader import SkinCapDataset
from src.models.fair_curriculum_cbm import FairCurriculumCBM, FairnessAwareSampler
from src.configs.ablation_configs import get_ablation_config, list_ablation_configs
from src.utils.metrics import compute_metrics
from src.utils.fairness_metrics import compute_all_fairness_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train Fair Curriculum CBM with Ablation')
    
    # Ablation configuration
    parser.add_argument('--ablation_key', type=str, required=True,
                       choices=list_ablation_configs(),
                       help='Ablation configuration key')
    
    # Experiment configuration
    parser.add_argument('--exp_name', type=str, required=True,
                       help='Experiment name for organizing results')
    parser.add_argument('--run_id', type=int, default=0,
                       help='Run ID for multi-run experiments')
    
    # Model configuration
    parser.add_argument('--backbone', type=str, default='swin',
                       help='Backbone model name')
    parser.add_argument('--num_concepts', type=int, default=23,
                       help='Number of concepts (SkinCap default: 23)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--fairness_lambda', type=float, default=1.0,
                       help='Weight for fairness loss')
    parser.add_argument('--adversarial_lambda', type=float, default=0.1,
                       help='Weight for adversarial loss')
    
    # Data configuration
    parser.add_argument('--data_root', type=str, default='/home/csc29/projects/SkinCAP',
                       help='Root directory for dataset (or image directory)')
    parser.add_argument('--raw_csv', type=str, default='/home/csc29/projects/SkinCAP/skincap_v240623.csv',
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
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='results/ablation',
                       help='Directory for saving results')
    parser.add_argument('--save_checkpoints', action='store_true',
                       help='Save model checkpoints')
    parser.add_argument('--checkpoint_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def create_ablation_model(ablation_config, backbone, num_concepts, fairness_lambda, 
                          adversarial_lambda, device):
    """
    Create a Fair Curriculum CBM model modified according to ablation config.
    
    Args:
        ablation_config: AblationConfig instance
        backbone: Backbone model name
        num_concepts: Number of concepts
        fairness_lambda: Weight for fairness loss
        adversarial_lambda: Weight for adversarial loss
        device: Torch device
        
    Returns:
        Modified FairCurriculumCBM model
    """
    concept_names = [f'concept_{i}' for i in range(num_concepts)]
    
    # Create base Fair Curriculum CBM model
    model = FairCurriculumCBM(
        num_concepts=num_concepts,
        backbone=backbone,
        num_groups=6,  # Fitzpatrick I-VI
        fairness_lambda=fairness_lambda,
        adversarial_lambda_target=adversarial_lambda,
        concept_names=concept_names
    ).to(device)
    
    # Store ablation config for use during training
    model.ablation_config = ablation_config
    
    return model


def apply_ablation_to_model(model, ablation_config, epoch, max_epochs):
    """
    Modify model's phase behavior based on ablation configuration.
    
    This temporarily overrides the model's phase detection to skip disabled phases.
    """
    # Store original method
    if not hasattr(model, '_original_get_phase_info'):
        model._original_get_phase_info = model.get_phase_info
    
    # Override get_phase_info to skip disabled phases
    def ablated_get_phase_info(ep, total_ep):
        """Modified phase info that skips disabled phases."""
        info = model._original_get_phase_info(ep, total_ep)
        
        # Get current phase progress
        prog = ep / total_ep
        
        if prog <= 0.25 and not ablation_config.use_phase1_balanced:
            # Phase 1 disabled: no fairness, no special sampling
            info['use_fairness_loss'] = False
            info['adversarial_active'] = False
        elif prog <= 0.50 and not ablation_config.use_phase2_dp:
            # Phase 2 disabled: continue with Phase 1 behavior (or baseline)
            if ablation_config.use_phase1_balanced:
                info['use_fairness_loss'] = False
            else:
                info['use_fairness_loss'] = False
            info['adversarial_active'] = False
        elif prog <= 0.75 and not ablation_config.use_phase3_eo:
            # Phase 3 disabled: continue with Phase 2 behavior (or baseline)
            if ablation_config.use_phase2_dp:
                info['fairness_focus'] = 'Demographic Parity (Phase 3 disabled)'
            else:
                info['use_fairness_loss'] = False
            info['adversarial_active'] = False
        elif prog > 0.75 and not ablation_config.use_phase4_error:
            # Phase 4 disabled: continue with Phase 3 behavior
            if ablation_config.use_phase3_eo:
                info['fairness_focus'] = 'Equalized Odds (Phase 4 disabled)'
                info['adversarial_active'] = ablation_config.use_adversarial
            elif ablation_config.use_phase2_dp:
                info['fairness_focus'] = 'Demographic Parity (Phases 3-4 disabled)'
                info['use_fairness_loss'] = True
                info['adversarial_active'] = False
            else:
                info['use_fairness_loss'] = False
                info['adversarial_active'] = False
        
        # Disable adversarial if ablation says so
        if not ablation_config.use_adversarial:
            info['adversarial_active'] = False
        
        return info
    
    model.get_phase_info = ablated_get_phase_info


def train_epoch_with_ablation(model, dataloader, optimizer, device, epoch, max_epochs):
    """
    Training epoch for ablated Fair Curriculum CBM.
    
    Uses model's native compute_loss() method with ablation modifications.
    """
    model.train()
    ablation_config = model.ablation_config
    
    # Apply ablation to model's phase behavior
    apply_ablation_to_model(model, ablation_config, epoch, max_epochs)
    
    # Get phase info from model (now ablation-aware)
    phase_info = model.get_phase_info(epoch, max_epochs)
    
    total_loss = 0
    total_concept_loss = 0
    total_binary_loss = 0
    total_fairness_loss = 0
    total_adversarial_loss = 0
    
    all_preds = []
    all_labels = []
    all_groups = []
    
    for batch in tqdm(dataloader, desc=f"Training ({phase_info['phase_name']})"):
        images, concept_labels, binary_labels, fitzpatrick = batch
        images = images.to(device)
        concept_labels = concept_labels.to(device)
        binary_labels = binary_labels.to(device)
        # Fitzpatrick is 1-6 from dataset, convert to 0-5
        group_labels = (fitzpatrick - 1).long().to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with features for adversarial
        concept_logits, binary_logits, features = model(images, return_features=True)
        
        # Use model's native compute_loss method
        loss_dict = model.compute_loss(
            concept_logits=concept_logits,
            binary_logits=binary_logits,
            concept_labels=concept_labels,
            binary_labels=binary_labels,
            groups=group_labels,  # Parameter is 'groups' not 'group_labels'
            features=features,
            epoch=epoch,
            total_epochs=max_epochs
        )
        
        loss = loss_dict['total']
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_concept_loss += loss_dict['concept'].item()
        total_binary_loss += loss_dict['binary'].item()
        total_fairness_loss += loss_dict.get('fairness', torch.tensor(0.0)).item()
        total_adversarial_loss += loss_dict.get('adversarial', torch.tensor(0.0)).item()
        
        # Collect predictions for metrics
        preds = torch.sigmoid(binary_logits).cpu().detach().numpy()
        all_preds.extend(preds)
        all_labels.extend(binary_labels.cpu().numpy())
        all_groups.extend(group_labels.cpu().numpy())
    
    # Compute metrics
    num_batches = len(dataloader)
    metrics = {
        'total_loss': total_loss / num_batches,
        'concept_loss': total_concept_loss / num_batches,
        'binary_loss': total_binary_loss / num_batches,
        'fairness_loss': total_fairness_loss / num_batches,
        'adversarial_loss': total_adversarial_loss / num_batches,
        'performance': compute_metrics(np.array(all_labels), np.array(all_preds)),
        'phase_info': phase_info
    }
    
    return metrics


def evaluate_with_fairness(model, dataloader, device):
    """Evaluate model with both performance and fairness metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_groups = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images, concept_labels, binary_labels, fitzpatrick = batch
            images = images.to(device)
            # Convert Fitzpatrick from 1-6 to 0-5 for consistency with training
            fitzpatrick = fitzpatrick - 1
            
            _, binary_logits = model(images)
            preds = torch.sigmoid(binary_logits).cpu().numpy().flatten()
            
            all_preds.extend(preds)
            all_labels.extend(binary_labels.numpy().flatten())
            all_groups.extend(fitzpatrick.numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_groups = np.array(all_groups)
    
    # Performance metrics
    perf_metrics = compute_metrics(all_labels, all_preds)
    
    # Fairness metrics
    fairness_metrics = compute_all_fairness_metrics(
        predictions=all_preds,
        labels=all_labels.astype(int),
        groups=all_groups,
        threshold=0.5
    )
    
    return {
        'performance': perf_metrics,
        'fairness': fairness_metrics
    }




# Removed manual fairness loss functions - using model's native compute_loss() instead


def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get ablation configuration
    ablation_config = get_ablation_config(args.ablation_key)
    print(f"\nAblation Configuration: {ablation_config.name}")
    print(f"  Key: {ablation_config.key}")
    print(f"  Phase 1 (Balanced Foundation): {'ENABLED' if ablation_config.use_phase1_balanced else 'DISABLED'}")
    print(f"  Phase 2 (Demographic Parity): {'ENABLED' if ablation_config.use_phase2_dp else 'DISABLED'}")
    print(f"  Phase 3 (Equalized Odds): {'ENABLED' if ablation_config.use_phase3_eo else 'DISABLED'}")
    print(f"  Phase 4 (Error-Driven): {'ENABLED' if ablation_config.use_phase4_error else 'DISABLED'}")
    print(f"  Adversarial Debiasing: {'ENABLED' if ablation_config.use_adversarial else 'DISABLED'}")
    print(f"\n  Active Phases: ", end='')
    active_phases = []
    if ablation_config.use_phase1_balanced:
        active_phases.append('1')
    if ablation_config.use_phase2_dp:
        active_phases.append('2')
    if ablation_config.use_phase3_eo:
        active_phases.append('3')
    if ablation_config.use_phase4_error:
        active_phases.append('4')
    print(', '.join(active_phases) if active_phases else 'NONE (baseline training only)')
    print()
    
    # Create output directory (for results.json only, not models)
    output_dir = Path(args.output_dir) / args.exp_name / ablation_config.key / f"run_{args.run_id:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create best_models directory (shared across all runs)
    best_models_dir = Path(args.output_dir) / args.exp_name / 'best_models'
    best_models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ablation configuration
    with open(output_dir / 'ablation_config.json', 'w') as f:
        json.dump({
            'ablation_key': args.ablation_key,
            'ablation_name': ablation_config.name,
            'phase1_balanced': ablation_config.use_phase1_balanced,
            'phase2_dp': ablation_config.use_phase2_dp,
            'phase3_eo': ablation_config.use_phase3_eo,
            'phase4_error': ablation_config.use_phase4_error,
            'adversarial': ablation_config.use_adversarial,
            'exp_name': args.exp_name,
            'run_id': args.run_id,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    # Load concept names
    concepts_path = Path(args.concepts_path)
    if not concepts_path.exists():
        # Fallback to celeba if skincap not found
        concepts_path = Path('data/concepts.txt')
    
    with open(concepts_path, 'r') as f:
        concepts = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(concepts)} concepts")
    
    # Load datasets
    print("Creating datasets...")
    
    # Common dataset kwargs
    dataset_kwargs = {
        'root_dir': args.data_root,
        'label_type': 'concept',  # Need all: concepts, binary, fitzpatrick
        'seed': args.seed
    }
    
    # Add raw_csv and split parameters if provided
    if args.raw_csv:
        dataset_kwargs.update({
            'raw_csv': args.raw_csv,
            'train_split': args.train_split,
            'val_split': args.val_split,
            'test_split': args.test_split
        })
    
    # Create datasets with splits
    train_dataset = SkinCapDataset(split='train', **dataset_kwargs)
    val_dataset = SkinCapDataset(split='val', **dataset_kwargs)
    test_dataset = SkinCapDataset(split='test', **dataset_kwargs)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Create validation and test dataloaders (no special sampling needed)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating ablation model...")
    model = create_ablation_model(
        ablation_config=ablation_config,
        backbone=args.backbone,
        num_concepts=args.num_concepts,
        fairness_lambda=args.fairness_lambda,
        adversarial_lambda=args.adversarial_lambda,
        device=device
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\nStarting training...")
    best_val_f1 = -1.0  # Initialize to -1 so any positive score saves
    results = {
        'train_history': [],
        'val_history': [],
        'test_results': None,
        'ablation_config': ablation_config.key
    }
    
    # Collect all training labels and groups for FairnessAwareSampler
    print("Collecting dataset labels and groups for phase-aware sampling...")
    all_groups = []
    all_labels = []
    for sample in train_dataset:
        # sample format: (image, concepts, binary_label, fitzpatrick)
        all_labels.append(sample[2].item() if torch.is_tensor(sample[2]) else sample[2])
        all_groups.append((sample[3].item() if torch.is_tensor(sample[3]) else sample[3]) - 1)
    all_groups = torch.tensor(all_groups, dtype=torch.long)
    all_labels = torch.tensor(all_labels, dtype=torch.float)
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Determine current phase and if it's active in ablation
        progress = epoch / args.epochs
        if progress <= 0.25:
            phase_num = 1
            phase_is_active = ablation_config.use_phase1_balanced
        elif progress <= 0.50:
            phase_num = 2
            phase_is_active = ablation_config.use_phase2_dp
        elif progress <= 0.75:
            phase_num = 3
            phase_is_active = ablation_config.use_phase3_eo
        else:
            phase_num = 4
            phase_is_active = ablation_config.use_phase4_error
        
        if phase_is_active:
            # Phase is enabled: use phase-aware sampling strategy
            sampler = FairnessAwareSampler(
                groups=all_groups,
                labels=all_labels,
                batch_size=args.batch_size,
                epoch=epoch,
                total_epochs=args.epochs,
                group_f1_scores=model.group_f1_scores
            )
            
            train_loader_epoch = DataLoader(
                train_dataset,
                batch_sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=True
            )
            phase_name = model.get_phase_info(epoch, args.epochs)['phase_name']
            print(f"Using phase-aware sampling (Phase {phase_num}: {phase_name})")
        else:
            # Phase is disabled: use regular random sampling
            train_loader_epoch = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True
            )
            print(f"Using random sampling (Phase {phase_num} DISABLED in ablation)")
        
        # Train
        train_metrics = train_epoch_with_ablation(
            model, train_loader_epoch, optimizer, device, epoch, args.epochs
        )
        
        # Update group F1 scores every 5 epochs for error-driven sampling (Phase 4)
        if epoch % 5 == 0 and epoch > 0:
            # Compute per-group F1 for error-driven sampling
            model.eval()
            group_preds = {g: [] for g in range(6)}
            group_labels = {g: [] for g in range(6)}
            
            with torch.no_grad():
                for batch in val_loader:
                    images, _, binary_labels, fitzpatrick = batch
                    images = images.to(device)
                    _, binary_logits = model(images)
                    preds = torch.sigmoid(binary_logits).cpu().numpy().flatten()
                    
                    for i, g in enumerate(fitzpatrick - 1):
                        g = g.item()
                        group_preds[g].append(preds[i])
                        group_labels[g].append(binary_labels[i].item())
            
            # Compute F1 per group
            model.group_f1_scores = {}
            for g in range(6):
                if len(group_preds[g]) > 0:
                    g_preds = np.array(group_preds[g])
                    g_labels = np.array(group_labels[g])
                    g_metrics = compute_metrics(g_labels, g_preds)
                    model.group_f1_scores[g] = g_metrics['f1']
            
            model.train()
        
        print(f"\nTrain Loss: {train_metrics['total_loss']:.4f}")
        print(f"  Concept: {train_metrics['concept_loss']:.4f}")
        print(f"  Binary: {train_metrics['binary_loss']:.4f}")
        print(f"  Fairness: {train_metrics['fairness_loss']:.4f}")
        print(f"  Adversarial: {train_metrics['adversarial_loss']:.4f}")
        print(f"Train F1: {train_metrics['performance']['f1']:.4f}")
        
        phase_info = train_metrics['phase_info']
        print(f"\n  Phase: {phase_info['phase_name']}")
        print(f"    Fairness Focus: {phase_info['fairness_focus']}")
        print(f"    Adversarial: {'YES' if phase_info['adversarial_active'] else 'NO'}")
        
        # Validate
        val_results = evaluate_with_fairness(model, val_loader, device)
        val_f1 = val_results['performance']['f1']
        val_perf_gap = val_results['fairness']['worst_group']['performance_gap']
        
        print(f"\nVal F1: {val_f1:.4f}")
        print(f"Val Performance Gap: {val_perf_gap:.4f}")
        print(f"Val DP Disparity: {val_results['fairness']['demographic_parity']['max_disparity']:.4f}")
        
        # Save results
        results['train_history'].append({
            'epoch': epoch,
            'metrics': train_metrics['performance'],
            'losses': {
                'total': train_metrics['total_loss'],
                'concept': train_metrics['concept_loss'],
                'binary': train_metrics['binary_loss'],
                'fairness': train_metrics['fairness_loss'],
                'adversarial': train_metrics['adversarial_loss']
            },
            'phase_info': train_metrics['phase_info']
        })
        
        results['val_history'].append({
            'epoch': epoch,
            'performance': val_results['performance'],
            'fairness': val_results['fairness']
        })
        
        # Update best model in shared best_models directory if this run is better
        if val_f1 > best_val_f1 and val_f1 >= 0.01:
            best_val_f1 = val_f1
            
            # Check against global best using file locking
            best_f1_file = best_models_dir / f"{ablation_config.key}_best_f1.txt"
            lock_file = best_models_dir / f"{ablation_config.key}.lock"
            
            with open(lock_file, 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                
                # Read current global best F1
                if best_f1_file.exists():
                    with open(best_f1_file, 'r') as f:
                        global_best_f1 = float(f.read().strip())
                else:
                    global_best_f1 = -1.0
                
                # Only save if this run is better than global best
                if val_f1 > global_best_f1:
                    # Update best F1 file
                    with open(best_f1_file, 'w') as f:
                        f.write(f"{val_f1}\n")
                    
                    # Save model to best_models directory
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_f1': val_f1,
                        'ablation_config': ablation_config.key,
                        'run_id': args.run_id,
                        'seed': args.seed
                    }, best_models_dir / f"{ablation_config.key}_best.pt")
                    
                    # Copy config files
                    shutil.copy(output_dir / 'ablation_config.json',
                               best_models_dir / f"{ablation_config.key}_best_config.json")
                    
                    # Save metadata
                    with open(best_models_dir / f"{ablation_config.key}_best_info.txt", 'w') as f:
                        f.write(f"Run: {args.run_id}, Seed: {args.seed}, Val F1: {val_f1:.4f}, Epoch: {epoch}\n")
                    
                    print(f"  ✓ NEW GLOBAL BEST {ablation_config.key}: Val F1 = {val_f1:.4f} (previous: {global_best_f1:.4f})")
                else:
                    print(f"  → Local best (F1: {val_f1:.4f}), but not global best (F1: {global_best_f1:.4f})")
                
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        elif val_f1 < 0.01:
            print(f"  → F1 too low ({val_f1:.4f}), not saving model")
        
        # Note: Checkpoints not saved - only best models saved to best_models/ directory
    
    # Final test evaluation
    print(f"\n{'='*60}")
    print("Final Test Evaluation")
    print(f"{'='*60}")
    
    # Use current model state for test evaluation (best epoch model already saved to best_models/ if good enough)
    print(f"Testing with final model state (best val F1: {best_val_f1:.4f})")
    
    test_results = evaluate_with_fairness(model, test_loader, device)
    results['test_results'] = {
        'performance': test_results['performance'],
        'fairness': test_results['fairness']
    }
    
    print(f"\nTest F1: {test_results['performance']['f1']:.4f}")
    print(f"Test Accuracy: {test_results['performance']['accuracy']:.4f}")
    print(f"Test Performance Gap: {test_results['fairness']['worst_group']['performance_gap']:.4f}")
    print(f"Test DP Disparity: {test_results['fairness']['demographic_parity']['max_disparity']:.4f}")
    
    # Save final results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_dir}")
    print("Training complete!")


if __name__ == '__main__':
    main()
