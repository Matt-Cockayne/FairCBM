"""
Evaluation script for comparing fairness across all 4 model types.

This script:
1. Loads trained models (Direct, Standard CBM, Curriculum CBM, Fair Curriculum CBM)
2. Evaluates on test set with standard and fairness metrics
3. Generates comparison tables and visualizations
4. Performs statistical significance testing

Usage:
    python evaluate_fairness_comparison.py --exp_name exp_001 --backbone swin
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import torch
import numpy as np
from pathlib import Path
import json
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import existing SynergyCBM components
from src.data.dataloader import SkinCapDataset
from src.models.direct_classifier import DirectClassifier
from src.models.standard_cbm import StandardCBM
from src.models.minimal_curriculum_cbm import MinimalCurriculumCBM
from src.utils.metrics import compute_metrics

# Import fairness components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.fairness_aware_cbm import FairnessAwareCBM
from src.utils.fairness_metrics import compute_all_fairness_metrics, compute_fairness_summary


def load_model(model_type, checkpoint_path, num_concepts, backbone='swin'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})
    
    # Create model
    if model_type == 'direct':
        model = DirectClassifier(backbone=backbone)
    elif model_type == 'standard_cbm':
        model = StandardCBM(num_concepts=num_concepts, backbone=backbone)
    elif model_type == 'curriculum_cbm':
        model = MinimalCurriculumCBM(num_concepts=num_concepts, backbone=backbone)
    elif model_type == 'fair_curriculum_cbm':
        fairness_lambda = config.get('fairness_lambda', 1.0)
        adversarial_lambda = config.get('adversarial_lambda', 0.5)
        model = FairnessAwareCBM(
            num_concepts=num_concepts,
            backbone=backbone,
            num_groups=6,
            fairness_lambda=fairness_lambda,
            adversarial_lambda=adversarial_lambda
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def evaluate_model(model, dataloader, device, model_type):
    """Evaluate model and return predictions."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    all_groups = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {model_type}"):
            images, _, binary_labels, fitzpatrick = batch
            images = images.to(device)
            
            if model_type == 'direct':
                logits = model(images)
            else:
                concept_logits, binary_logits = model(images)
                logits = binary_logits
            
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(binary_labels.numpy())
            all_groups.extend(fitzpatrick.numpy())
    
    return {
        'predictions': np.array(all_preds),
        'probabilities': np.array(all_probs),
        'labels': np.array(all_labels),
        'groups': np.array(all_groups)
    }


def compute_group_metrics(predictions, labels, probabilities, groups, group_names):
    """Compute per-group performance metrics."""
    group_metrics = []
    
    for group_idx, group_name in enumerate(group_names):
        mask = (groups == group_idx)
        if mask.sum() == 0:
            continue
        
        group_preds = predictions[mask]
        group_labels = labels[mask]
        group_probs = probabilities[mask]
        
        metrics = compute_metrics(group_labels, group_probs)
        metrics['group'] = group_name
        metrics['n_samples'] = mask.sum()
        group_metrics.append(metrics)
    
    return pd.DataFrame(group_metrics)


def create_comparison_table(results_dict):
    """Create comparison table for all models."""
    rows = []
    
    for model_type, results in results_dict.items():
        row = {
            'Model': model_type,
            'F1': results['standard_metrics']['f1'],
            'Accuracy': results['standard_metrics']['accuracy'],
            'AUC': results['standard_metrics']['auc'],
            'Demographic Parity': results['fairness_metrics']['demographic_parity']['max_disparity'],
            'Equalized Odds (TPR)': results['fairness_metrics']['equalized_odds']['max_tpr_disparity'],
            'Equalized Odds (FPR)': results['fairness_metrics']['equalized_odds']['max_fpr_disparity'],
            'Performance Gap': results['fairness_metrics']['worst_group']['performance_gap'],
            'Worst-Group F1': results['fairness_metrics']['worst_group']['worst_f1']
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_per_group_performance(results_dict, save_path):
    """Plot per-group F1 scores for all models."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    group_names = [f"Fitz-{i+1}" for i in range(6)]
    x = np.arange(len(group_names))
    width = 0.2
    
    for i, (model_type, results) in enumerate(results_dict.items()):
        predictions = results['predictions']
        labels = results['labels']
        probabilities = results['probabilities']
        groups = (results['groups'] - 1).astype(int)  # Convert 1-6 to 0-5
        
        group_df = compute_group_metrics(predictions, labels, probabilities, groups, group_names)
        f1_scores = [group_df[group_df['group'] == gn]['f1'].values[0] if len(group_df[group_df['group'] == gn]) > 0 else 0 
                     for gn in group_names]
        
        ax.bar(x + i * width, f1_scores, width, label=model_type)
    
    ax.set_xlabel('Fitzpatrick Skin Type', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Group Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(group_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved per-group performance plot to {save_path}")


def plot_fairness_metrics_comparison(comparison_df, save_path):
    """Plot fairness metrics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [
        ('Demographic Parity', 'Lower is better'),
        ('Equalized Odds (TPR)', 'Lower is better'),
        ('Performance Gap', 'Lower is better'),
        ('Worst-Group F1', 'Higher is better')
    ]
    
    for ax, (metric, label) in zip(axes.flat, metrics):
        values = comparison_df[metric].values
        models = comparison_df['Model'].values
        
        colors = ['#ff7f0e' if 'fair' in m.lower() else '#1f77b4' for m in models]
        bars = ax.barh(models, values, color=colors)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=10)
        
        ax.set_xlabel(label, fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved fairness metrics comparison to {save_path}")


def plot_performance_fairness_tradeoff(comparison_df, save_path):
    """Plot performance vs. fairness tradeoff."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x = comparison_df['Performance Gap'].values
    y = comparison_df['F1'].values
    models = comparison_df['Model'].values
    
    colors = ['red' if 'fair' in m.lower() else 'blue' for m in models]
    
    for i, (xi, yi, model, color) in enumerate(zip(x, y, models, colors)):
        ax.scatter(xi, yi, s=200, c=color, alpha=0.6, edgecolors='black', linewidth=2)
        ax.annotate(model, (xi, yi), xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3))
    
    ax.set_xlabel('Performance Gap (Lower is better)', fontsize=12)
    ax.set_ylabel('Overall F1 Score (Higher is better)', fontsize=12)
    ax.set_title('Performance-Fairness Tradeoff', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add ideal region
    ax.axhline(y=0.70, color='green', linestyle='--', alpha=0.5, label='Target F1 ≥ 0.70')
    ax.axvline(x=0.15, color='green', linestyle='--', alpha=0.5, label='Target Gap ≤ 0.15')
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved performance-fairness tradeoff to {save_path}")


def statistical_significance_test(results1, results2, metric='f1', n_bootstrap=1000):
    """
    Perform bootstrap hypothesis test for statistical significance.
    
    Args:
        results1, results2: Evaluation results dictionaries
        metric: Metric to compare ('f1', 'auc', etc.)
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        dict: Test results including p-value and confidence interval
    """
    # Get predictions and labels
    labels1 = results1['labels']
    probs1 = results1['probabilities']
    
    labels2 = results2['labels']
    probs2 = results2['probabilities']
    
    # Compute observed difference
    if metric == 'f1':
        score1 = compute_metrics(labels1, probs1)['f1']
        score2 = compute_metrics(labels2, probs2)['f1']
    elif metric == 'auc':
        score1 = compute_metrics(labels1, probs1)['auc']
        score2 = compute_metrics(labels2, probs2)['auc']
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    observed_diff = score2 - score1
    
    # Bootstrap resampling
    n_samples = len(labels1)
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        if metric == 'f1':
            boot_score1 = compute_metrics(labels1[indices], probs1[indices])['f1']
            boot_score2 = compute_metrics(labels2[indices], probs2[indices])['f1']
        elif metric == 'auc':
            boot_score1 = compute_metrics(labels1[indices], probs1[indices])['auc']
            boot_score2 = compute_metrics(labels2[indices], probs2[indices])['auc']
        
        bootstrap_diffs.append(boot_score2 - boot_score1)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Compute p-value (two-tailed)
    p_value = 2 * min(
        np.mean(bootstrap_diffs <= 0),
        np.mean(bootstrap_diffs >= 0)
    )
    
    # Confidence interval
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    
    return {
        'observed_difference': observed_diff,
        'p_value': p_value,
        'ci_95': (ci_lower, ci_upper),
        'score1': score1,
        'score2': score2,
        'significant': p_value < 0.05
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare fairness across models')
    
    # Experiment configuration
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment name')
    parser.add_argument('--backbone', type=str, default='swin',
                        help='Backbone architecture')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory with training results')
    
    # Data configuration
    parser.add_argument('--data_root', type=str, default='data/skincap',
                        help='Root directory for dataset')
    parser.add_argument('--concepts_path', type=str, default='data/skincap_concepts.txt',
                        help='Path to concept names file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    # Evaluation configuration
    parser.add_argument('--checkpoint', type=str, default='best_model.pt',
                        choices=['best_model.pt', 'final_model.pt'],
                        help='Which checkpoint to use')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap samples for significance testing')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load concept names
    concepts_path = Path(args.concepts_path)
    if not concepts_path.exists():
        concepts_path = Path('data/concepts.txt')
    
    with open(concepts_path, 'r') as f:
        concepts = [line.strip() for line in f if line.strip()]
    num_concepts = len(concepts)
    
    # Create test dataloader
    print("Creating test dataset...")
    test_dataset = SkinCapDataset(
        root_dir=args.data_root,
        split='test',
        label_type='concept'  # Use concept for all models
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Test set: {len(test_dataset)} samples")
    
    # Model types to evaluate
    model_types = ['direct', 'standard_cbm', 'curriculum_cbm', 'fair_curriculum_cbm']
    results_dict = {}
    
    # Load and evaluate each model
    results_dir = Path(args.results_dir) / args.exp_name
    
    for model_type in model_types:
        checkpoint_path = results_dir / model_type / args.checkpoint
        
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found for {model_type}: {checkpoint_path}")
            continue
        
        print(f"\nEvaluating {model_type}...")
        
        # Load model
        model = load_model(model_type, checkpoint_path, num_concepts, args.backbone)
        model = model.to(device)
        
        # Evaluate
        eval_results = evaluate_model(model, test_loader, device, model_type)
        
        # Compute standard metrics
        standard_metrics = compute_metrics(
            eval_results['labels'],
            eval_results['probabilities']
        )
        
        # Compute fairness metrics
        groups = (eval_results['groups'] - 1).astype(int)  # Convert 1-6 to 0-5
        fairness_metrics = compute_all_fairness_metrics(
            predictions=eval_results['predictions'],
            labels=eval_results['labels'].astype(int),
            probabilities=eval_results['probabilities'],
            groups=groups,
            group_names=[f"Fitz-{i+1}" for i in range(6)]
        )
        
        results_dict[model_type] = {
            **eval_results,
            'standard_metrics': standard_metrics,
            'fairness_metrics': fairness_metrics
        }
        
        print(f"{model_type} - F1: {standard_metrics['f1']:.4f}, "
              f"Performance Gap: {fairness_metrics['worst_group']['performance_gap']:.4f}")
    
    # Create comparison directory
    comparison_dir = results_dir / 'comparison'
    comparison_dir.mkdir(exist_ok=True)
    
    # Generate comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    comparison_df = create_comparison_table(results_dict)
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(comparison_dir / 'comparison_table.csv', index=False)
    
    # Statistical significance tests
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    
    if 'curriculum_cbm' in results_dict and 'fair_curriculum_cbm' in results_dict:
        print("\nCurriculum CBM vs. Fair Curriculum CBM:")
        
        for metric in ['f1', 'auc']:
            test_result = statistical_significance_test(
                results_dict['curriculum_cbm'],
                results_dict['fair_curriculum_cbm'],
                metric=metric,
                n_bootstrap=args.n_bootstrap
            )
            
            print(f"\n{metric.upper()}:")
            print(f"  Curriculum CBM: {test_result['score1']:.4f}")
            print(f"  Fair Curriculum CBM: {test_result['score2']:.4f}")
            print(f"  Difference: {test_result['observed_difference']:.4f}")
            print(f"  95% CI: [{test_result['ci_95'][0]:.4f}, {test_result['ci_95'][1]:.4f}]")
            print(f"  p-value: {test_result['p_value']:.4f}")
            print(f"  Significant: {'Yes' if test_result['significant'] else 'No'}")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_per_group_performance(results_dict, comparison_dir / 'per_group_performance.png')
    plot_fairness_metrics_comparison(comparison_df, comparison_dir / 'fairness_metrics_comparison.png')
    plot_performance_fairness_tradeoff(comparison_df, comparison_dir / 'performance_fairness_tradeoff.png')
    
    # Generate fairness summary
    fairness_summary = {}
    for model_type, results in results_dict.items():
        groups = (results['groups'] - 1).astype(int)
        summary = compute_fairness_summary(
            predictions=results['predictions'],
            labels=results['labels'].astype(int),
            probabilities=results['probabilities'],
            groups=groups,
            group_names=[f"Fitz-{i+1}" for i in range(6)]
        )
        fairness_summary[model_type] = summary
    
    with open(comparison_dir / 'fairness_summary.json', 'w') as f:
        json.dump(fairness_summary, f, indent=2)
    
    print(f"\nEvaluation complete! Results saved to {comparison_dir}")


if __name__ == '__main__':
    main()
