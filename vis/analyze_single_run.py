#!/usr/bin/env python3
"""
Quick analysis script for a single model run.

Usage:
    python scripts/analyze_single_run.py --exp_dir results/test_single_1516680/fair_curriculum_cbm
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_single_run(exp_dir):
    """Analyze results from a single training run."""
    exp_path = Path(exp_dir)
    history_file = exp_path / 'history.json'
    
    if not history_file.exists():
        print(f"Error: history.json not found in {exp_dir}")
        return
    
    # Load history
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    print("="*70)
    print(f"ANALYSIS: {exp_path.name}")
    print("="*70)
    
    # Training summary
    train_history = history['train']
    print(f"\nTraining: {len(train_history)} epochs")
    print(f"  Initial Train F1: {train_history[0]['metrics']['f1']:.4f}")
    print(f"  Final Train F1:   {train_history[-1]['metrics']['f1']:.4f}")
    
    # Validation summary
    val_history = history['val']
    val_f1s = [epoch['standard_metrics']['f1'] for epoch in val_history]
    best_val_idx = val_f1s.index(max(val_f1s))
    best_val_epoch = val_history[best_val_idx]
    
    print(f"\nValidation: {len(val_history)} evaluations")
    print(f"  Best Val F1: {max(val_f1s):.4f} (epoch {best_val_epoch['epoch']})")
    print(f"  Final Val F1: {val_f1s[-1]:.4f}")
    
    # Test results
    if 'test' in history and history['test']:
        test_results = history['test'][0]
        test_metrics = test_results['standard_metrics']
        
        print(f"\n" + "="*70)
        print("TEST SET RESULTS")
        print("="*70)
        print(f"\nStandard Metrics:")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  F1 Score:  {test_metrics['f1']:.4f}")
        print(f"  AUC:       {test_metrics['auc']:.4f}")
        
        # Fairness metrics
        if 'fairness_metrics' in test_results:
            fairness = test_results['fairness_metrics']
            
            print(f"\nFairness Metrics:")
            
            # Demographic parity
            if 'demographic_parity' in fairness:
                dp = fairness['demographic_parity']
                print(f"\n  Demographic Parity:")
                print(f"    Statistical Parity Difference: {dp.get('statistical_parity_difference', 'N/A')}")
                print(f"    Group Positive Rates:")
                for group, rate in dp['group_positive_rates'].items():
                    count = dp['group_counts'][group]
                    print(f"      Fitzpatrick {int(group)+1}: {rate:.4f} (n={count})")
            
            # Equalized odds
            if 'equalized_odds' in fairness:
                eo = fairness['equalized_odds']
                print(f"\n  Equalized Odds:")
                print(f"    TPR Disparity:  {eo.get('tpr_disparity', 'N/A')}")
                print(f"    FPR Disparity:  {eo.get('fpr_disparity', 'N/A')}")
                print(f"    EOD:            {eo.get('equalized_odds_difference', 'N/A')}")
            
            # Performance parity
            if 'worst_group_performance' in fairness:
                wgp = fairness['worst_group_performance']
                print(f"\n  Performance Parity:")
                print(f"    Overall F1:     {wgp.get('overall_f1', 'N/A'):.4f}")
                print(f"    Worst Group F1: {wgp.get('worst_f1', 'N/A'):.4f} (Group {wgp.get('worst_group', 'N/A')})")
                print(f"    Best Group F1:  {wgp.get('best_f1', 'N/A'):.4f} (Group {wgp.get('best_group', 'N/A')})")
                print(f"    Performance Gap: {wgp.get('gap', 'N/A'):.4f}")
    
    # Generate plots
    print(f"\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    output_dir = exp_path / 'analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # F1 Score
    train_f1 = [e['metrics']['f1'] for e in train_history]
    val_f1 = [e['standard_metrics']['f1'] for e in val_history]
    val_epochs = [e['epoch'] for e in val_history]
    
    axes[0, 0].plot(range(1, len(train_f1)+1), train_f1, label='Train', alpha=0.7)
    axes[0, 0].plot(val_epochs, val_f1, label='Val', marker='o')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].set_title('F1 Score over Training')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss components
    if 'fairness_loss' in train_history[0]:
        concept_loss = [e['concept_loss'] for e in train_history]
        binary_loss = [e['binary_loss'] for e in train_history]
        fairness_loss = [e['fairness_loss'] for e in train_history]
        adversarial_loss = [e['adversarial_loss'] for e in train_history]
        
        axes[0, 1].plot(concept_loss, label='Concept', alpha=0.7)
        axes[0, 1].plot(binary_loss, label='Binary', alpha=0.7)
        axes[0, 1].plot(fairness_loss, label='Fairness', alpha=0.7)
        axes[0, 1].plot(adversarial_loss, label='Adversarial', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Adversarial lambda schedule
        adv_lambda = [e['adversarial_lambda'] for e in train_history]
        axes[1, 0].plot(adv_lambda, color='red', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Adversarial Lambda')
        axes[1, 0].set_title('Adversarial Lambda Warmup Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Adversarial alpha schedule
        adv_alpha = [e['adversarial_alpha'] for e in train_history]
        axes[1, 1].plot(adv_alpha, color='purple', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Adversarial Alpha')
        axes[1, 1].set_title('Gradient Reversal Alpha Schedule')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'training_curves.png'}")
    
    # Plot 2: Group performance
    if 'test' in history and history['test'] and 'fairness_metrics' in history['test'][0]:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        fairness = history['test'][0]['fairness_metrics']
        if 'worst_group_performance' in fairness and 'group_f1' in fairness['worst_group_performance']:
            group_f1 = fairness['worst_group_performance']['group_f1']
            groups = sorted([int(g) for g in group_f1.keys()])
            f1_scores = [group_f1[str(g)] for g in groups]
            
            bars = ax.bar([f'F{g+1}' for g in groups], f1_scores, 
                         color=sns.color_palette("viridis", len(groups)))
            ax.axhline(test_metrics['f1'], color='red', linestyle='--', 
                      label=f'Overall F1: {test_metrics["f1"]:.3f}')
            ax.set_xlabel('Fitzpatrick Skin Type')
            ax.set_ylabel('F1 Score')
            ax.set_title('Test F1 Score by Fitzpatrick Type')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add values on bars
            for bar, score in zip(bars, f1_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'group_performance.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_dir / 'group_performance.png'}")
    
    plt.close('all')
    print(f"\n" + "="*70)
    print(f"Analysis complete! Results saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze single model run')
    parser.add_argument('--exp_dir', type=str, required=True,
                       help='Path to experiment directory (e.g., results/test_single_1516680/fair_curriculum_cbm)')
    
    args = parser.parse_args()
    analyze_single_run(args.exp_dir)
