"""
Analysis script for multi-run fairness experiments.

This script:
1. Aggregates results from 100 runs per model type
2. Computes statistical summaries (mean, std, CI)
3. Performs pairwise statistical tests
4. Generates publication-ready visualizations
5. Creates comprehensive comparison tables

Usage:
    python analyze_multi_run_results.py --exp_name multi_run_1234567 --backbone swin
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11


def load_run_results(results_dir, exp_name, model_type, n_runs=100):
    """Load results from all runs for a model type."""
    results_list = []
    
    for run_id in range(1, n_runs + 1):
        history_path = results_dir / exp_name / f"run_{run_id}" / model_type / "history.json"
        
        if not history_path.exists():
            print(f"Warning: Missing results for {model_type} run {run_id}")
            continue
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Extract test results
        if len(history['test']) > 0:
            test_results = history['test'][0]
            results_list.append({
                'run_id': run_id,
                'model_type': model_type,
                **test_results['standard_metrics'],
                **flatten_fairness_metrics(test_results.get('fairness_metrics', {}))
            })
    
    return pd.DataFrame(results_list)


def flatten_fairness_metrics(fairness_metrics):
    """Flatten nested fairness metrics dictionary."""
    flat = {}
    
    if 'demographic_parity' in fairness_metrics:
        flat['demographic_parity'] = fairness_metrics['demographic_parity']['max_disparity']
        flat['disparate_impact'] = fairness_metrics['demographic_parity']['disparate_impact']
    
    if 'equalized_odds' in fairness_metrics:
        flat['equalized_odds_tpr'] = fairness_metrics['equalized_odds']['max_tpr_disparity']
        flat['equalized_odds_fpr'] = fairness_metrics['equalized_odds']['max_fpr_disparity']
    
    if 'worst_group' in fairness_metrics:
        flat['performance_gap'] = fairness_metrics['worst_group']['performance_gap']
        flat['worst_group_f1'] = fairness_metrics['worst_group']['worst_f1']
        flat['worst_group_accuracy'] = fairness_metrics['worst_group']['worst_accuracy']
    
    if 'calibration' in fairness_metrics:
        flat['calibration_disparity'] = fairness_metrics['calibration']['max_ece_disparity']
    
    return flat


def compute_summary_statistics(df, metrics):
    """Compute mean, std, and 95% CI for metrics."""
    summary = {}
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        values = df[metric].values
        mean = np.mean(values)
        std = np.std(values)
        se = stats.sem(values)
        ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=se)
        
        summary[metric] = {
            'mean': mean,
            'std': std,
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }
    
    return summary


def pairwise_ttest(df1, df2, metric):
    """Perform paired t-test between two model results."""
    values1 = df1[metric].values
    values2 = df2[metric].values
    
    # Ensure same number of runs
    n = min(len(values1), len(values2))
    values1 = values1[:n]
    values2 = values2[:n]
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(values1, values2)
    
    # Effect size (Cohen's d)
    diff = values1 - values2
    cohens_d = np.mean(diff) / np.std(diff)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_diff': np.mean(diff),
        'significant': p_value < 0.05
    }


def create_summary_table(results_dict, metrics):
    """Create summary table with mean ± std for all models."""
    rows = []
    
    for model_type, summary in results_dict.items():
        row = {'Model': model_type}
        
        for metric in metrics:
            if metric in summary:
                mean = summary[metric]['mean']
                std = summary[metric]['std']
                row[metric] = f"{mean:.4f} ± {std:.4f}"
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_metric_distributions(all_results_df, metrics, save_dir):
    """Plot violin plots for metric distributions."""
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    model_order = ['direct', 'standard_cbm', 'curriculum_cbm', 'fair_curriculum_cbm']
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Create violin plot
        parts = ax.violinplot(
            [all_results_df[all_results_df['model_type'] == mt][metric].values 
             for mt in model_order if mt in all_results_df['model_type'].values],
            positions=range(len(model_order)),
            showmeans=True,
            showmedians=True
        )
        
        # Color violins
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels(model_order, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Distribution of {metric.replace("_", " ").title()}', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    # Remove empty subplots
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metric_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved metric distributions to {save_dir / 'metric_distributions.png'}")


def plot_fairness_performance_scatter(all_results_df, save_dir):
    """Scatter plot: Overall performance vs. fairness."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    model_types = all_results_df['model_type'].unique()
    colors = {'direct': '#ff7f0e', 'standard_cbm': '#2ca02c', 
              'curriculum_cbm': '#1f77b4', 'fair_curriculum_cbm': '#d62728'}
    
    # F1 vs Performance Gap
    ax = axes[0]
    for model in model_types:
        df = all_results_df[all_results_df['model_type'] == model]
        ax.scatter(df['performance_gap'], df['f1'], 
                  c=colors.get(model, 'gray'), label=model, s=50, alpha=0.6)
    
    ax.set_xlabel('Performance Gap (Lower is better)', fontsize=12)
    ax.set_ylabel('Overall F1 Score (Higher is better)', fontsize=12)
    ax.set_title('Performance-Fairness Tradeoff (F1)', fontsize=14, fontweight='bold')
    ax.axhline(y=0.70, color='green', linestyle='--', alpha=0.5, label='Target F1 ≥ 0.70')
    ax.axvline(x=0.15, color='green', linestyle='--', alpha=0.5, label='Target Gap ≤ 0.15')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # AUC vs Demographic Parity
    ax = axes[1]
    for model in model_types:
        df = all_results_df[all_results_df['model_type'] == model]
        ax.scatter(df['demographic_parity'], df['auc'], 
                  c=colors.get(model, 'gray'), label=model, s=50, alpha=0.6)
    
    ax.set_xlabel('Demographic Parity Disparity (Lower is better)', fontsize=12)
    ax.set_ylabel('Overall AUC (Higher is better)', fontsize=12)
    ax.set_title('Performance-Fairness Tradeoff (AUC)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'fairness_performance_scatter.png', dpi=300, bbox_inches='tight')
    print(f"Saved fairness-performance scatter to {save_dir / 'fairness_performance_scatter.png'}")


def plot_pairwise_comparison(summary_dict, metric, save_dir):
    """Bar plot comparing models on a specific metric."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(summary_dict.keys())
    means = [summary_dict[m][metric]['mean'] for m in models]
    stds = [summary_dict[m][metric]['std'] for m in models]
    
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728']
    
    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
               f'{mean:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{metric.replace("_", " ").title()} Comparison (Mean ± Std)', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{metric}_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved {metric} comparison to {save_dir / f'{metric}_comparison.png'}")


def generate_latex_table(summary_table_df, save_path):
    """Generate LaTeX table for publication."""
    latex = summary_table_df.to_latex(index=False, escape=False)
    
    with open(save_path, 'w') as f:
        f.write(latex)
    
    print(f"Saved LaTeX table to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze multi-run fairness experiments')
    
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment name (e.g., multi_run_1234567)')
    parser.add_argument('--backbone', type=str, default='swin',
                        help='Backbone architecture')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory with results')
    parser.add_argument('--n_runs', type=int, default=100,
                        help='Number of runs to analyze')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    analysis_dir = results_dir / args.exp_name / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("MULTI-RUN FAIRNESS ANALYSIS")
    print("="*80)
    print(f"Experiment: {args.exp_name}")
    print(f"Backbone: {args.backbone}")
    print(f"Number of runs: {args.n_runs}")
    print(f"Results directory: {results_dir}")
    print("")
    
    # Model types
    model_types = ['direct', 'standard_cbm', 'curriculum_cbm', 'fair_curriculum_cbm']
    
    # Load results for all models
    print("Loading results...")
    all_results = []
    results_dict = {}
    
    for model_type in model_types:
        print(f"  Loading {model_type}...")
        df = load_run_results(results_dir, args.exp_name, model_type, args.n_runs)
        
        if len(df) == 0:
            print(f"    Warning: No results found for {model_type}")
            continue
        
        all_results.append(df)
        results_dict[model_type] = df
        print(f"    Loaded {len(df)} runs")
    
    all_results_df = pd.concat(all_results, ignore_index=True)
    print(f"\nTotal results loaded: {len(all_results_df)}")
    
    # Metrics to analyze
    standard_metrics = ['f1', 'accuracy', 'precision', 'recall', 'auc']
    fairness_metrics = ['demographic_parity', 'equalized_odds_tpr', 'equalized_odds_fpr',
                       'performance_gap', 'worst_group_f1', 'calibration_disparity']
    all_metrics = standard_metrics + fairness_metrics
    
    # Compute summary statistics
    print("\n" + "="*80)
    print("COMPUTING SUMMARY STATISTICS")
    print("="*80)
    
    summary_dict = {}
    for model_type, df in results_dict.items():
        print(f"\n{model_type}:")
        summary = compute_summary_statistics(df, all_metrics)
        summary_dict[model_type] = summary
        
        # Print key metrics
        if 'f1' in summary:
            print(f"  F1: {summary['f1']['mean']:.4f} ± {summary['f1']['std']:.4f}")
        if 'performance_gap' in summary:
            print(f"  Performance Gap: {summary['performance_gap']['mean']:.4f} ± {summary['performance_gap']['std']:.4f}")
        if 'worst_group_f1' in summary:
            print(f"  Worst-Group F1: {summary['worst_group_f1']['mean']:.4f} ± {summary['worst_group_f1']['std']:.4f}")
    
    # Create summary table
    summary_table = create_summary_table(summary_dict, all_metrics)
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_table.to_string(index=False))
    summary_table.to_csv(analysis_dir / 'summary_table.csv', index=False)
    generate_latex_table(summary_table, analysis_dir / 'summary_table.tex')
    
    # Statistical significance tests
    print("\n" + "="*80)
    print("PAIRWISE STATISTICAL TESTS")
    print("="*80)
    
    test_results = {}
    baseline = 'curriculum_cbm'
    comparison = 'fair_curriculum_cbm'
    
    if baseline in results_dict and comparison in results_dict:
        print(f"\n{baseline} vs {comparison}:")
        
        for metric in all_metrics:
            if metric not in results_dict[baseline].columns:
                continue
            
            test_result = pairwise_ttest(
                results_dict[baseline],
                results_dict[comparison],
                metric
            )
            test_results[metric] = test_result
            
            print(f"\n  {metric}:")
            print(f"    Mean difference: {test_result['mean_diff']:.6f}")
            print(f"    t-statistic: {test_result['t_statistic']:.4f}")
            print(f"    p-value: {test_result['p_value']:.6f}")
            print(f"    Cohen's d: {test_result['cohens_d']:.4f}")
            print(f"    Significant: {'✓ Yes' if test_result['significant'] else '✗ No'}")
    
    # Save test results
    test_results_df = pd.DataFrame(test_results).T
    test_results_df.to_csv(analysis_dir / 'statistical_tests.csv')
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Key metrics for visualization
    viz_metrics = ['f1', 'performance_gap', 'worst_group_f1', 'demographic_parity']
    
    plot_metric_distributions(all_results_df, viz_metrics, analysis_dir)
    plot_fairness_performance_scatter(all_results_df, analysis_dir)
    
    for metric in ['f1', 'performance_gap', 'worst_group_f1']:
        if metric in summary_dict[list(summary_dict.keys())[0]]:
            plot_pairwise_comparison(summary_dict, metric, analysis_dir)
    
    # Save raw results
    all_results_df.to_csv(analysis_dir / 'all_results.csv', index=False)
    
    # Generate final report
    report = {
        'experiment': args.exp_name,
        'backbone': args.backbone,
        'n_runs': args.n_runs,
        'models': list(summary_dict.keys()),
        'summary_statistics': {k: {metric: summary_dict[k][metric] 
                                   for metric in all_metrics if metric in summary_dict[k]}
                              for k in summary_dict.keys()},
        'statistical_tests': test_results
    }
    
    with open(analysis_dir / 'analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {analysis_dir}")
    print(f"  - summary_table.csv: Summary statistics for all models")
    print(f"  - summary_table.tex: LaTeX table for publication")
    print(f"  - statistical_tests.csv: Pairwise significance tests")
    print(f"  - all_results.csv: Raw results from all runs")
    print(f"  - *.png: Visualization plots")
    print(f"  - analysis_report.json: Complete analysis report")


if __name__ == '__main__':
    main()
