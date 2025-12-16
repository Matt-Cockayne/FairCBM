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


def load_run_results(results_dir, model_type, n_runs=100):
    """
    Load results from all runs for a model type.
    
    Searches across all multi_run_* directories in results_dir since each
    SLURM array job creates a separate multi_run_JOBID directory with only
    its specific run_X subdirectories.
    """
    results_list = []
    
    # Find all multi_run directories
    multi_run_dirs = sorted(results_dir.glob("multi_run_*"))
    
    if not multi_run_dirs:
        print(f"Warning: No multi_run_* directories found in {results_dir}")
        return pd.DataFrame()
    
    print(f"  Searching across {len(multi_run_dirs)} multi_run directories...")
    
    # Search all run directories across all multi_run directories
    runs_found = 0
    for multi_run_dir in multi_run_dirs:
        # Find all run_X directories in this multi_run
        for run_dir in sorted(multi_run_dir.glob("run_*")):
            history_path = run_dir / model_type / "history.json"
            
            if not history_path.exists():
                continue
            
            # Extract run_id from directory name
            run_id = int(run_dir.name.split("_")[1])
            
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                # Extract test results
                if len(history['test']) > 0:
                    test_results = history['test'][0]
                    results_list.append({
                        'run_id': run_id,
                        'job_dir': multi_run_dir.name,
                        'model_type': model_type,
                        **test_results['standard_metrics'],
                        **flatten_fairness_metrics(test_results.get('fairness_metrics', {}))
                    })
                    runs_found += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"    Warning: Error loading {history_path}: {e}")
                continue
    
    print(f"  Found {runs_found} completed runs for {model_type}")
    
    if runs_found < n_runs:
        print(f"  Warning: Expected {n_runs} runs but found {runs_found}")
    
    return pd.DataFrame(results_list)


def flatten_fairness_metrics(fairness_metrics):
    """Flatten nested fairness metrics dictionary and extract per-group metrics."""
    flat = {}
    
    if 'demographic_parity' in fairness_metrics:
        dp = fairness_metrics['demographic_parity']
        flat['demographic_parity'] = dp.get('max_disparity', np.nan)
        flat['disparate_impact_ratio'] = dp.get('disparate_impact_ratio', np.nan)
        flat['statistical_parity_diff'] = dp.get('statistical_parity_difference', np.nan)
    
    if 'equalized_odds' in fairness_metrics:
        eo = fairness_metrics['equalized_odds']
        flat['equalized_odds_tpr'] = eo.get('tpr_disparity', np.nan)
        flat['equalized_odds_fpr'] = eo.get('fpr_disparity', np.nan)
        flat['equalized_odds_diff'] = eo.get('equalized_odds_difference', np.nan)
    
    if 'equal_opportunity' in fairness_metrics:
        eop = fairness_metrics['equal_opportunity']
        flat['equal_opportunity_diff'] = eop.get('equal_opportunity_difference', np.nan)
        flat['equal_opportunity_tpr'] = eop.get('tpr_disparity', np.nan)
    
    if 'worst_group' in fairness_metrics:
        wg = fairness_metrics['worst_group']
        flat['performance_gap'] = wg.get('performance_gap', np.nan)
        flat['worst_group_f1'] = wg.get('worst_group_f1', np.nan)  # Fixed: was 'worst_f1'
        flat['best_group_f1'] = wg.get('best_group_f1', np.nan)
        flat['worst_group_accuracy'] = wg.get('worst_accuracy', np.nan)
        flat['performance_gap_ratio'] = wg.get('performance_gap_ratio', np.nan)
        
        # Extract per-Fitzpatrick group metrics (groups 0-5 = Fitzpatrick 1-6)
        group_f1 = wg.get('group_f1', {})
        group_precision = wg.get('group_precision', {})
        group_recall = wg.get('group_recall', {})
        group_accuracy = wg.get('group_accuracy', {})
        
        for group_idx in range(6):
            group_key = str(group_idx)
            flat[f'fitz_{group_idx+1}_f1'] = group_f1.get(group_key, np.nan)
            flat[f'fitz_{group_idx+1}_precision'] = group_precision.get(group_key, np.nan)
            flat[f'fitz_{group_idx+1}_recall'] = group_recall.get(group_key, np.nan)
            flat[f'fitz_{group_idx+1}_accuracy'] = group_accuracy.get(group_key, np.nan)
    
    if 'calibration' in fairness_metrics:
        cal = fairness_metrics['calibration']
        flat['calibration_disparity'] = cal.get('calibration_disparity', np.nan)
        flat['mean_ece'] = cal.get('mean_ece', np.nan)
    
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
    
    # Remove NaN and infinite values only (keep zeros - they indicate underperformance)
    valid_mask = np.isfinite(values1) & np.isfinite(values2)
    values1 = values1[valid_mask]
    values2 = values2[valid_mask]
    
    if len(values1) < 2:
        return {
            't_statistic': np.nan,
            'p_value': np.nan,
            'cohens_d': np.nan,
            'mean_diff': np.nan,
            'significant': False
        }
    
    # Compute difference
    diff = values1 - values2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    # Paired t-test - will handle zero values correctly
    try:
        t_stat, p_value = stats.ttest_rel(values1, values2)
    except Exception as e:
        # Only fails if arrays are constant (no variance in either)
        t_stat = np.nan
        p_value = np.nan
    
    # Effect size (Cohen's d) - handle zero std (happens when all differences are identical)
    if std_diff > 1e-10:  # Use small epsilon to avoid division by near-zero
        cohens_d = mean_diff / std_diff
    else:
        # No variance in differences means effect is either 0 (no difference) or undefined
        cohens_d = 0.0 if abs(mean_diff) < 1e-10 else np.inf
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_diff': mean_diff,
        'significant': p_value < 0.05 if not np.isnan(p_value) else False
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
    """Plot box plots for metric distributions (handles NaN values better than violin)."""
    # Filter metrics that actually have data
    available_metrics = [m for m in metrics if m in all_results_df.columns and 
                         all_results_df[m].notna().sum() > 0]
    
    if not available_metrics:
        print("Warning: No valid metrics found for distribution plot")
        return
    
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    model_types = sorted(all_results_df['model_type'].unique())
    colors = {'direct': '#ff7f0e', 'standard_cbm': '#2ca02c', 'curriculum_cbm': '#1f77b4',
              'fair_cbm': '#9467bd', 'fair_curriculum_cbm': '#d62728'}
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        # Prepare data for box plot
        data_to_plot = []
        labels = []
        box_colors = []
        
        for model in model_types:
            values = all_results_df[all_results_df['model_type'] == model][metric].dropna().values
            if len(values) > 0:
                data_to_plot.append(values)
                labels.append(model)
                box_colors.append(colors.get(model, 'gray'))
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                           showmeans=True, meanline=True)
            
            # Color boxes
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    # Remove empty subplots
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metric_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved metric distributions to {save_dir / 'metric_distributions.png'}")


def plot_fairness_impact_comparison(all_results_df, save_dir):
    """Show net effect histogram of fairness interventions (fair - baseline)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Define comparisons: baseline -> fair version
    comparisons = [
        ('curriculum_cbm', 'fair_curriculum_cbm', 'Fair Curriculum CBM - Curriculum CBM'),
        ('standard_cbm', 'fair_cbm', 'Fair CBM - Standard CBM')
    ]
    
    # Metrics to compare (metric, direction)
    # direction: 'higher' means positive change is good, 'lower' means negative change is good
    metrics = [
        ('f1', 'F1 Score', 'higher'),
        ('performance_gap', 'Performance Gap', 'lower'),
        ('worst_group_f1', 'Worst-Group F1', 'higher'),
        ('demographic_parity', 'Demographic Parity', 'lower'),
        ('equalized_odds_tpr', 'Equalized Odds (TPR)', 'lower'),
        ('calibration_disparity', 'Calibration Disparity', 'lower')
    ]
    
    for comp_idx, (baseline, fair, title) in enumerate(comparisons):
        ax = axes[comp_idx]
        
        if baseline not in all_results_df['model_type'].values or fair not in all_results_df['model_type'].values:
            ax.text(0.5, 0.5, f'{title}\nnot available', ha='center', va='center')
            ax.set_title(title)
            continue
        
        # Compute net effects for each metric
        net_effects = []
        metric_labels = []
        colors = []
        is_improvement = []
        
        for metric, ylabel, direction in metrics:
            if metric not in all_results_df.columns:
                continue
            
            baseline_data = all_results_df[all_results_df['model_type'] == baseline][metric].dropna()
            fair_data = all_results_df[all_results_df['model_type'] == fair][metric].dropna()
            
            if len(baseline_data) > 0 and len(fair_data) > 0:
                baseline_mean = baseline_data.mean()
                fair_mean = fair_data.mean()
                net_effect = fair_mean - baseline_mean
                
                net_effects.append(net_effect)
                metric_labels.append(ylabel)
                
                # Determine if this is an improvement
                if direction == 'higher':
                    improved = net_effect > 0
                else:  # direction == 'lower'
                    improved = net_effect < 0
                
                is_improvement.append(improved)
                colors.append('#2ca02c' if improved else '#d62728')
        
        if net_effects:
            y = np.arange(len(metric_labels))
            
            # Add reference line at zero
            ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.7, zorder=1)
            
            # Color background: green for positive improvement zone, red for negative
            x_max = max(abs(min(net_effects)), abs(max(net_effects))) * 1.2
            ax.axvspan(-x_max, 0, alpha=0.05, color='red', zorder=0)
            ax.axvspan(0, x_max, alpha=0.05, color='green', zorder=0)
            
            # Plot horizontal bars
            bars = ax.barh(y, net_effects, color=colors, alpha=0.8, 
                          edgecolor='black', linewidth=1.5)
            
            # Add value labels at end of bars
            for i, (bar, effect, improved) in enumerate(zip(bars, net_effects, is_improvement)):
                x_pos = effect + (0.02 * x_max if effect > 0 else -0.02 * x_max)
                ha = 'left' if effect > 0 else 'right'
                symbol = '✓' if improved else '✗'
                ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
                       f'{symbol} {effect:+.4f}',
                       ha=ha, va='center', fontsize=10, fontweight='bold',
                       color='darkgreen' if improved else 'darkred')
            
            ax.set_yticks(y)
            ax.set_yticklabels(metric_labels, fontsize=11)
            ax.set_xlabel('Net Effect (Fair - Baseline)', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3, zorder=0)
            ax.set_xlim([-x_max, x_max])
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ca02c', edgecolor='black', label='Improvement ✓'),
                Patch(facecolor='#d62728', edgecolor='black', label='Degradation ✗')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.suptitle('Net Effect of Fairness Interventions on Metrics', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_dir / 'fairness_impact_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved fairness impact comparison to {save_dir / 'fairness_impact_comparison.png'}")


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


def plot_per_fitzpatrick_performance(all_results_df, save_dir):
    """Plot performance metrics for each Fitzpatrick skin type across all models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    model_types = sorted(all_results_df['model_type'].unique())
    colors = {'direct': '#ff7f0e', 'standard_cbm': '#2ca02c', 'curriculum_cbm': '#1f77b4', 
              'fair_cbm': '#9467bd', 'fair_curriculum_cbm': '#d62728'}
    fitz_types = [f'Fitz-{i}' for i in range(1, 7)]
    x = np.arange(len(fitz_types))
    width = 0.15
    
    # F1 Score by Fitzpatrick Type
    ax = axes[0, 0]
    for i, model in enumerate(model_types):
        df = all_results_df[all_results_df['model_type'] == model]
        f1_means = [df[f'fitz_{j+1}_f1'].mean() for j in range(6)]
        f1_stds = [df[f'fitz_{j+1}_f1'].std() for j in range(6)]
        ax.bar(x + i*width, f1_means, width, yerr=f1_stds, label=model, 
               color=colors.get(model, 'gray'), alpha=0.8, capsize=3)
    
    ax.set_xlabel('Fitzpatrick Skin Type', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score by Fitzpatrick Skin Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(fitz_types)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Precision by Fitzpatrick Type
    ax = axes[0, 1]
    for i, model in enumerate(model_types):
        df = all_results_df[all_results_df['model_type'] == model]
        prec_means = [df[f'fitz_{j+1}_precision'].mean() for j in range(6)]
        prec_stds = [df[f'fitz_{j+1}_precision'].std() for j in range(6)]
        ax.bar(x + i*width, prec_means, width, yerr=prec_stds, label=model,
               color=colors.get(model, 'gray'), alpha=0.8, capsize=3)
    
    ax.set_xlabel('Fitzpatrick Skin Type', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision by Fitzpatrick Skin Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(fitz_types)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Recall by Fitzpatrick Type
    ax = axes[1, 0]
    for i, model in enumerate(model_types):
        df = all_results_df[all_results_df['model_type'] == model]
        rec_means = [df[f'fitz_{j+1}_recall'].mean() for j in range(6)]
        rec_stds = [df[f'fitz_{j+1}_recall'].std() for j in range(6)]
        ax.bar(x + i*width, rec_means, width, yerr=rec_stds, label=model,
               color=colors.get(model, 'gray'), alpha=0.8, capsize=3)
    
    ax.set_xlabel('Fitzpatrick Skin Type', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Recall by Fitzpatrick Skin Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(fitz_types)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Accuracy by Fitzpatrick Type
    ax = axes[1, 1]
    for i, model in enumerate(model_types):
        df = all_results_df[all_results_df['model_type'] == model]
        acc_means = [df[f'fitz_{j+1}_accuracy'].mean() for j in range(6)]
        acc_stds = [df[f'fitz_{j+1}_accuracy'].std() for j in range(6)]
        ax.bar(x + i*width, acc_means, width, yerr=acc_stds, label=model,
               color=colors.get(model, 'gray'), alpha=0.8, capsize=3)
    
    ax.set_xlabel('Fitzpatrick Skin Type', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy by Fitzpatrick Skin Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(fitz_types)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'per_fitzpatrick_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved per-Fitzpatrick performance to {save_dir / 'per_fitzpatrick_performance.png'}")


def plot_fitzpatrick_trends(all_results_df, save_dir):
    """Plot F1 trends across Fitzpatrick types."""
    model_types = sorted(all_results_df['model_type'].unique())
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = {'direct': '#ff7f0e', 'standard_cbm': '#2ca02c', 'curriculum_cbm': '#1f77b4',
              'fair_cbm': '#9467bd', 'fair_curriculum_cbm': '#d62728'}
    markers = {'direct': 'o', 'standard_cbm': 's', 'curriculum_cbm': '^',
               'fair_cbm': 'D', 'fair_curriculum_cbm': 'v'}
    
    # Plot: All models F1 trends
    fitz_types = list(range(1, 7))
    
    for model in model_types:
        df = all_results_df[all_results_df['model_type'] == model]
        means = [df[f'fitz_{i}_f1'].mean() for i in fitz_types]
        stds = [df[f'fitz_{i}_f1'].std() for i in fitz_types]
        
        ax.plot(fitz_types, means, marker=markers.get(model, 'o'),
               color=colors.get(model, 'gray'), label=model, linewidth=2,
               markersize=8, alpha=0.8)
        ax.fill_between(fitz_types, 
                       [m - s for m, s in zip(means, stds)],
                       [m + s for m, s in zip(means, stds)],
                       color=colors.get(model, 'gray'), alpha=0.2)
    
    ax.set_xlabel('Fitzpatrick Skin Type', fontsize=12)
    ax.set_ylabel('Mean F1 Score', fontsize=12)
    ax.set_title('F1 Performance Across Fitzpatrick Types', fontsize=14, fontweight='bold')
    ax.set_xticks(fitz_types)
    ax.set_xticklabels([f'Type {i}' for i in fitz_types])
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'fitzpatrick_trends.png', dpi=300, bbox_inches='tight')
    print(f"Saved Fitzpatrick trends to {save_dir / 'fitzpatrick_trends.png'}")


def plot_fairness_improvement_heatmap(all_results_df, save_dir):
    """Heatmap showing per-Fitzpatrick F1 scores for each model."""
    model_types = sorted(all_results_df['model_type'].unique())
    fitz_types = [f'Fitz-{i}' for i in range(1, 7)]
    
    # Create matrix of mean F1 scores
    f1_matrix = np.zeros((len(model_types), 6))
    for i, model in enumerate(model_types):
        df = all_results_df[all_results_df['model_type'] == model]
        for j in range(6):
            f1_matrix[i, j] = df[f'fitz_{j+1}_f1'].mean()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(f1_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1.0)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(len(model_types)))
    ax.set_xticklabels(fitz_types)
    ax.set_yticklabels(model_types)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean F1 Score', rotation=270, labelpad=20, fontsize=12)
    
    # Add text annotations
    for i in range(len(model_types)):
        for j in range(6):
            text = ax.text(j, i, f'{f1_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    ax.set_title('Mean F1 Score by Model and Fitzpatrick Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fitzpatrick Skin Type', fontsize=12)
    ax.set_ylabel('Model Type', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'fairness_improvement_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved fairness improvement heatmap to {save_dir / 'fairness_improvement_heatmap.png'}")


def create_fitzpatrick_summary_table(all_results_df, save_dir):
    """Create comprehensive summary table for per-Fitzpatrick performance."""
    model_types = sorted(all_results_df['model_type'].unique())
    
    # Create summary for F1 scores
    summary_data = []
    for model in model_types:
        df = all_results_df[all_results_df['model_type'] == model]
        row = {'Model': model}
        
        # Per-Fitzpatrick F1 scores (mean ± std)
        for fitz_idx in range(1, 7):
            f1_col = f'fitz_{fitz_idx}_f1'
            mean_f1 = df[f1_col].mean()
            std_f1 = df[f1_col].std()
            row[f'Fitz-{fitz_idx}'] = f'{mean_f1:.3f} ± {std_f1:.3f}'
        
        # Performance gap statistics
        f1_values = [df[f'fitz_{i}_f1'].mean() for i in range(1, 7)]
        row['Min F1'] = f'{min(f1_values):.3f}'
        row['Max F1'] = f'{max(f1_values):.3f}'
        row['Range'] = f'{max(f1_values) - min(f1_values):.3f}'
        row['Std Dev'] = f'{np.std(f1_values):.3f}'
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    csv_path = save_dir / 'fitzpatrick_summary.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved Fitzpatrick summary table to {csv_path}")
    
    # Save as LaTeX
    latex_path = save_dir / 'fitzpatrick_summary.tex'
    generate_latex_table(summary_df, latex_path)
    
    # Print to console
    print("\n" + "="*120)
    print("PER-FITZPATRICK F1 SCORE SUMMARY")
    print("="*120)
    print(summary_df.to_string(index=False))
    print("="*120)
    
    return summary_df


def compute_group_disparity_ratios(history_dict, metric_name, group_key):
    """Compute disparity ratios for a metric across Fitzpatrick groups.
    
    Returns dict with ratios relative to Fitzpatrick Type 3 (group index 2).
    For zero reference values, uses absolute differences instead of ratios.
    """
    test_data = history_dict.get('test', [{}])[-1]  # Get last epoch test results
    fairness_metrics = test_data.get('fairness_metrics', {})
    
    if metric_name == 'fpr':
        group_values = fairness_metrics.get('equalized_odds', {}).get('group_fpr', {})
    elif metric_name == 'fnr':
        group_values = fairness_metrics.get('equalized_odds', {}).get('group_fnr', {})
    elif metric_name == 'tpr':
        group_values = fairness_metrics.get('equal_opportunity', {}).get('group_tpr', {})
    elif metric_name == 'positive_rate':
        group_values = fairness_metrics.get('demographic_parity', {}).get('group_positive_rates', {})
    elif metric_name == 'precision':
        group_values = fairness_metrics.get('worst_group', {}).get('group_precision', {})
    elif metric_name == 'f1':
        group_values = fairness_metrics.get('worst_group', {}).get('group_f1', {})
    else:
        return None
    
    # Convert string keys to int and get values
    group_dict = {int(k): v for k, v in group_values.items() if isinstance(v, (int, float))}
    
    if not group_dict:
        return None
    
    # Always use Fitzpatrick Type 3 (group index 2) as reference
    ref_group = 2
    
    # Check if reference group exists in data
    if ref_group not in group_dict:
        return None
    
    ref_value = group_dict[ref_group]
    
    # Compute disparity ratios: group_value / reference_value
    # Special handling for zero reference values
    ratios = {}
    for group_id, value in group_dict.items():
        if ref_value > 1e-10:  # Normal case: compute ratio
            ratios[group_id] = value / ref_value
        else:  # Reference is zero or near-zero
            # For error rates (FPR, FNR): if ref=0, any non-zero is bad (ratio = inf concept)
            # For performance (TPR, F1, precision): if ref=0, we're all equally bad
            # Use additive difference + 1.0 to stay in ratio space
            if value > 1e-10:
                # Non-zero value when reference is zero: this is a disparity
                ratios[group_id] = 1.0 + (value - ref_value) * 10  # Scale up small differences
            else:
                # Both zero: no disparity
                ratios[group_id] = 1.0
    
    return ratios, ref_group, ref_value


def plot_disparity_metrics(all_results_df, results_dir, save_dir):
    """Plot Aequitas-style per-group disparity ratios for error rate parity."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    model_types = sorted(all_results_df['model_type'].unique())
    fitz_labels = ['Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5', 'Type 6']
    
    metrics_info = [
        ('fpr', 'False Positive Rate Parity', 'FPR Disparity'),
        ('fnr', 'False Negative Rate Parity', 'FNR Disparity'),
        ('positive_rate', 'Statistical Parity', 'Selection Rate Disparity')
    ]
    
    for plot_idx, (metric_name, title, ylabel) in enumerate(metrics_info):
        ax = axes[plot_idx]
        
        # Collect disparity ratios for each model across runs
        model_ratios = {model: {fitz: [] for fitz in range(6)} for model in model_types}
        reference_groups = []  # Track reference group across runs
        
        # Load individual run results
        for _, row in all_results_df.iterrows():
            model = row['model_type']
            run_id = row['run_id']
            job_dir = row['job_dir']
            
            # Load history.json for this run
            history_path = results_dir / job_dir / f"run_{run_id}" / model / "history.json"
            if history_path.exists():
                import json
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                result = compute_group_disparity_ratios(history, metric_name, 'group')
                if result:
                    ratios, ref_group, ref_value = result
                    reference_groups.append(ref_group)
                    for fitz in range(6):
                        if fitz in ratios:
                            model_ratios[model][fitz].append(ratios[fitz])
        
        # Plot bars for each model (horizontal)
        y = np.arange(len(fitz_labels))
        height = 0.15
        colors_map = {'direct': '#ff7f0e', 'standard_cbm': '#2ca02c', 'curriculum_cbm': '#1f77b4',
                      'fair_cbm': '#9467bd', 'fair_curriculum_cbm': '#d62728'}
        
        # Add fairness threshold zones (Aequitas 80% rule: 0.8 to 1.25)
        ax.axvspan(0.8, 1.25, alpha=0.2, color='green', zorder=0, label='Fair (80% rule)')
        ax.axvspan(0, 0.8, alpha=0.15, color='red', zorder=0)
        ax.axvspan(1.25, 3.0, alpha=0.15, color='red', zorder=0)
        ax.axvline(x=1.0, color='black', linestyle='-', linewidth=1.5, alpha=0.5, label='Parity (ratio=1.0)')
        ax.axvline(x=0.8, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(x=1.25, color='green', linestyle='--', linewidth=2, alpha=0.7)
        
        for model_idx, model in enumerate(model_types):
            means = []
            stds = []
            for fitz in range(6):
                values = model_ratios[model][fitz]
                if values:
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                else:
                    means.append(np.nan)
                    stds.append(0)
            
            offset = height * (model_idx - len(model_types)/2 + 0.5)
            bars = ax.barh(y + offset, means, height, xerr=stds, 
                          label=model.replace('_', ' ').title(),
                          color=colors_map.get(model, '#gray'),
                          alpha=0.8, capsize=3, edgecolor='black', linewidth=0.5)
            
            # Color bars based on fairness (in 0.8-1.25 range)
            for bar, mean_val in zip(bars, means):
                if not np.isnan(mean_val):
                    if 0.8 <= mean_val <= 1.25:
                        bar.set_edgecolor('darkgreen')
                        bar.set_linewidth(2)
                    else:
                        bar.set_edgecolor('darkred')
                        bar.set_linewidth(2)
        
        # Add reference group to title (always Type 3)
        title_with_ref = f'{title}\n(Reference: Type 3)'
        
        ax.set_ylabel('Fitzpatrick Skin Type', fontsize=12, fontweight='bold')
        ax.set_xlabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title_with_ref, fontsize=13, fontweight='bold')
        ax.set_yticks(y)
        ax.set_yticklabels(fitz_labels, fontsize=10)
        ax.set_xlim([0, 3.0])
        ax.legend(loc='upper right', fontsize=8, ncol=1)
        ax.grid(axis='x', alpha=0.3, zorder=0)
    
    plt.suptitle('Aequitas Fairness Audit: Disparity Ratios (Reference = Fitzpatrick Type 3)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / 'aequitas_disparity_parity.png', dpi=300, bbox_inches='tight')
    print(f"Saved Aequitas disparity parity plot to {save_dir / 'aequitas_disparity_parity.png'}")


def plot_impact_metrics(all_results_df, results_dir, save_dir):
    """Plot Aequitas-style per-group disparity ratios for performance parity."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    model_types = sorted(all_results_df['model_type'].unique())
    fitz_labels = ['Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5', 'Type 6']
    
    metrics_info = [
        ('tpr', 'True Positive Rate Parity (Equal Opportunity)', 'TPR Disparity'),
        ('precision', 'Precision Parity (Positive Predictive Value)', 'Precision Disparity'),
        ('f1', 'F1 Score Parity', 'F1 Disparity')
    ]
    
    for plot_idx, (metric_name, title, ylabel) in enumerate(metrics_info):
        ax = axes[plot_idx]
        
        # Collect disparity ratios for each model across runs
        model_ratios = {model: {fitz: [] for fitz in range(6)} for model in model_types}
        reference_groups = []  # Track reference group across runs
        
        # Load individual run results
        for _, row in all_results_df.iterrows():
            model = row['model_type']
            run_id = row['run_id']
            job_dir = row['job_dir']
            
            # Load history.json for this run
            history_path = results_dir / job_dir / f"run_{run_id}" / model / "history.json"
            if history_path.exists():
                import json
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                result = compute_group_disparity_ratios(history, metric_name, 'group')
                if result:
                    ratios, ref_group, ref_value = result
                    reference_groups.append(ref_group)
                    for fitz in range(6):
                        if fitz in ratios:
                            model_ratios[model][fitz].append(ratios[fitz])
        
        # Plot bars for each model (horizontal)
        y = np.arange(len(fitz_labels))
        height = 0.15
        colors_map = {'direct': '#ff7f0e', 'standard_cbm': '#2ca02c', 'curriculum_cbm': '#1f77b4',
                      'fair_cbm': '#9467bd', 'fair_curriculum_cbm': '#d62728'}
        
        # Add fairness threshold zones (Aequitas 80% rule: 0.8 to 1.25)
        ax.axvspan(0.8, 1.25, alpha=0.2, color='green', zorder=0, label='Fair (80% rule)')
        ax.axvspan(0, 0.8, alpha=0.15, color='red', zorder=0)
        ax.axvspan(1.25, 1.5, alpha=0.15, color='red', zorder=0)
        ax.axvline(x=1.0, color='black', linestyle='-', linewidth=1.5, alpha=0.5, label='Parity (ratio=1.0)')
        ax.axvline(x=0.8, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(x=1.25, color='green', linestyle='--', linewidth=2, alpha=0.7)
        
        for model_idx, model in enumerate(model_types):
            means = []
            stds = []
            for fitz in range(6):
                values = model_ratios[model][fitz]
                if values:
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                else:
                    means.append(np.nan)
                    stds.append(0)
            
            offset = height * (model_idx - len(model_types)/2 + 0.5)
            bars = ax.barh(y + offset, means, height, xerr=stds, 
                          label=model.replace('_', ' ').title(),
                          color=colors_map.get(model, '#gray'),
                          alpha=0.8, capsize=3, edgecolor='black', linewidth=0.5)
            
            # Color bars based on fairness (in 0.8-1.25 range)
            for bar, mean_val in zip(bars, means):
                if not np.isnan(mean_val):
                    if 0.8 <= mean_val <= 1.25:
                        bar.set_edgecolor('darkgreen')
                        bar.set_linewidth(2)
                    else:
                        bar.set_edgecolor('darkred')
                        bar.set_linewidth(2)
        
        # Add reference group to title (always Type 3)
        title_with_ref = f'{title}\n(Reference: Type 3)'
        
        # Add reference group to title (always Type 3)
        title_with_ref = f'{title}\n(Reference: Type 3)'
        
        ax.set_ylabel('Fitzpatrick Skin Type', fontsize=12, fontweight='bold')
        ax.set_xlabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title_with_ref, fontsize=13, fontweight='bold')
        ax.set_yticks(y)
        ax.set_yticklabels(fitz_labels, fontsize=10)
        ax.set_xlim([0, 1.5])
        ax.legend(loc='upper right', fontsize=8, ncol=1)
        ax.grid(axis='x', alpha=0.3, zorder=0)
    
    plt.suptitle('Aequitas Fairness Audit: Performance Parity (Reference = Fitzpatrick Type 3)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / 'aequitas_performance_parity.png', dpi=300, bbox_inches='tight')
    print(f"Saved Aequitas performance parity plot to {save_dir / 'aequitas_performance_parity.png'}")


def plot_calibration_metrics(all_results_df, results_dir, save_dir):
    """Plot Aequitas-style per-group calibration error comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    model_types = sorted(all_results_df['model_type'].unique())
    fitz_labels = ['Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5', 'Type 6']
    
    # Calibration metrics: show absolute ECE per group
    metrics_info = [
        ('ece', 'Expected Calibration Error by Group', 'ECE'),
        ('ece_disparity', 'ECE Disparity Ratio', 'ECE Ratio'),
        ('overall_cal', 'Overall Calibration Disparity', 'Max ECE Difference')
    ]
    
    for plot_idx, (metric_name, title, ylabel) in enumerate(metrics_info):
        ax = axes[plot_idx]
        
        if metric not in all_results_df.columns or all_results_df[metric].notna().sum() == 0:
            ax.text(0.5, 0.5, f'{metric}\nnot available', ha='center', va='center',
                   fontsize=12, transform=ax.transAxes)
            ax.set_title(ylabel, fontsize=11, fontweight='bold')
            ax.axis('off')
            continue
        
        # Prepare data
        data_to_plot = []
        labels = []
        pass_fail = []
        
        for model in model_types:
            values = all_results_df[all_results_df['model_type'] == model][metric].dropna().values
            if len(values) > 0:
                data_to_plot.append(values)
                labels.append(model.replace('_', '\n'))
                mean_val = np.mean(values)
                passes = mean_val <= thresh_good
                pass_fail.append(passes)
        
        if data_to_plot:
            # Add colored background zones
            ylims = [min([min(d) for d in data_to_plot]), max([max(d) for d in data_to_plot])]
            y_range = ylims[1] - ylims[0]
            ylims[0] = max(0, ylims[0] - 0.1 * y_range)
            ylims[1] = ylims[1] + 0.15 * y_range
            
            # Lower is better
            ax.axhspan(0, thresh_good, alpha=0.15, color='green', zorder=0)
            ax.axhspan(thresh_good, thresh_bad, alpha=0.15, color='yellow', zorder=0)
            ax.axhspan(thresh_bad, ylims[1], alpha=0.15, color='red', zorder=0)
            ax.axhline(y=thresh_good, color='green', linestyle='--', linewidth=2.5, 
                      label=f'Fair ≤ {thresh_good}', alpha=0.8)
            ax.axhline(y=thresh_bad, color='orange', linestyle='--', linewidth=2, 
                      label=f'High bias ≥ {thresh_bad}', alpha=0.7)
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                           showmeans=True, meanline=True)
            
            # Color boxes based on pass/fail
            for patch, passes in zip(bp['boxes'], pass_fail):
                if passes:
                    patch.set_facecolor('lightgreen')
                    patch.set_edgecolor('darkgreen')
                    patch.set_linewidth(2.5)
                else:
                    patch.set_facecolor('lightcoral')
                    patch.set_edgecolor('darkred')
                    patch.set_linewidth(2.5)
                patch.set_alpha(0.7)
            
            ax.set_ylim(ylims)
        
        ax.set_xticklabels(labels, rotation=0, fontsize=9, ha='center')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{ylabel}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, zorder=0)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'calibration_fairness_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Saved calibration fairness metrics to {save_dir / 'calibration_fairness_metrics.png'}")


def main():
    parser = argparse.ArgumentParser(description='Analyze multi-run fairness experiments')
    
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory with results (searches all multi_run_* subdirectories)')
    parser.add_argument('--backbone', type=str, default='swin',
                        help='Backbone architecture (for reference only)')
    parser.add_argument('--n_runs', type=int, default=100,
                        help='Expected number of runs to analyze')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for analysis (default: results/analysis)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Create analysis directory
    if args.output_dir:
        analysis_dir = Path(args.output_dir)
    else:
        analysis_dir = results_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("MULTI-RUN FAIRNESS ANALYSIS")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Backbone: {args.backbone}")
    print(f"Expected runs: {args.n_runs}")
    print(f"Analysis output: {analysis_dir}")
    print("")
    
    # Model types
    model_types = ['direct', 'standard_cbm', 'curriculum_cbm', 'fair_cbm', 'fair_curriculum_cbm']
    
    # Load results for all models
    print("Loading results...")
    all_results = []
    results_dict = {}
    
    for model_type in model_types:
        print(f"\nLoading {model_type}...")
        df = load_run_results(results_dir, model_type, args.n_runs)
        
        if len(df) == 0:
            print(f"  Warning: No results found for {model_type}")
            continue
        
        all_results.append(df)
        results_dict[model_type] = df
    
    if not all_results:
        print("\nERROR: No results found for any model type!")
        print("Please check that training has completed and history.json files exist.")
        return
    
    all_results_df = pd.concat(all_results, ignore_index=True)
    print(f"\nTotal results loaded: {len(all_results_df)}")
    
    # Metrics to analyze
    standard_metrics = ['f1', 'accuracy', 'precision', 'recall', 'auc', 'sensitivity', 'specificity']
    fairness_metrics = ['demographic_parity', 'disparate_impact_ratio', 'statistical_parity_diff',
                       'equalized_odds_tpr', 'equalized_odds_fpr', 'equalized_odds_diff',
                       'equal_opportunity_diff', 'equal_opportunity_tpr',
                       'performance_gap', 'performance_gap_ratio',
                       'worst_group_f1', 'best_group_f1',
                       'calibration_disparity', 'mean_ece']
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
    viz_metrics = ['f1', 'accuracy', 'performance_gap', 'worst_group_f1', 
                   'demographic_parity', 'equalized_odds_diff', 'calibration_disparity']
    
    plot_metric_distributions(all_results_df, viz_metrics, analysis_dir)
    plot_fairness_impact_comparison(all_results_df, analysis_dir)
    
    # Individual metric comparisons
    for metric in ['f1', 'performance_gap', 'worst_group_f1', 'demographic_parity']:
        if metric in all_results_df.columns and all_results_df[metric].notna().sum() > 0:
            # Check if metric exists in summary_dict for at least one model
            if any(metric in summary_dict.get(m, {}) for m in summary_dict):
                plot_pairwise_comparison(summary_dict, metric, analysis_dir)
    
    # Per-Fitzpatrick visualizations
    print("\n" + "="*80)
    print("GENERATING PER-FITZPATRICK VISUALIZATIONS")
    print("="*80)
    fitzpatrick_summary = create_fitzpatrick_summary_table(all_results_df, analysis_dir)
    plot_per_fitzpatrick_performance(all_results_df, analysis_dir)
    plot_fitzpatrick_trends(all_results_df, analysis_dir)
    plot_fairness_improvement_heatmap(all_results_df, analysis_dir)
    
    # Aequitas fairness audit plots
    print("\n" + "="*80)
    print("GENERATING AEQUITAS FAIRNESS AUDIT PLOTS")
    print("="*80)
    plot_disparity_metrics(all_results_df, results_dir, analysis_dir)
    plot_impact_metrics(all_results_df, results_dir, analysis_dir)
    # plot_calibration_metrics(all_results_df, results_dir, analysis_dir)  # TODO: Fix calibration plot
    
    # Save raw results
    all_results_df.to_csv(analysis_dir / 'all_results.csv', index=False)
    
    # Generate final report
    report = {
        'results_directory': str(results_dir),
        'backbone': args.backbone,
        'expected_runs': args.n_runs,
        'actual_runs': {k: len(v) for k, v in results_dict.items()},
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
