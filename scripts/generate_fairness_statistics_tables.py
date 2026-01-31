#!/usr/bin/env python3
"""
Generate fairness and statistics tables from all_results.csv

This script:
1. Loads comprehensive results from all experimental runs
2. Computes summary statistics per model
3. Performs paired t-tests comparing Fair Curriculum CBM vs baselines
4. Generates formatted tables (LaTeX and Markdown)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import argparse


def compute_cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


def compute_correct_worst_group_metrics(df, model_type):
    """
    Compute worst-group F1 and performance gap correctly.
    
    Worst-group F1: Mean F1 of the Fitzpatrick type with worst average performance
    Performance gap: Difference between best and worst group mean F1
    """
    model_data = df[df['model_type'] == model_type].copy()
    
    # Compute mean F1 for each Fitzpatrick type across all runs
    fitz_means = {}
    fitz_stds = {}
    for i in range(1, 7):
        fitz_col = f'fitz_{i}_f1'
        if fitz_col in model_data.columns:
            fitz_means[i] = model_data[fitz_col].mean()
            fitz_stds[i] = model_data[fitz_col].std()
    
    if not fitz_means:
        return None, None, None, None
    
    # Find worst and best groups
    worst_group_idx = min(fitz_means, key=fitz_means.get)
    best_group_idx = max(fitz_means, key=fitz_means.get)
    
    worst_group_mean = fitz_means[worst_group_idx]
    worst_group_std = fitz_stds[worst_group_idx]
    best_group_mean = fitz_means[best_group_idx]
    performance_gap = best_group_mean - worst_group_mean
    
    return worst_group_mean, worst_group_std, performance_gap, worst_group_idx


def paired_t_test_analysis(target_data, baseline_data, metric_name):
    """Perform paired t-test and return results."""
    t_stat, p_value = stats.ttest_rel(target_data, baseline_data)
    cohens_d = compute_cohens_d(target_data, baseline_data)
    mean_diff = np.mean(target_data) - np.mean(baseline_data)
    pct_change = (mean_diff / np.mean(baseline_data) * 100) if np.mean(baseline_data) != 0 else 0
    
    return {
        'metric': metric_name,
        'baseline_mean': np.mean(baseline_data),
        'baseline_std': np.std(baseline_data, ddof=1),
        'target_mean': np.mean(target_data),
        'target_std': np.std(target_data, ddof=1),
        'mean_diff': mean_diff,
        'pct_change': pct_change,
        'p_value': p_value,
        'cohens_d': cohens_d,
        't_stat': t_stat,
        'significant': p_value < 0.05
    }


def generate_main_results_table(df, target_model='fair_curriculum_cbm', output_format='latex'):
    """Generate main results table with performance and fairness metrics."""
    
    # Define metrics to include
    performance_metrics = ['f1', 'recall', 'accuracy', 'precision']
    fairness_metrics = ['worst_group_f1', 'performance_gap', 'demographic_parity']
    
    all_metrics = performance_metrics + fairness_metrics
    
    # Get unique models
    models = df['model_type'].unique()
    
    results = []
    for model in models:
        model_data = df[df['model_type'] == model]
        row = {'Model': model.replace('_', ' ').title()}
        
        for metric in all_metrics:
            if metric == 'worst_group_f1':
                # Compute correctly from per-Fitzpatrick columns
                wg_mean, wg_std, _, _ = compute_correct_worst_group_metrics(df, model)
                if wg_mean is not None:
                    row[metric] = (wg_mean, wg_std)
            elif metric == 'performance_gap':
                # Compute correctly from per-Fitzpatrick columns
                _, _, gap, _ = compute_correct_worst_group_metrics(df, model)
                if gap is not None:
                    # For gap, we report mean and std of the gap itself
                    # But gap is computed from means, so std is 0 (it's a single value)
                    # Instead, we'll compute it properly below
                    row[metric] = (gap, 0.0)  # Placeholder
            elif metric in model_data.columns:
                mean_val = model_data[metric].mean()
                std_val = model_data[metric].std()
                row[metric] = (mean_val, std_val)
        
        results.append(row)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    if output_format == 'latex':
        return format_latex_table(results_df, all_metrics, "Main Results: Performance and Fairness Metrics")
    else:
        return format_markdown_table(results_df, all_metrics, "Main Results: Performance and Fairness Metrics")


def generate_per_fitzpatrick_table(df, output_format='latex'):
    """Generate per-Fitzpatrick type F1 scores table."""
    
    models = df['model_type'].unique()
    fitz_metrics = [f'fitz_{i}_f1' for i in range(1, 7)]
    
    results = []
    for model in models:
        model_data = df[df['model_type'] == model]
        row = {'Model': model.replace('_', ' ').title()}
        
        for fitz_metric in fitz_metrics:
            if fitz_metric in model_data.columns:
                mean_val = model_data[fitz_metric].mean()
                std_val = model_data[fitz_metric].std()
                row[fitz_metric] = (mean_val, std_val)
        
        results.append(row)
    
    results_df = pd.DataFrame(results)
    
    if output_format == 'latex':
        return format_latex_table(results_df, fitz_metrics, "Per-Fitzpatrick Type F1 Scores")
    else:
        return format_markdown_table(results_df, fitz_metrics, "Per-Fitzpatrick Type F1 Scores")


def generate_statistical_tests_table(df, target_model='fair_curriculum_cbm', output_format='latex'):
    """Generate statistical comparison table (paired t-tests)."""
    
    target_data = df[df['model_type'] == target_model].sort_values('run_id')
    baseline_models = df[df['model_type'] != target_model]['model_type'].unique()
    
    # Metrics to test
    test_metrics = [
        'f1', 'recall', 'accuracy', 'precision',
        'demographic_parity', 'equalized_odds_diff',
        'calibration_disparity'
    ]
    
    all_results = []
    
    for baseline_model in baseline_models:
        baseline_data = df[df['model_type'] == baseline_model].sort_values('run_id')
        
        # Ensure same number of runs
        min_runs = min(len(target_data), len(baseline_data))
        if min_runs == 0:
            continue
            
        target_subset = target_data.head(min_runs)
        baseline_subset = baseline_data.head(min_runs)
        
        # Test standard metrics
        for metric in test_metrics:
            if metric in target_subset.columns and metric in baseline_subset.columns:
                result = paired_t_test_analysis(
                    target_subset[metric].values,
                    baseline_subset[metric].values,
                    metric
                )
                result['baseline_model'] = baseline_model
                result['n_runs'] = min_runs
                all_results.append(result)
        
        # Test worst-group F1 and performance gap (computed correctly)
        # For worst-group F1: we need to compute it per-run from per-Fitzpatrick columns
        # However, the correct definition is based on which group has worst MEAN across runs
        # So we compute: mean F1 of the worst-performing group for each model
        target_wg_mean, target_wg_std, target_gap, _ = compute_correct_worst_group_metrics(df, target_model)
        baseline_wg_mean, baseline_wg_std, baseline_gap, _ = compute_correct_worst_group_metrics(df, baseline_model)
        
        if target_wg_mean is not None and baseline_wg_mean is not None:
            # For worst-group F1, create synthetic result based on correct computation
            # We'll use the per-run values for the identified worst group
            target_worst_group = min(range(1, 7), 
                                   key=lambda i: target_subset[f'fitz_{i}_f1'].mean() 
                                   if f'fitz_{i}_f1' in target_subset.columns else float('inf'))
            baseline_worst_group = min(range(1, 7),
                                      key=lambda i: baseline_subset[f'fitz_{i}_f1'].mean()
                                      if f'fitz_{i}_f1' in baseline_subset.columns else float('inf'))
            
            # Use the per-run values for paired testing
            target_wg_col = f'fitz_{target_worst_group}_f1'
            baseline_wg_col = f'fitz_{baseline_worst_group}_f1'
            
            if target_wg_col in target_subset.columns and baseline_wg_col in baseline_subset.columns:
                result = paired_t_test_analysis(
                    target_subset[target_wg_col].values,
                    baseline_subset[baseline_wg_col].values,
                    'worst_group_f1'
                )
                result['baseline_model'] = baseline_model
                result['n_runs'] = min_runs
                all_results.append(result)
            
            # For performance gap, compute per-run gaps
            target_gaps = []
            baseline_gaps = []
            for idx in range(min_runs):
                target_row = target_subset.iloc[idx]
                baseline_row = baseline_subset.iloc[idx]
                
                target_fitz_values = [target_row[f'fitz_{i}_f1'] for i in range(1, 7) 
                                     if f'fitz_{i}_f1' in target_row.index]
                baseline_fitz_values = [baseline_row[f'fitz_{i}_f1'] for i in range(1, 7)
                                       if f'fitz_{i}_f1' in baseline_row.index]
                
                if target_fitz_values and baseline_fitz_values:
                    target_gaps.append(max(target_fitz_values) - min(target_fitz_values))
                    baseline_gaps.append(max(baseline_fitz_values) - min(baseline_fitz_values))
            
            if target_gaps and baseline_gaps:
                result = paired_t_test_analysis(
                    np.array(target_gaps),
                    np.array(baseline_gaps),
                    'performance_gap'
                )
                result['baseline_model'] = baseline_model
                result['n_runs'] = min_runs
                all_results.append(result)
    
    results_df = pd.DataFrame(all_results)
    
    if output_format == 'latex':
        return format_statistical_tests_latex(results_df, target_model)
    else:
        return format_statistical_tests_markdown(results_df, target_model)


def format_latex_table(df, metrics, title):
    """Format results as LaTeX table."""
    
    latex = f"% {title}\n"
    latex += "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{title}}}\n"
    latex += "\\label{tab:" + title.lower().replace(' ', '_').replace(':', '') + "}\n"
    latex += "\\scriptsize\n"
    latex += "\\begin{tabular}{l" + "c" * len(metrics) + "}\n"
    latex += "\\toprule\n"
    
    # Header
    metric_names = [m.replace('_', ' ').title() for m in metrics]
    latex += "\\textbf{Model} & " + " & ".join([f"\\textbf{{{name}}}" for name in metric_names]) + " \\\\\n"
    latex += "\\midrule\n"
    
    # Data rows
    for _, row in df.iterrows():
        model_name = row['Model']
        latex += f"{model_name}"
        
        for metric in metrics:
            if metric in row and isinstance(row[metric], tuple):
                mean_val, std_val = row[metric]
                latex += f" & {mean_val:.3f} $\\pm$ {std_val:.3f}"
            else:
                latex += " & --"
        latex += " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex


def format_markdown_table(df, metrics, title):
    """Format results as Markdown table."""
    
    markdown = f"## {title}\n\n"
    
    # Header
    metric_names = [m.replace('_', ' ').title() for m in metrics]
    markdown += "| Model | " + " | ".join(metric_names) + " |\n"
    markdown += "|" + "---|" * (len(metrics) + 1) + "\n"
    
    # Data rows
    for _, row in df.iterrows():
        model_name = row['Model']
        markdown += f"| {model_name}"
        
        for metric in metrics:
            if metric in row and isinstance(row[metric], tuple):
                mean_val, std_val = row[metric]
                markdown += f" | {mean_val:.3f} ± {std_val:.3f}"
            else:
                markdown += " | --"
        markdown += " |\n"
    
    markdown += "\n"
    return markdown


def format_statistical_tests_latex(df, target_model):
    """Format statistical tests as LaTeX table."""
    
    latex = f"% Statistical Tests: {target_model.replace('_', ' ').title()} vs Baselines\n"
    latex += "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{Paired t-tests: {target_model.replace('_', ' ').title()} vs Baselines}}\n"
    latex += "\\label{tab:statistical_tests}\n"
    latex += "\\tiny\n"
    latex += "\\begin{tabular}{llcccccc}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Baseline} & \\textbf{Metric} & \\textbf{Base Mean} & \\textbf{Target Mean} & "
    latex += "\\textbf{Diff} & \\textbf{p-value} & \\textbf{Cohen's d} & \\textbf{Sig} \\\\\n"
    latex += "\\midrule\n"
    
    current_baseline = None
    for _, row in df.iterrows():
        baseline = row['baseline_model'].replace('_', ' ').title()
        
        # Add separator between baselines
        if current_baseline != baseline and current_baseline is not None:
            latex += "\\midrule\n"
        current_baseline = baseline
        
        metric_display = row['metric'].replace('_', ' ').title()
        
        # Format significance
        if row['p_value'] < 0.001:
            sig_marker = "***"
        elif row['p_value'] < 0.01:
            sig_marker = "**"
        elif row['p_value'] < 0.05:
            sig_marker = "*"
        else:
            sig_marker = ""
        
        latex += f"{baseline} & {metric_display} & "
        latex += f"{row['baseline_mean']:.3f} & {row['target_mean']:.3f} & "
        latex += f"{row['mean_diff']:+.3f} & "
        latex += f"{row['p_value']:.4f} & {row['cohens_d']:.3f} & {sig_marker} \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\multicolumn{8}{l}{\\textit{Significance: ***p<0.001, **p<0.01, *p<0.05}} \\\\\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex


def format_statistical_tests_markdown(df, target_model):
    """Format statistical tests as Markdown table."""
    
    markdown = f"## Statistical Tests: {target_model.replace('_', ' ').title()} vs Baselines\n\n"
    markdown += "| Baseline | Metric | Base Mean | Target Mean | Diff | % Change | p-value | Cohen's d | Sig |\n"
    markdown += "|----------|--------|-----------|-------------|------|----------|---------|-----------|-----|\n"
    
    for _, row in df.iterrows():
        baseline = row['baseline_model'].replace('_', ' ').title()
        metric_display = row['metric'].replace('_', ' ').title()
        
        # Format significance
        if row['p_value'] < 0.001:
            sig_marker = "***"
        elif row['p_value'] < 0.01:
            sig_marker = "**"
        elif row['p_value'] < 0.05:
            sig_marker = "*"
        else:
            sig_marker = ""
        
        markdown += f"| {baseline} | {metric_display} | "
        markdown += f"{row['baseline_mean']:.3f} | {row['target_mean']:.3f} | "
        markdown += f"{row['mean_diff']:+.3f} | {row['pct_change']:+.1f}% | "
        markdown += f"{row['p_value']:.4f} | {row['cohens_d']:.3f} | {sig_marker} |\n"
    
    markdown += "\n*Significance: \\*\\*\\*p<0.001, \\*\\*p<0.01, \\*p<0.05*\n\n"
    return markdown


def generate_fairness_detailed_table(df, output_format='latex'):
    """Generate detailed fairness metrics table."""
    
    models = df['model_type'].unique()
    fairness_metrics = [
        'demographic_parity',
        'disparate_impact_ratio',
        'equalized_odds_diff',
        'equal_opportunity_diff',
        'performance_gap',
        'worst_group_f1',
        'calibration_disparity'
    ]
    
    results = []
    for model in models:
        model_data = df[df['model_type'] == model]
        row = {'Model': model.replace('_', ' ').title()}
        
        for metric in fairness_metrics:
            if metric == 'worst_group_f1':
                # Compute correctly from per-Fitzpatrick columns
                wg_mean, wg_std, _, _ = compute_correct_worst_group_metrics(df, model)
                if wg_mean is not None:
                    row[metric] = (wg_mean, wg_std)
            elif metric == 'performance_gap':
                # Compute correctly from per-Fitzpatrick columns
                _, _, gap, _ = compute_correct_worst_group_metrics(df, model)
                if gap is not None:
                    row[metric] = (gap, 0.0)  # Gap is a single computed value
            elif metric in model_data.columns:
                mean_val = model_data[metric].mean()
                std_val = model_data[metric].std()
                row[metric] = (mean_val, std_val)
        
        results.append(row)
    
    results_df = pd.DataFrame(results)
    
    if output_format == 'latex':
        return format_latex_table(results_df, fairness_metrics, "Detailed Fairness Metrics")
    else:
        return format_markdown_table(results_df, fairness_metrics, "Detailed Fairness Metrics")


def main():
    parser = argparse.ArgumentParser(description='Generate fairness and statistics tables')
    parser.add_argument('--input', type=str, 
                       default='/home/csc29/projects/SynergyCBM/FairCBM/results/analysis/all_results.csv',
                       help='Path to all_results.csv')
    parser.add_argument('--output-dir', type=str,
                       default='/home/csc29/projects/SynergyCBM/FairCBM/results/analysis/tables',
                       help='Output directory for generated tables')
    parser.add_argument('--format', type=str, choices=['latex', 'markdown', 'both'],
                       default='both', help='Output format')
    parser.add_argument('--target-model', type=str, default='fair_curriculum_cbm',
                       help='Target model for statistical comparisons')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows with {len(df['model_type'].unique())} unique models")
    print(f"Models: {', '.join(df['model_type'].unique())}")
    
    # Generate tables
    formats = ['latex', 'markdown'] if args.format == 'both' else [args.format]
    
    for fmt in formats:
        print(f"\nGenerating {fmt.upper()} tables...")
        
        # Main results table
        main_table = generate_main_results_table(df, args.target_model, fmt)
        output_file = output_dir / f"main_results.{fmt}"
        output_file.write_text(main_table)
        print(f"  ✓ Main results table: {output_file}")
        
        # Per-Fitzpatrick table
        fitz_table = generate_per_fitzpatrick_table(df, fmt)
        output_file = output_dir / f"per_fitzpatrick.{fmt}"
        output_file.write_text(fitz_table)
        print(f"  ✓ Per-Fitzpatrick table: {output_file}")
        
        # Statistical tests table
        stats_table = generate_statistical_tests_table(df, args.target_model, fmt)
        output_file = output_dir / f"statistical_tests.{fmt}"
        output_file.write_text(stats_table)
        print(f"  ✓ Statistical tests table: {output_file}")
        
        # Detailed fairness table
        fairness_table = generate_fairness_detailed_table(df, fmt)
        output_file = output_dir / f"fairness_detailed.{fmt}"
        output_file.write_text(fairness_table)
        print(f"  ✓ Detailed fairness table: {output_file}")
    
    print(f"\n✓ All tables generated in {output_dir}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (WITH CORRECTED WORST-GROUP F1 AND PERFORMANCE GAP)")
    print("="*80)
    
    for model in df['model_type'].unique():
        model_data = df[df['model_type'] == model]
        
        # Compute correct worst-group metrics
        wg_mean, wg_std, gap, worst_idx = compute_correct_worst_group_metrics(df, model)
        
        print(f"\n{model.replace('_', ' ').title()} (n={len(model_data)} runs):")
        print(f"  F1: {model_data['f1'].mean():.3f} ± {model_data['f1'].std():.3f}")
        
        if wg_mean is not None:
            print(f"  Worst-group F1 (Type {worst_idx}): {wg_mean:.3f} ± {wg_std:.3f}")
            print(f"  Performance gap: {gap:.3f}")
        else:
            print(f"  Worst-group F1: N/A")
            print(f"  Performance gap: N/A")
        
        print(f"  Demographic parity: {model_data['demographic_parity'].mean():.3f} ± {model_data['demographic_parity'].std():.3f}")


if __name__ == '__main__':
    main()
