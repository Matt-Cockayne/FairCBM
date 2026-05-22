"""
Comprehensive Fairness Analysis for Main Experiment Results

This script provides detailed analysis of the 20-run main experiment to understand:
1. Performance gap variability and what's driving it
2. Per-group performance patterns across all models
3. Statistical significance of fairness improvements
4. Detailed breakdown for paper-ready results

Usage:
    python scripts/comprehensive_fairness_analysis.py --results_dir results/history --output_dir results/detailed_analysis
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_all_results(results_dir):
    """Load all results from history directory structure."""
    results_dir = Path(results_dir)
    all_data = defaultdict(list)
    
    model_types = ['direct', 'standard_cbm', 'curriculum_cbm', 'fair_standard_cbm', 'fair_curriculum_cbm']
    
    print("Loading results from history directory...")
    for model_type in model_types:
        print(f"\n  Loading {model_type}...")
        runs_found = 0
        
        # Iterate through all run directories
        for run_dir in sorted(results_dir.glob("run_*")):
            run_id = int(run_dir.name.split("_")[1])
            history_file = run_dir / model_type / "history.json"
            
            if not history_file.exists():
                continue
            
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                if 'test' not in history or len(history['test']) == 0:
                    print(f"    Warning: No test results in {history_file}")
                    continue
                
                test_result = history['test'][0]
                
                # Extract all metrics
                result = {
                    'run_id': run_id,
                    'model_type': model_type,
                }
                
                # Binary metrics
                if 'binary_metrics' in test_result:
                    result.update(test_result['binary_metrics'])
                
                # Flatten fairness metrics
                if 'binary_fairness' in test_result:
                    fairness = test_result['binary_fairness']
                    
                    # Demographic parity
                    if 'demographic_parity' in fairness:
                        dp = fairness['demographic_parity']
                        result['demographic_parity'] = dp.get('max_disparity', dp.get('disparity', np.nan))
                        result['disparate_impact_ratio'] = dp.get('disparate_impact_ratio', np.nan)
                    
                    # Equalized odds
                    if 'equalized_odds' in fairness:
                        eo = fairness['equalized_odds']
                        result['equalized_odds_tpr'] = eo.get('tpr_disparity', np.nan)
                        result['equalized_odds_fpr'] = eo.get('fpr_disparity', np.nan)
                        result['equalized_odds_diff'] = eo.get('equalized_odds_difference', np.nan)
                    
                    # Equal opportunity
                    if 'equal_opportunity' in fairness:
                        eop = fairness['equal_opportunity']
                        result['equal_opportunity_diff'] = eop.get('tpr_disparity', np.nan)
                    
                    # Worst group performance
                    if 'worst_group' in fairness:
                        wg = fairness['worst_group']
                        result['performance_gap'] = wg.get('performance_gap', np.nan)
                        result['worst_group_f1'] = wg.get('worst_group_f1', np.nan)
                        result['best_group_f1'] = wg.get('best_group_f1', np.nan)
                        
                        # Per-group F1 scores
                        if 'group_f1' in wg:
                            for group_id, f1_score in wg['group_f1'].items():
                                result[f'group_{group_id}_f1'] = f1_score
                    
                    # Calibration
                    if 'calibration' in fairness:
                        cal = fairness['calibration']
                        if 'group_ece' in cal:
                            group_eces = [v for v in cal['group_ece'].values() if v is not None and not np.isnan(v)]
                            if group_eces:
                                result['mean_ece'] = np.mean(group_eces)
                                result['calibration_disparity'] = max(group_eces) - min(group_eces)
                
                all_data[model_type].append(result)
                runs_found += 1
                
            except Exception as e:
                print(f"    Error loading {history_file}: {e}")
                continue
        
        print(f"    Found {runs_found} runs")
    
    # Convert to DataFrames
    dfs = {}
    for model_type, data in all_data.items():
        if data:
            dfs[model_type] = pd.DataFrame(data)
    
    return dfs


def analyze_performance_gap_drivers(dfs, output_dir):
    """Analyze what's driving the performance gap in each model."""
    print("\n" + "="*80)
    print("PERFORMANCE GAP ANALYSIS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    model_labels = {
        'direct': 'Direct',
        'standard_cbm': 'Standard CBM',
        'curriculum_cbm': 'Curriculum CBM',
        'fair_standard_cbm': 'Fair Standard CBM',
        'fair_curriculum_cbm': 'Fair Curriculum CBM'
    }
    
    summary_data = []
    
    for idx, (model_type, df) in enumerate(dfs.items()):
        ax = axes[idx]
        
        # Extract per-group F1 scores
        group_cols = [col for col in df.columns if col.startswith('group_') and col.endswith('_f1')]
        
        if not group_cols:
            print(f"\n{model_labels[model_type]}: No per-group data available")
            continue
        
        # Create data for box plot
        group_data = []
        group_labels = []
        for col in sorted(group_cols):
            group_id = col.split('_')[1]
            group_data.append(df[col].dropna().values)
            group_labels.append(f'Fitz-{int(group_id)+1}')
        
        # Box plot
        bp = ax.boxplot(group_data, labels=group_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        # Add mean line
        means = [np.mean(data) for data in group_data]
        ax.plot(range(1, len(means)+1), means, 'ro-', linewidth=2, markersize=8, label='Mean')
        
        # Styling
        ax.set_title(f'{model_labels[model_type]}', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=11)
        ax.set_xlabel('Fitzpatrick Type', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        
        # Calculate statistics
        print(f"\n{model_labels[model_type]}:")
        print(f"  Performance Gap: {df['performance_gap'].mean():.3f} ± {df['performance_gap'].std():.3f}")
        print(f"  Worst Group F1:  {df['worst_group_f1'].mean():.3f} ± {df['worst_group_f1'].std():.3f}")
        print(f"  Per-Group F1 (mean ± std):")
        
        group_stats = {}
        for col, label in zip(sorted(group_cols), group_labels):
            values = df[col].dropna()
            mean_f1 = values.mean()
            std_f1 = values.std()
            # Count how many runs had F1=0 for this group
            zero_count = (values == 0).sum()
            print(f"    {label}: {mean_f1:.3f} ± {std_f1:.3f} (zeros: {zero_count}/{len(values)})")
            group_stats[label] = {'mean': mean_f1, 'std': std_f1, 'zeros': zero_count}
        
        summary_data.append({
            'Model': model_labels[model_type],
            'Perf Gap': f"{df['performance_gap'].mean():.3f} ± {df['performance_gap'].std():.3f}",
            'Worst F1': f"{df['worst_group_f1'].mean():.3f} ± {df['worst_group_f1'].std():.3f}",
            **{f'{label} F1': f"{stats['mean']:.3f}" for label, stats in group_stats.items()}
        })
    
    # Remove empty subplot
    if len(dfs) < 6:
        fig.delaxes(axes[5])
    
    plt.suptitle('Per-Group F1 Score Distribution Across All Runs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_gap_drivers.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to {output_dir / 'performance_gap_drivers.png'}")
    
    # Save summary table
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'per_group_summary.csv', index=False)
    print(f"Saved summary to {output_dir / 'per_group_summary.csv'}")
    
    return summary_df


def analyze_zero_f1_patterns(dfs, output_dir):
    """Analyze patterns of F1=0 occurrences across groups."""
    print("\n" + "="*80)
    print("ZERO F1 PATTERN ANALYSIS")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    model_labels = {
        'direct': 'Direct',
        'standard_cbm': 'Standard CBM',
        'curriculum_cbm': 'Curriculum CBM',
        'fair_standard_cbm': 'Fair Standard CBM',
        'fair_curriculum_cbm': 'Fair Curriculum CBM'
    }
    
    # Count zero F1 occurrences per group per model
    zero_counts = defaultdict(lambda: defaultdict(int))
    total_runs = defaultdict(int)
    
    for model_type, df in dfs.items():
        group_cols = [col for col in df.columns if col.startswith('group_') and col.endswith('_f1')]
        total_runs[model_type] = len(df)
        
        for col in group_cols:
            group_id = int(col.split('_')[1])
            zeros = (df[col] == 0).sum()
            zero_counts[model_type][group_id] = zeros
    
    # Create heatmap data
    models = list(model_labels.keys())
    groups = range(6)
    heatmap_data = np.zeros((len(models), len(groups)))
    
    for i, model in enumerate(models):
        for j, group in enumerate(groups):
            if model in zero_counts:
                heatmap_data[i, j] = zero_counts[model].get(group, 0)
    
    # Heatmap: Count of zeros
    im1 = axes[0].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    axes[0].set_xticks(range(6))
    axes[0].set_xticklabels([f'Fitz-{i+1}' for i in range(6)])
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels([model_labels[m] for m in models])
    axes[0].set_title('Number of Runs with F1=0 per Group', fontweight='bold')
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(groups)):
            text = axes[0].text(j, i, f'{int(heatmap_data[i, j])}',
                               ha="center", va="center", color="black" if heatmap_data[i, j] < 10 else "white")
    
    plt.colorbar(im1, ax=axes[0], label='Number of Runs (out of 20)')
    
    # Heatmap: Percentage
    heatmap_pct = np.zeros((len(models), len(groups)))
    for i, model in enumerate(models):
        for j, group in enumerate(groups):
            if model in total_runs and total_runs[model] > 0:
                heatmap_pct[i, j] = (heatmap_data[i, j] / total_runs[model]) * 100
    
    im2 = axes[1].imshow(heatmap_pct, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    axes[1].set_xticks(range(6))
    axes[1].set_xticklabels([f'Fitz-{i+1}' for i in range(6)])
    axes[1].set_yticks(range(len(models)))
    axes[1].set_yticklabels([model_labels[m] for m in models])
    axes[1].set_title('Percentage of Runs with F1=0 per Group', fontweight='bold')
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(groups)):
            text = axes[1].text(j, i, f'{heatmap_pct[i, j]:.0f}%',
                               ha="center", va="center", color="black" if heatmap_pct[i, j] < 50 else "white")
    
    plt.colorbar(im2, ax=axes[1], label='Percentage (%)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'zero_f1_patterns.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved zero F1 pattern analysis to {output_dir / 'zero_f1_patterns.png'}")
    
    # Print detailed statistics
    print("\nZero F1 Statistics:")
    for model_type in models:
        if model_type in zero_counts:
            print(f"\n  {model_labels[model_type]} ({total_runs[model_type]} runs):")
            for group_id in range(6):
                count = zero_counts[model_type].get(group_id, 0)
                pct = (count / total_runs[model_type]) * 100 if total_runs[model_type] > 0 else 0
                print(f"    Fitz-{group_id+1}: {count} runs ({pct:.1f}%)")


def statistical_comparison(dfs, output_dir):
    """Perform comprehensive statistical comparisons."""
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("="*80)
    
    metrics = ['f1', 'performance_gap', 'worst_group_f1', 'demographic_parity', 
               'equalized_odds_diff', 'equal_opportunity_diff']
    
    baseline_models = ['direct', 'standard_cbm', 'curriculum_cbm']
    target_model = 'fair_curriculum_cbm'
    
    if target_model not in dfs:
        print(f"ERROR: {target_model} not found in results!")
        return
    
    results = []
    
    for baseline in baseline_models:
        if baseline not in dfs:
            continue
        
        print(f"\n{baseline.upper()} vs {target_model.upper()}:")
        print("-" * 60)
        
        for metric in metrics:
            if metric not in dfs[baseline].columns or metric not in dfs[target_model].columns:
                continue
            
            baseline_values = dfs[baseline][metric].dropna()
            target_values = dfs[target_model][metric].dropna()
            
            if len(baseline_values) < 2 or len(target_values) < 2:
                continue
            
            # Paired t-test
            min_n = min(len(baseline_values), len(target_values))
            baseline_values = baseline_values.values[:min_n]
            target_values = target_values.values[:min_n]
            
            t_stat, p_value = stats.ttest_rel(target_values, baseline_values)
            
            # Effect size (Cohen's d)
            diff = target_values - baseline_values
            cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
            
            # Determine if improvement (lower is better for gap metrics, higher for performance)
            is_gap_metric = 'gap' in metric or 'disparity' in metric or 'diff' in metric
            improved = (np.mean(diff) < 0 if is_gap_metric else np.mean(diff) > 0)
            
            baseline_mean = baseline_values.mean()
            target_mean = target_values.mean()
            pct_change = ((target_mean - baseline_mean) / baseline_mean * 100) if baseline_mean != 0 else 0
            
            results.append({
                'Baseline': baseline,
                'Metric': metric,
                'Baseline Mean': f"{baseline_mean:.3f}",
                'Target Mean': f"{target_mean:.3f}",
                'Difference': f"{target_mean - baseline_mean:+.3f}",
                '% Change': f"{pct_change:+.1f}%",
                'p-value': f"{p_value:.4f}",
                "Cohen's d": f"{cohens_d:.3f}",
                'Significant': 'Yes' if p_value < 0.05 else 'No',
                'Improved': 'Yes' if improved else 'No'
            })
            
            sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            improvement_marker = "↑" if improved else "↓"
            
            print(f"  {metric:25s}: {baseline_mean:.3f} → {target_mean:.3f} "
                  f"({pct_change:+.1f}%) {improvement_marker} "
                  f"[p={p_value:.4f} {sig_marker}, d={cohens_d:.2f}]")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'statistical_tests.csv', index=False)
    print(f"\nSaved statistical tests to {output_dir / 'statistical_tests.csv'}")
    
    return results_df


def create_paper_ready_table(dfs, output_dir):
    """Create paper-ready comparison table."""
    print("\n" + "="*80)
    print("GENERATING PAPER-READY TABLE")
    print("="*80)
    
    model_labels = {
        'direct': 'Direct',
        'standard_cbm': 'Standard CBM',
        'curriculum_cbm': 'Curriculum CBM',
        'fair_standard_cbm': 'Fair Standard CBM',
        'fair_curriculum_cbm': 'Fair Curriculum CBM (Ours)'
    }
    
    metrics = {
        'f1': 'Overall F1',
        'accuracy': 'Accuracy',
        'worst_group_f1': 'Worst-Group F1',
        'performance_gap': 'Perf. Gap ↓',
        'demographic_parity': 'DP Disparity ↓',
        'equalized_odds_diff': 'EO Disparity ↓'
    }
    
    table_data = []
    
    for model_type, label in model_labels.items():
        if model_type not in dfs:
            continue
        
        df = dfs[model_type]
        row = {'Model': label}
        
        for metric_key, metric_name in metrics.items():
            if metric_key in df.columns:
                values = df[metric_key].dropna()
                if len(values) > 0:
                    mean = values.mean()
                    std = values.std()
                    row[metric_name] = f"{mean:.3f} ± {std:.3f}"
                else:
                    row[metric_name] = "N/A"
            else:
                row[metric_name] = "N/A"
        
        table_data.append(row)
    
    table_df = pd.DataFrame(table_data)
    
    # Save CSV
    table_df.to_csv(output_dir / 'paper_ready_table.csv', index=False)
    print(f"\nSaved paper-ready table to {output_dir / 'paper_ready_table.csv'}")
    
    # Generate LaTeX
    latex_lines = []
    latex_lines.append("\\begin{table}[!htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Performance and fairness comparison across 5 models (mean $\\pm$ std over 20 runs). $\\downarrow$ indicates lower is better.}")
    latex_lines.append("\\label{tab:main_results}")
    latex_lines.append("\\small")
    latex_lines.append("\\begin{tabular}{@{}l" + "c"*6 + "@{}}")
    latex_lines.append("\\toprule")
    
    # Header
    headers = ['\\textbf{Model}'] + [f'\\textbf{{{col}}}' for col in table_df.columns[1:]]
    latex_lines.append(" & ".join(headers) + " \\\\")
    latex_lines.append("\\midrule")
    
    # Data rows
    for _, row in table_df.iterrows():
        values = [str(row[col]) for col in table_df.columns]
        latex_lines.append(" & ".join(values) + " \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_str = "\n".join(latex_lines)
    with open(output_dir / 'paper_ready_table.tex', 'w') as f:
        f.write(latex_str)
    
    print(f"Saved LaTeX table to {output_dir / 'paper_ready_table.tex'}")
    print("\nTable Preview:")
    print(table_df.to_string(index=False))
    
    return table_df


def variance_analysis(dfs, output_dir):
    """Analyze variance across runs to understand reliability."""
    print("\n" + "="*80)
    print("VARIANCE ANALYSIS (Are 20 runs enough?)")
    print("="*80)
    
    key_metrics = ['f1', 'performance_gap', 'worst_group_f1']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(key_metrics):
        ax = axes[idx]
        
        data = []
        labels = []
        
        for model_type, df in dfs.items():
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    data.append(values.values)
                    labels.append(model_type.replace('_', '\n'))
        
        # Box plot
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'variance_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved variance analysis to {output_dir / 'variance_analysis.png'}")
    
    # Calculate coefficient of variation
    print("\nCoefficient of Variation (CV = std/mean):")
    print("  (Higher CV indicates more variability, suggesting need for more runs)")
    print()
    
    cv_data = []
    for model_type, df in dfs.items():
        print(f"  {model_type}:")
        for metric in key_metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0 and values.mean() > 0:
                    cv = values.std() / values.mean()
                    print(f"    {metric}: CV = {cv:.3f}")
                    cv_data.append({
                        'Model': model_type,
                        'Metric': metric,
                        'Mean': values.mean(),
                        'Std': values.std(),
                        'CV': cv
                    })
    
    cv_df = pd.DataFrame(cv_data)
    cv_df.to_csv(output_dir / 'variance_stats.csv', index=False)
    print(f"\nSaved variance statistics to {output_dir / 'variance_stats.csv'}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive fairness analysis for paper')
    parser.add_argument('--results_dir', type=str, default='results/history',
                        help='Directory containing run_* subdirectories')
    parser.add_argument('--output_dir', type=str, default='results/detailed_analysis',
                        help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE FAIRNESS ANALYSIS FOR PAPER")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load all results
    dfs = load_all_results(results_dir)
    
    if not dfs:
        print("\nERROR: No results found!")
        return
    
    print(f"\nLoaded {len(dfs)} model types with {sum(len(df) for df in dfs.values())} total runs")
    
    # Run all analyses
    summary_df = analyze_performance_gap_drivers(dfs, output_dir)
    analyze_zero_f1_patterns(dfs, output_dir)
    stats_df = statistical_comparison(dfs, output_dir)
    table_df = create_paper_ready_table(dfs, output_dir)
    variance_analysis(dfs, output_dir)
    
    # Generate comprehensive report
    print("\n" + "="*80)
    print("SUMMARY RECOMMENDATIONS FOR PAPER")
    print("="*80)
    
    if 'fair_curriculum_cbm' in dfs:
        fcbm = dfs['fair_curriculum_cbm']
        
        print("\nYour Fair Curriculum CBM Results (20 runs):")
        print(f"  Overall F1:      {fcbm['f1'].mean():.3f} ± {fcbm['f1'].std():.3f}")
        print(f"  Performance Gap: {fcbm['performance_gap'].mean():.3f} ± {fcbm['performance_gap'].std():.3f}")
        print(f"  Worst-Group F1:  {fcbm['worst_group_f1'].mean():.3f} ± {fcbm['worst_group_f1'].std():.3f}")
        
        # Check CV
        gap_cv = fcbm['performance_gap'].std() / fcbm['performance_gap'].mean()
        print(f"\nPerformance Gap CV: {gap_cv:.3f}")
        if gap_cv > 0.5:
            print("  ⚠️  HIGH VARIANCE! Consider running 50-100 seeds for reliable estimates.")
        else:
            print("  ✓ Variance is reasonable for 20 runs.")
        
        # Best comparison
        if 'curriculum_cbm' in dfs:
            ccbm = dfs['curriculum_cbm']
            f1_improvement = ((fcbm['f1'].mean() - ccbm['f1'].mean()) / ccbm['f1'].mean()) * 100
            gap_reduction = ((ccbm['performance_gap'].mean() - fcbm['performance_gap'].mean()) / ccbm['performance_gap'].mean()) * 100
            
            print(f"\nCompared to Curriculum CBM (best baseline):")
            print(f"  F1 improvement:         {f1_improvement:+.1f}%")
            print(f"  Performance gap change: {-gap_reduction:+.1f}%")
            
            if gap_reduction < 0:
                print("\n  ⚠️  WARNING: Your fairness-focused curriculum has WORSE performance gap")
                print("      than the difficulty-based curriculum. This needs explanation in paper!")
            else:
                print(f"\n  ✓ Gap reduction of {gap_reduction:.1f}% while improving F1 by {f1_improvement:.1f}%")
    
    print("\n" + "="*80)
    print("Analysis complete! Check the output directory for detailed results.")
    print("="*80)


if __name__ == '__main__':
    main()
