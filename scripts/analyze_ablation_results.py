"""
Analyze Ablation Study Results

Aggregates results from all ablation runs and computes statistics for the ablation table.

Usage:
    python analyze_ablation_results.py --results_dir results/ablation/ablation_study_20260105
    python analyze_ablation_results.py --results_dir results/ablation/ablation_study_20260105 --output ablation_analysis.json

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
import json
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze ablation study results')
    parser.add_argument('--results_dir', type=str, nargs='+', required=True,
                       help='Path(s) to ablation results directory (can specify multiple)')
    parser.add_argument('--main_experiment_dir', type=str, default=None,
                       help='Path to main experiment results (for full model baseline)')
    parser.add_argument('--output', type=str, default='ablation_analysis.json',
                       help='Output filename for analysis results')
    parser.add_argument('--confidence', type=float, default=0.95,
                       help='Confidence level for intervals')
    return parser.parse_args()


def get_ablation_key_from_run_id(run_id: int) -> str:
    """
    Map run ID to ablation key based on SLURM array job assignment.
    
    Jobs 0-99: full_model
    Jobs 100-199: no_phase1
    Jobs 200-299: no_phase2
    Jobs 300-399: no_phase3
    Jobs 400-499: no_phase4
    Jobs 500-599: no_adversarial
    
    Args:
        run_id: Run ID number
        
    Returns:
        Ablation key string
    """
    if run_id < 100:
        return 'full_model'
    elif run_id < 200:
        return 'no_phase1'
    elif run_id < 300:
        return 'no_phase2'
    elif run_id < 400:
        return 'no_phase3'
    elif run_id < 500:
        return 'no_phase4'
    elif run_id < 600:
        return 'no_adversarial'
    else:
        return 'unknown'


def load_ablation_results(results_dirs: List[Path]) -> Dict[str, List[Dict]]:
    """
    Load all ablation results from one or more directories.
    
    Supports two directory structures:
    1. New: results_dir/ablation_key/run_X/fair_curriculum_cbm/history.json
    2. Old: results_dir/run_X/fair_curriculum_cbm/history.json (with ablation_key in config)
    
    Args:
        results_dirs: List of paths to ablation results directories
        
    Returns:
        Dict mapping ablation_key to list of run results
    """
    results = defaultdict(list)
    loaded_runs = set()
    
    for results_dir in results_dirs:
        print(f"\nScanning directory: {results_dir}")
        
        if not results_dir.exists():
            print(f"  Warning: Directory not found")
            continue
        
        # Check if this uses the new structure (ablation subdirectories)
        ablation_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name in 
                        ['full_model', 'no_phase1', 'no_phase2', 'no_phase3', 'no_phase4', 'no_adversarial']]
        
        if ablation_dirs:
            # New structure: results_dir/ablation_key/run_X/...
            print(f"  Using NEW directory structure (ablation subdirectories)")
            for ablation_dir in ablation_dirs:
                ablation_key = ablation_dir.name
                run_dirs = sorted(ablation_dir.glob('run_*'))
                
                for run_dir in run_dirs:
                    model_dir = run_dir / 'fair_curriculum_cbm'
                    if not model_dir.exists():
                        continue
                    
                    # Extract run ID
                    try:
                        run_id = int(run_dir.name.split('_')[1])
                    except (IndexError, ValueError):
                        continue
                    
                    run_key = (results_dir.name, ablation_key, run_id)
                    if run_key in loaded_runs:
                        continue
                    
                    # Load results
                    run_results = load_single_run(model_dir)
                    if run_results:
                        results[ablation_key].append(run_results)
                        loaded_runs.add(run_key)
        
        else:
            # Old structure: results_dir/run_X/... (get ablation_key from config)
            print(f"  Using OLD directory structure (flat run directories)")
            run_dirs = sorted(results_dir.glob('run_*'))
            
            for run_dir in run_dirs:
                try:
                    run_id = int(run_dir.name.split('_')[1])
                except (IndexError, ValueError):
                    continue
                
                model_dir = run_dir / 'fair_curriculum_cbm'
                if not model_dir.exists():
                    continue
                
                # Load config to get ablation_key
                config_file = model_dir / 'config.json'
                ablation_key = None
                
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                            ablation_key = config.get('ablation_key')
                    except Exception:
                        pass
                
                if not ablation_key:
                    ablation_key = get_ablation_key_from_run_id(run_id)
                
                run_key = (results_dir.name, run_id)
                if run_key in loaded_runs:
                    continue
                
                # Load results
                run_results = load_single_run(model_dir)
                if run_results:
                    results[ablation_key].append(run_results)
                    loaded_runs.add(run_key)
    
    # Print summary
    print(f"\n{'='*70}")
    print("LOADING SUMMARY")
    print(f"{'='*70}")
    for ablation_key in ['full_model', 'no_phase1', 'no_phase2', 'no_phase3', 'no_phase4', 'no_adversarial']:
        count = len(results[ablation_key])
        print(f"  {ablation_key:20} {count:3d} runs")
    print(f"{'='*70}\n")
    
    return dict(results)


def load_single_run(model_dir: Path) -> dict:
    """Load results from a single run directory."""
    results_file = model_dir / 'results.json'
    history_file = model_dir / 'history.json'
    
    run_results = None
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            run_results = json.load(f)
            if 'test_results' in run_results:
                run_results = run_results['test_results']
    elif history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
            if 'test' in history and history['test']:
                test_result = history['test'][-1]
                run_results = {
                    'performance': test_result.get('binary_metrics', {}),
                    'fairness': test_result.get('binary_fairness', {})
                }
    
    return run_results


def compute_statistics(values: List[float], confidence: float = 0.95) -> Dict:
    """
    Compute statistics for a list of values.
    
    Args:
        values: List of numeric values
        confidence: Confidence level for intervals
        
    Returns:
        Dict with mean, std, median, min, max, ci_lower, ci_upper
    """
    if not values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'n': 0
        }
    
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0.0
    median = np.median(values)
    
    # Compute confidence interval
    if n > 1:
        se = std / np.sqrt(n)
        ci = stats.t.interval(confidence, n - 1, loc=mean, scale=se)
        ci_lower, ci_upper = ci
    else:
        ci_lower = ci_upper = mean
    
    return {
        'mean': float(mean),
        'std': float(std),
        'median': float(median),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n': int(n)
    }


def analyze_ablation_results(ablation_results: Dict[str, List[Dict]], 
                              confidence: float = 0.95) -> Dict:
    """
    Analyze ablation results and compute summary statistics.
    
    Args:
        ablation_results: Dict mapping ablation_key to list of run results
        confidence: Confidence level for intervals
        
    Returns:
        Dict with summary statistics per ablation
    """
    analysis = {}
    
    # Metrics to analyze
    performance_metrics = ['f1', 'accuracy', 'precision', 'recall', 'auc']
    
    # Fairness metrics are nested dicts, define paths to scalar values
    fairness_metric_paths = {
        'demographic_parity': ['demographic_parity', 'statistical_parity_difference'],
        'equalized_odds_diff': ['equalized_odds', 'equalized_odds_difference'],
        'equal_opportunity_diff': ['equal_opportunity', 'equal_opportunity_difference'],
        'worst_group_ece': ['calibration', 'worst_group_ece'],
        'performance_gap': ['worst_group', 'performance_gap'],
        'worst_group_f1': ['worst_group', 'worst_group_f1']
    }
    
    for ablation_key, runs in ablation_results.items():
        ablation_stats = {
            'num_runs': len(runs),
            'performance': {},
            'fairness': {}
        }
        
        # Check if this is pre-aggregated stats (from main experiment summary)
        if isinstance(runs, dict) and 'performance' in runs and 'fairness' in runs:
            # Already aggregated - use directly
            ablation_stats = runs
            analysis[ablation_key] = ablation_stats
            continue
        
        # Extract metrics from all runs
        for metric in performance_metrics:
            values = []
            for run in runs:
                if 'performance' in run and metric in run['performance']:
                    values.append(run['performance'][metric])
            
            if values:
                ablation_stats['performance'][metric] = compute_statistics(values, confidence)
        
        # Extract fairness metrics (handle nested dicts)
        for metric_name, path in fairness_metric_paths.items():
            values = []
            for run in runs:
                if 'fairness' not in run:
                    continue
                
                # Navigate nested dict structure
                value = run['fairness']
                try:
                    for key in path:
                        if isinstance(value, dict) and key in value:
                            value = value[key]
                        else:
                            value = None
                            break
                    
                    if value is not None and not isinstance(value, dict):
                        values.append(float(value))
                except (KeyError, TypeError, ValueError):
                    continue
            
            if values:
                ablation_stats['fairness'][metric_name] = compute_statistics(values, confidence)
        
        analysis[ablation_key] = ablation_stats
    
    return analysis


def compare_ablations(analysis: Dict) -> Dict:
    """
    Compare ablations to full model and compute deltas.
    
    Args:
        analysis: Analysis dict from analyze_ablation_results
        
    Returns:
        Dict with comparison results
    """
    if 'full_model' not in analysis:
        print("Warning: full_model not found in analysis")
        return {}
    
    full_model = analysis['full_model']
    comparisons = {}
    
    ablation_keys = ['no_phase1', 'no_phase2', 'no_phase3', 'no_phase4', 'no_adversarial']
    
    for ablation_key in ablation_keys:
        if ablation_key not in analysis:
            continue
        
        ablation = analysis[ablation_key]
        comparison = {
            'performance': {},
            'fairness': {}
        }
        
        # Compare performance metrics
        for metric in full_model['performance'].keys():
            if metric in ablation['performance']:
                full_mean = full_model['performance'][metric]['mean']
                abl_mean = ablation['performance'][metric]['mean']
                delta = abl_mean - full_mean
                pct_change = (delta / full_mean * 100) if full_mean != 0 else 0.0
                
                comparison['performance'][metric] = {
                    'full_model_mean': full_mean,
                    'ablation_mean': abl_mean,
                    'delta': delta,
                    'pct_change': pct_change
                }
        
        # Compare fairness metrics
        for metric in full_model['fairness'].keys():
            if metric in ablation['fairness']:
                full_mean = full_model['fairness'][metric]['mean']
                abl_mean = ablation['fairness'][metric]['mean']
                delta = abl_mean - full_mean
                pct_change = (delta / full_mean * 100) if full_mean != 0 else 0.0
                
                comparison['fairness'][metric] = {
                    'full_model_mean': full_mean,
                    'ablation_mean': abl_mean,
                    'delta': delta,
                    'pct_change': pct_change
                }
        
        comparisons[ablation_key] = comparison
    
    return comparisons


def create_ablation_table(analysis: Dict, comparisons: Dict) -> pd.DataFrame:
    """
    Create ablation table similar to the LaTeX format.
    
    Args:
        analysis: Analysis dict from analyze_ablation_results
        comparisons: Comparison dict from compare_ablations
        
    Returns:
        pandas DataFrame with ablation table
    """
    rows = []
    
    # Full model row
    if 'full_model' in analysis:
        full = analysis['full_model']
        
        # Format worst-group F1 with error if available
        worst_group_str = f"{full['fairness']['worst_group_f1']['mean']:.3f}"
        if 'std' in full['fairness']['worst_group_f1'] and full['fairness']['worst_group_f1']['std'] > 0:
            worst_group_str += f" ± {full['fairness']['worst_group_f1']['std']:.3f}"
        
        # Format performance gap with error if available
        perf_gap_str = f"{full['fairness']['performance_gap']['mean']:.3f}"
        if 'std' in full['fairness']['performance_gap'] and full['fairness']['performance_gap']['std'] > 0:
            perf_gap_str += f" ± {full['fairness']['performance_gap']['std']:.3f}"
        
        rows.append({
            'Configuration': 'Full Model (Ours)',
            'Overall F1': f"{full['performance']['f1']['mean']:.3f} ± {full['performance']['f1']['std']:.3f}",
            'Worst-Group F1': worst_group_str,
            'Perf. Gap': perf_gap_str,
            'DP Disparity': f"{full['fairness']['demographic_parity']['mean']:.3f} ± {full['fairness']['demographic_parity']['std']:.3f}",
            'F1 Delta': '—',
            'Gap Delta': '—'
        })
    
    # Ablation rows
    ablation_names = {
        'no_phase1': 'w/o Phase 1 (no balanced init)',
        'no_phase2': 'w/o Phase 2 (no DP focus)',
        'no_phase3': 'w/o Phase 3 (no EO + adversarial)',
        'no_phase4': 'w/o Phase 4 (no error-driven)',
        'no_adversarial': 'w/o Adversarial Debiasing'
    }
    
    for ablation_key in ['no_phase1', 'no_phase2', 'no_phase3', 'no_phase4', 'no_adversarial']:
        if ablation_key not in analysis:
            continue
        
        abl = analysis[ablation_key]
        comp = comparisons.get(ablation_key, {})
        
        # Format worst-group F1 with error if available
        worst_group_str = f"{abl['fairness']['worst_group_f1']['mean']:.3f}"
        if 'std' in abl['fairness']['worst_group_f1'] and abl['fairness']['worst_group_f1']['std'] > 0:
            worst_group_str += f" ± {abl['fairness']['worst_group_f1']['std']:.3f}"
        
        # Format performance gap with error if available
        perf_gap_str = f"{abl['fairness']['performance_gap']['mean']:.3f}"
        if 'std' in abl['fairness']['performance_gap'] and abl['fairness']['performance_gap']['std'] > 0:
            perf_gap_str += f" ± {abl['fairness']['performance_gap']['std']:.3f}"
        
        # Format deltas
        f1_delta = ''
        gap_delta = ''
        
        if 'performance' in comp and 'f1' in comp['performance']:
            delta = comp['performance']['f1']['delta']
            pct = comp['performance']['f1']['pct_change']
            f1_delta = f"{delta:+.3f} ({pct:+.1f}%)"
        
        if 'fairness' in comp and 'performance_gap' in comp['fairness']:
            delta = comp['fairness']['performance_gap']['delta']
            pct = comp['fairness']['performance_gap']['pct_change']
            gap_delta = f"{delta:+.3f} ({pct:+.1f}%)"
        
        rows.append({
            'Configuration': ablation_names.get(ablation_key, ablation_key),
            'Overall F1': f"{abl['performance']['f1']['mean']:.3f} ± {abl['performance']['f1']['std']:.3f}",
            'Worst-Group F1': worst_group_str,
            'Perf. Gap': perf_gap_str,
            'DP Disparity': f"{abl['fairness']['demographic_parity']['mean']:.3f} ± {abl['fairness']['demographic_parity']['std']:.3f}",
            'F1 Delta': f1_delta,
            'Gap Delta': gap_delta
        })
    
    return pd.DataFrame(rows)


def generate_latex_table(df: pd.DataFrame) -> str:
    """
    Generate LaTeX table code from DataFrame.
    
    Args:
        df: Ablation table DataFrame
        
    Returns:
        LaTeX table code string
    """
    latex = r'''\begin{table}[!htbp]
\centering
\caption{Ablation study removing individual phases (100-run baseline per configuration).}
\label{tab:ablation}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Configuration} & \textbf{Overall F1} & \textbf{Worst-Group F1} & \textbf{Perf. Gap} & \textbf{DP Disp.} \\
\midrule
'''
    
    for _, row in df.iterrows():
        latex += f"{row['Configuration']} & {row['Overall F1']} & {row['Worst-Group F1']} & {row['Perf. Gap']} & {row['DP Disparity']} \\\\\n"
    
    latex += r'''\bottomrule
\end{tabular}
\end{table}
'''
    
    return latex


def load_main_experiment_results(main_exp_dir: Path) -> dict:
    """
    Load full model results from main experiment summary table.
    
    Args:
        main_exp_dir: Path to main experiment results directory
        
    Returns:
        Dict with aggregated statistics for full model
    """
    print(f"\nLoading full model from main experiment: {main_exp_dir}")
    
    # Look for summary_table.csv in analysis subdirectory
    summary_file = main_exp_dir / 'analysis' / 'summary_table.csv'
    
    if not summary_file.exists():
        print(f"  Warning: Summary table not found at {summary_file}")
        return None
    
    try:
        # Load summary table
        df = pd.read_csv(summary_file)
        
        # Find fair_curriculum_cbm row
        fcbm_row = df[df['Model'] == 'fair_curriculum_cbm']
        
        if fcbm_row.empty:
            print(f"  Warning: fair_curriculum_cbm not found in summary table")
            return None
        
        # Parse the values (format: "mean ± std")
        def parse_value(val_str):
            if pd.isna(val_str):
                return 0.0, 0.0
            parts = str(val_str).split('±')
            mean = float(parts[0].strip())
            std = float(parts[1].strip()) if len(parts) > 1 else 0.0
            return mean, std
        
        # Extract metrics
        f1_mean, f1_std = parse_value(fcbm_row['f1'].values[0])
        recall_mean, recall_std = parse_value(fcbm_row['recall'].values[0])
        worst_group_f1_mean, worst_group_f1_std = parse_value(fcbm_row['worst_group_f1'].values[0])
        perf_gap_mean, perf_gap_std = parse_value(fcbm_row['performance_gap'].values[0])
        dp_mean, dp_std = parse_value(fcbm_row['statistical_parity_diff'].values[0])
        
        print(f"  ✓ Loaded fair_curriculum_cbm from summary table")
        print(f"    F1: {f1_mean:.3f} ± {f1_std:.3f}")
        print(f"    Worst-Group F1: {worst_group_f1_mean:.3f} ± {worst_group_f1_std:.3f}")
        print(f"    Performance Gap: {perf_gap_mean:.3f} ± {perf_gap_std:.3f}")
        
        # Return in the same format as analyze_ablation_results expects
        # Create a dict that mimics the aggregated statistics format
        return {
            'num_runs': 100,  # From the main experiment
            'performance': {
                'f1': {
                    'mean': f1_mean,
                    'std': f1_std,
                    'n': 100
                },
                'recall': {
                    'mean': recall_mean,
                    'std': recall_std,
                    'n': 100
                }
            },
            'fairness': {
                'worst_group_f1': {
                    'mean': worst_group_f1_mean,
                    'std': worst_group_f1_std,
                    'n': 100
                },
                'performance_gap': {
                    'mean': perf_gap_mean,
                    'std': perf_gap_std,
                    'n': 100
                },
                'demographic_parity': {
                    'mean': dp_mean,
                    'std': dp_std,
                    'n': 100
                }
            }
        }
        
    except Exception as e:
        print(f"  Error loading summary table: {e}")
        return None


def main():
    args = parse_args()
    
    # Convert results_dir arguments to Path objects
    results_dirs = [Path(d) for d in args.results_dir]
    
    # Check that at least one directory exists
    valid_dirs = [d for d in results_dirs if d.exists()]
    if not valid_dirs:
        print(f"Error: No valid results directories found")
        print(f"Tried: {', '.join(str(d) for d in results_dirs)}")
        return
    
    print(f"Analyzing ablation results from {len(valid_dirs)} director{'y' if len(valid_dirs) == 1 else 'ies'}:")
    for d in valid_dirs:
        print(f"  - {d}")
    print()
    
    # Load results
    print("Loading ablation results...")
    ablation_results = load_ablation_results(valid_dirs)
    
    # Load main experiment full model if specified
    if args.main_experiment_dir:
        main_exp_path = Path(args.main_experiment_dir)
        if main_exp_path.exists():
            main_exp_stats = load_main_experiment_results(main_exp_path)
            if main_exp_stats:
                # Replace full_model with main experiment results
                # The summary table gives us aggregated stats directly
                ablation_results['full_model_stats'] = main_exp_stats
                print(f"  ✓ Using main experiment summary for full_model baseline")
        else:
            print(f"  Warning: Main experiment directory not found: {main_exp_path}")
    
    if not ablation_results:
        print("Error: No results found")
        return
    
    print()
    
    # Analyze results
    print("Computing statistics...")
    analysis = analyze_ablation_results(ablation_results, confidence=args.confidence)
    
    # If we loaded main experiment stats, use them for full_model
    if 'full_model_stats' in ablation_results:
        analysis['full_model'] = ablation_results['full_model_stats']
    
    # Compare to full model
    print("Comparing ablations to full model...")
    comparisons = compare_ablations(analysis)
    
    # Create table
    print("\nCreating ablation table...")
    table_df = create_ablation_table(analysis, comparisons)
    
    print("\nAblation Table:")
    print("=" * 80)
    print(table_df.to_string(index=False))
    print("=" * 80)
    
    # Generate LaTeX
    latex_table = generate_latex_table(table_df)
    
    print("\nLaTeX Table Code:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)
    
    # Save results to first valid directory
    output_path = valid_dirs[0] / args.output
    
    output_data = {
        'analysis': analysis,
        'comparisons': comparisons,
        'table': table_df.to_dict(orient='records'),
        'latex': latex_table
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Also save LaTeX separately
    latex_path = output_path.with_suffix('.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    
    # Save CSV
    csv_path = output_path.with_suffix('.csv')
    table_df.to_csv(csv_path, index=False)
    
    print(f"\nResults saved:")
    print(f"  JSON: {output_path}")
    print(f"  LaTeX: {latex_path}")
    print(f"  CSV: {csv_path}")
    
    # Print summary insights
    print("\n" + "=" * 80)
    print("SUMMARY INSIGHTS")
    print("=" * 80)
    
    if 'full_model' in analysis:
        full = analysis['full_model']
        print(f"\nFull Model Performance:"), 'no_adversarial'
        print(f"  F1: {full['performance']['f1']['mean']:.3f} ± {full['performance']['f1']['std']:.3f}")
        print(f"  Performance Gap: {full['fairness']['performance_gap']['mean']:.3f} ± {full['fairness']['performance_gap']['std']:.3f}")
        print(f"  DP Disparity: {full['fairness']['demographic_parity']['mean']:.3f} ± {full['fairness']['demographic_parity']['std']:.3f}")
    
    print("\nPhase Contributions:")
    
    for ablation_key in ['no_phase1', 'no_phase2', 'no_phase3', 'no_phase4']:
        if ablation_key not in comparisons:
            continue
        
        comp = comparisons[ablation_key]
        phase_num = ablation_key.replace('no_phase', '')
        
        print(f"\n  Phase {phase_num} Impact (when removed):")
        
        if 'performance' in comp and 'f1' in comp['performance']:
            f1_change = comp['performance']['f1']['pct_change']
            print(f"    F1: {f1_change:+.1f}%")
        
        if 'fairness' in comp and 'performance_gap' in comp['fairness']:
            gap_change = comp['fairness']['performance_gap']['pct_change']
            print(f"    Performance Gap: {gap_change:+.1f}%")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")


if __name__ == '__main__':
    main()
