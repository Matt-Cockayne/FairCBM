"""
Generate LaTeX Ablation Table

Creates publication-ready LaTeX table from ablation study results with formatting options.

Usage:
    python generate_latex_table.py --results_dir results/ablation/ablation_study_20260105
    python generate_latex_table.py --results_dir results/ablation/ablation_study_20260105 --style compact
    python generate_latex_table.py --results_dir results/ablation/ablation_study_20260105 --show_ci

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
from typing import Dict


def parse_args():
    parser = argparse.ArgumentParser(description='Generate LaTeX ablation table')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to ablation results directory')
    parser.add_argument('--analysis_file', type=str, default='ablation_analysis.json',
                       help='Analysis file name')
    parser.add_argument('--output', type=str, default='ablation_table.tex',
                       help='Output LaTeX filename')
    parser.add_argument('--style', type=str, default='standard',
                       choices=['standard', 'compact', 'detailed'],
                       help='Table style')
    parser.add_argument('--show_ci', action='store_true',
                       help='Show confidence intervals')
    parser.add_argument('--show_deltas', action='store_true', default=True,
                       help='Show delta columns')
    parser.add_argument('--caption', type=str,
                       default='Ablation study: removing individual phases. All metrics on test set.',
                       help='Table caption')
    parser.add_argument('--label', type=str, default='tab:ablation',
                       help='Table label')
    return parser.parse_args()


def load_analysis(results_dir: Path, analysis_file: str) -> Dict:
    """Load ablation analysis results."""
    analysis_path = results_dir / analysis_file
    
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis file not found: {analysis_path}")
    
    with open(analysis_path, 'r') as f:
        data = json.load(f)
    
    return data


def format_metric(value: float, show_sign: bool = False, decimals: int = 3) -> str:
    """Format metric value for display."""
    if show_sign:
        return f"{value:+.{decimals}f}"
    return f"{value:.{decimals}f}"


def generate_standard_table(analysis: Dict, comparisons: Dict, 
                            show_ci: bool = False, show_deltas: bool = True,
                            caption: str = None, label: str = None) -> str:
    """Generate standard ablation table."""
    
    latex = r'''\begin{table}[!htbp]
\centering
'''
    
    if caption:
        latex += f"\\caption{{{caption}}}\n"
    
    if label:
        latex += f"\\label{{{label}}}\n"
    
    latex += r'''\small
'''
    
    # Table header
    if show_deltas:
        latex += r'''\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Configuration} & \textbf{Overall F1} & \textbf{Perf. Gap} & \textbf{DP Disparity} & \textbf{$\Delta$ F1} & \textbf{$\Delta$ Gap} & \textbf{$\Delta$ DP} \\
\midrule
'''
    else:
        latex += r'''\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Configuration} & \textbf{Overall F1} & \textbf{Perf. Gap} & \textbf{DP Disparity} \\
\midrule
'''
    
    # Full model row
    if 'full_model' in analysis:
        full = analysis['full_model']
        f1 = full['performance']['f1']['mean']
        gap = full['fairness']['performance_gap']['mean']
        dp = full['fairness']['demographic_parity']['mean']
        
        if show_ci:
            f1_ci = full['performance']['f1']
            gap_ci = full['fairness']['performance_gap']
            dp_ci = full['fairness']['demographic_parity']
            
            latex += f"Full Model (Ours) & ${format_metric(f1)} \pm {format_metric(f1_ci['std'])}$ & "
            latex += f"${format_metric(gap)} \pm {format_metric(gap_ci['std'])}$ & "
            latex += f"${format_metric(dp)} \pm {format_metric(dp_ci['std'])}$"
        else:
            latex += f"Full Model (Ours) & {format_metric(f1)} & {format_metric(gap)} & {format_metric(dp)}"
        
        if show_deltas:
            latex += " & — & — & —"
        
        latex += " \\\\\n"
    
    # Ablation rows
    ablation_configs = [
        ('no_phase1', 'w/o Phase 1 (no balanced init)'),
        ('no_phase2', 'w/o Phase 2 (no DP focus)'),
        ('no_phase3', 'w/o Phase 3 (no EO + adversarial)'),
        ('no_phase4', 'w/o Phase 4 (no error-driven)')
    ]
    
    for ablation_key, ablation_name in ablation_configs:
        if ablation_key not in analysis:
            continue
        
        abl = analysis[ablation_key]
        f1 = abl['performance']['f1']['mean']
        gap = abl['fairness']['performance_gap']['mean']
        dp = abl['fairness']['demographic_parity']['mean']
        
        if show_ci:
            f1_ci = abl['performance']['f1']
            gap_ci = abl['fairness']['performance_gap']
            dp_ci = abl['fairness']['demographic_parity']
            
            latex += f"{ablation_name} & ${format_metric(f1)} \pm {format_metric(f1_ci['std'])}$ & "
            latex += f"${format_metric(gap)} \pm {format_metric(gap_ci['std'])}$ & "
            latex += f"${format_metric(dp)} \pm {format_metric(dp_ci['std'])}$"
        else:
            latex += f"{ablation_name} & {format_metric(f1)} & {format_metric(gap)} & {format_metric(dp)}"
        
        # Add deltas
        if show_deltas and ablation_key in comparisons:
            comp = comparisons[ablation_key]
            
            f1_delta = comp['performance']['f1']['delta']
            gap_delta = comp['fairness']['performance_gap']['delta']
            dp_delta = comp['fairness']['demographic_parity']['delta']
            
            latex += f" & {format_metric(f1_delta, show_sign=True)}"
            latex += f" & {format_metric(gap_delta, show_sign=True)}"
            latex += f" & {format_metric(dp_delta, show_sign=True)}"
        
        latex += " \\\\\n"
    
    latex += r'''\bottomrule
\end{tabular}
\end{table}
'''
    
    return latex


def generate_compact_table(analysis: Dict, comparisons: Dict,
                          caption: str = None, label: str = None) -> str:
    """Generate compact ablation table (no deltas, just main metrics)."""
    
    latex = r'''\begin{table}[!htbp]
\centering
'''
    
    if caption:
        latex += f"\\caption{{{caption}}}\n"
    
    if label:
        latex += f"\\label{{{label}}}\n"
    
    latex += r'''\small
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Configuration} & \textbf{F1} & \textbf{Gap} & \textbf{DP} \\
\midrule
'''
    
    # Full model
    if 'full_model' in analysis:
        full = analysis['full_model']
        f1 = full['performance']['f1']['mean']
        gap = full['fairness']['performance_gap']['mean']
        dp = full['fairness']['demographic_parity']['mean']
        
        latex += f"Full Model & {format_metric(f1)} & {format_metric(gap)} & {format_metric(dp)} \\\\\n"
    
    # Ablations
    ablation_configs = [
        ('no_phase1', 'w/o Phase 1'),
        ('no_phase2', 'w/o Phase 2'),
        ('no_phase3', 'w/o Phase 3'),
        ('no_phase4', 'w/o Phase 4')
    ]
    
    for ablation_key, ablation_name in ablation_configs:
        if ablation_key not in analysis:
            continue
        
        abl = analysis[ablation_key]
        f1 = abl['performance']['f1']['mean']
        gap = abl['fairness']['performance_gap']['mean']
        dp = abl['fairness']['demographic_parity']['mean']
        
        latex += f"{ablation_name} & {format_metric(f1)} & {format_metric(gap)} & {format_metric(dp)} \\\\\n"
    
    latex += r'''\bottomrule
\end{tabular}
\end{table}
'''
    
    return latex


def generate_detailed_table(analysis: Dict, comparisons: Dict,
                           caption: str = None, label: str = None) -> str:
    """Generate detailed ablation table with error bars and worst-group F1."""
    
    latex = r'''\begin{table}[!htbp]
\centering
'''
    
    if caption:
        latex += f"\\caption{{{caption}}}\n"
    
    if label:
        latex += f"\\label{{{label}}}\n"
    
    latex += r'''\small
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Model} & \textbf{Overall F1} & \textbf{Worst-Group F1} & \textbf{Perf. Gap} & \textbf{DP Disparity} \\
\midrule
'''
    
    # Full model with error bars
    if 'full_model' in analysis:
        full = analysis['full_model']
        f1_mean = full['performance']['f1']['mean']
        f1_std = full['performance']['f1']['std']
        gap_mean = full['fairness']['performance_gap']['mean']
        gap_std = full['fairness']['performance_gap']['std']
        dp_mean = full['fairness']['demographic_parity']['mean']
        dp_std = full['fairness']['demographic_parity']['std']
        
        # Try to get worst group F1 if available
        worst_f1_mean = 0.0
        worst_f1_std = 0.0
        if 'worst_group_f1' in full['fairness']:
            worst_f1_mean = full['fairness']['worst_group_f1']['mean']
            worst_f1_std = full['fairness']['worst_group_f1']['std']
        
        latex += f"Fair Curriculum CBM & \\textbf{{{format_metric(f1_mean)}}} $\\pm$ {format_metric(f1_std, decimals=2)} & "
        latex += f"\\textbf{{{format_metric(worst_f1_mean)}}} $\\pm$ {format_metric(worst_f1_std, decimals=2)} & "
        latex += f"\\textbf{{{format_metric(gap_mean)}}} $\\pm$ {format_metric(gap_std, decimals=2)} & "
        latex += f"\\textbf{{{format_metric(dp_mean)}}} $\\pm$ {format_metric(dp_std, decimals=2)} \\\\\n"
    
    # Ablations with error bars
    ablation_configs = [
        ('no_phase1', 'w/o Phase 1 (no balanced init)'),
        ('no_phase2', 'w/o Phase 2 (no DP focus)'),
        ('no_phase3', 'w/o Phase 3 (no EO + adversarial)'),
        ('no_phase4', 'w/o Phase 4 (no error-driven)')
    ]
    
    for ablation_key, ablation_name in ablation_configs:
        if ablation_key not in analysis:
            continue
        
        abl = analysis[ablation_key]
        f1_mean = abl['performance']['f1']['mean']
        f1_std = abl['performance']['f1']['std']
        gap_mean = abl['fairness']['performance_gap']['mean']
        gap_std = abl['fairness']['performance_gap']['std']
        dp_mean = abl['fairness']['demographic_parity']['mean']
        dp_std = abl['fairness']['demographic_parity']['std']
        
        # Try to get worst group F1 if available
        worst_f1_mean = 0.0
        worst_f1_std = 0.0
        if 'worst_group_f1' in abl['fairness']:
            worst_f1_mean = abl['fairness']['worst_group_f1']['mean']
            worst_f1_std = abl['fairness']['worst_group_f1']['std']
        
        latex += f"{ablation_name} & {format_metric(f1_mean)} $\\pm$ {format_metric(f1_std, decimals=2)} & "
        latex += f"{format_metric(worst_f1_mean)} $\\pm$ {format_metric(worst_f1_std, decimals=2)} & "
        latex += f"{format_metric(gap_mean)} $\\pm$ {format_metric(gap_std, decimals=2)} & "
        latex += f"{format_metric(dp_mean)} $\\pm$ {format_metric(dp_std, decimals=2)} \\\\\n"
    
    latex += r'''\bottomrule
\end{tabular}
\end{table}
'''
    
    return latex


def generate_analysis_text(analysis: Dict, comparisons: Dict) -> str:
    """Generate LaTeX text describing ablation results."""
    
    if 'full_model' not in analysis:
        return ""
    
    full = analysis['full_model']
    
    text = "\n% Ablation Study Analysis\n"
    text += "% =======================\n\n"
    
    # Find most impactful phases
    impacts = []
    
    for ablation_key in ['no_phase1', 'no_phase2', 'no_phase3', 'no_phase4']:
        if ablation_key not in comparisons:
            continue
        
        comp = comparisons[ablation_key]
        phase_num = ablation_key.replace('no_phase', '')
        
        gap_pct = comp['fairness']['performance_gap']['pct_change']
        impacts.append((phase_num, gap_pct, ablation_key))
    
    # Sort by absolute impact
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    
    if len(impacts) >= 2:
        most_impactful = impacts[0]
        second_impactful = impacts[1]
        
        text += f"% Most impactful phase: Phase {most_impactful[0]} "
        text += f"(+{abs(most_impactful[1]):.1f}% gap increase when removed)\n"
        
        text += f"% Second most impactful: Phase {second_impactful[0]} "
        text += f"(+{abs(second_impactful[1]):.1f}% gap increase)\n\n"
        
        # Generate narrative text
        text += "Phase " + most_impactful[0]
        
        phase_names = {
            '1': 'balanced foundation',
            '2': 'demographic parity focus',
            '3': 'equalized odds + adversarial debiasing',
            '4': 'error-driven sampling'
        }
        
        text += f" ({phase_names.get(most_impactful[0], 'unknown')}) proves critical for fairness, "
        text += f"with removal increasing performance gaps by {abs(most_impactful[1]):.0f}\\%.\n"
        
        text += f"Phase {second_impactful[0]} "
        text += f"({phase_names.get(second_impactful[0], 'unknown')}) shows the second strongest impact, "
        text += f"with ablation degrading gaps by {abs(second_impactful[1]):.0f}\\%.\n"
        
        # Average contribution
        avg_contribution = sum(abs(x[1]) for x in impacts) / len(impacts)
        text += f"Each phase contributes an average of {avg_contribution:.1f}\\% gap reduction.\n"
    
    return text


def main():
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print(f"Generating LaTeX table from: {results_dir}")
    
    # Load analysis
    try:
        data = load_analysis(results_dir, args.analysis_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run analyze_ablation_results.py first to generate the analysis file.")
        return
    
    # Generate table based on style
    if args.style == 'standard':
        latex = generate_standard_table(
            data['analysis'], data['comparisons'],
            show_ci=args.show_ci, show_deltas=args.show_deltas,
            caption=args.caption, label=args.label
        )
    elif args.style == 'compact':
        latex = generate_compact_table(
            data['analysis'], data['comparisons'],
            caption=args.caption, label=args.label
        )
    elif args.style == 'detailed':
        latex = generate_detailed_table(
            data['analysis'], data['comparisons'],
            caption=args.caption, label=args.label
        )
    
    # Generate analysis text
    analysis_text = generate_analysis_text(data['analysis'], data['comparisons'])
    
    # Save LaTeX
    output_path = results_dir / args.output
    
    with open(output_path, 'w') as f:
        f.write("% Ablation Study Table\n")
        f.write(f"% Generated from: {results_dir}\n")
        f.write(f"% Style: {args.style}\n\n")
        f.write(latex)
        f.write("\n\n")
        f.write(analysis_text)
    
    print(f"\nLaTeX table saved to: {output_path}")
    
    # Print preview
    print("\n" + "=" * 80)
    print("LaTeX Table Preview:")
    print("=" * 80)
    print(latex)
    print("=" * 80)
    
    if analysis_text:
        print("\nAnalysis Text:")
        print("=" * 80)
        print(analysis_text)
        print("=" * 80)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
