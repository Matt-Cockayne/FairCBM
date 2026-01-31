"""
Analyze consistent ablation study results from train_all_models.py.

This script analyzes ablation results that use the same training infrastructure
as the main experiment, ensuring proper baseline comparison.

Usage:
    python scripts/analyze_consistent_ablation.py --exp_name ablation_study_20260108
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def load_test_results(exp_dir: Path, ablation_key: str):
    """Load test results for a specific ablation configuration."""
    results = []
    
    # Find all run directories
    run_dirs = sorted(exp_dir.glob('run_*'))
    
    for run_dir in run_dirs:
        # Check config for ablation key
        config_path = run_dir / 'fair_curriculum_cbm' / 'config.json'
        if not config_path.exists():
            continue
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Skip if not the ablation we're looking for
        if config.get('ablation_key') != ablation_key:
            continue
        
        # Load test results
        test_results_path = run_dir / 'fair_curriculum_cbm' / 'test_results.json'
        if test_results_path.exists():
            with open(test_results_path, 'r') as f:
                test_results = json.load(f)
            results.append({
                'run_dir': str(run_dir),
                'seed': config.get('seed', -1),
                'results': test_results
            })
    
    return results


def compute_statistics(results):
    """Compute mean and std for key metrics."""
    if len(results) == 0:
        return {
            'num_runs': 0,
            'f1_mean': 0.0, 'f1_std': 0.0,
            'accuracy_mean': 0.0, 'accuracy_std': 0.0,
            'dp_mean': 0.0, 'dp_std': 0.0,
            'eo_mean': 0.0, 'eo_std': 0.0,
            'gap_mean': 0.0, 'gap_std': 0.0
        }
    
    # Extract metrics
    f1_scores = [r['results']['binary_metrics']['f1'] for r in results]
    accuracy = [r['results']['binary_metrics']['accuracy'] for r in results]
    
    dp = []
    eo = []
    gaps = []
    for r in results:
        if 'binary_fairness' in r['results']:
            fairness = r['results']['binary_fairness']
            dp.append(fairness['demographic_parity']['max_disparity'])
            eo.append(fairness['equalized_odds']['max_disparity'])
            gaps.append(fairness['performance_gap']['gap'])
    
    return {
        'num_runs': len(results),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'accuracy_mean': np.mean(accuracy),
        'accuracy_std': np.std(accuracy),
        'dp_mean': np.mean(dp) if dp else 0.0,
        'dp_std': np.std(dp) if dp else 0.0,
        'eo_mean': np.mean(eo) if eo else 0.0,
        'eo_std': np.std(eo) if eo else 0.0,
        'gap_mean': np.mean(gaps) if gaps else 0.0,
        'gap_std': np.std(gaps) if gaps else 0.0
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True, help='Experiment name')
    parser.add_argument('--results_dir', default='results', help='Results directory')
    args = parser.parse_args()
    
    exp_dir = Path(args.results_dir) / args.exp_name
    
    print(f"Analyzing: {exp_dir}")
    print()
    
    # Load results for each ablation
    ablations = ['full_model', 'no_phase2', 'no_phase3', 'no_phase4', 'no_adversarial']
    stats = {}
    
    for ablation in ablations:
        print(f"Loading {ablation}...")
        results = load_test_results(exp_dir, ablation)
        stats[ablation] = compute_statistics(results)
        print(f"  Found {stats[ablation]['num_runs']} runs")
    
    print()
    print("=" * 80)
    print("ABLATION RESULTS")
    print("=" * 80)
    print()
    
    # Create table
    names = {
        'full_model': 'Full Model',
        'no_phase2': 'w/o Phase 2 (DP)',
        'no_phase3': 'w/o Phase 3 (EO)',
        'no_phase4': 'w/o Phase 4 (PG)',
        'no_adversarial': 'w/o Adversarial'
    }
    
    for ablation in ablations:
        s = stats[ablation]
        print(f"{names[ablation]:<25} (n={s['num_runs']:3d})")
        print(f"  F1:       {s['f1_mean']:.3f} ± {s['f1_std']:.3f}")
        print(f"  Accuracy: {s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f}")
        print(f"  DP:       {s['dp_mean']:.3f} ± {s['dp_std']:.3f}")
        print(f"  EO:       {s['eo_mean']:.3f} ± {s['eo_std']:.3f}")
        print(f"  Gap:      {s['gap_mean']:.3f} ± {s['gap_std']:.3f}")
        print()
    
    # Compute phase contributions
    if 'full_model' in stats and stats['full_model']['num_runs'] > 0:
        print("=" * 80)
        print("PHASE CONTRIBUTIONS (vs Full Model)")
        print("=" * 80)
        print()
        
        baseline = stats['full_model']
        for ablation in ['no_phase2', 'no_phase3', 'no_phase4', 'no_adversarial']:
            if ablation not in stats or stats[ablation]['num_runs'] == 0:
                continue
            
            s = stats[ablation]
            delta_f1 = baseline['f1_mean'] - s['f1_mean']
            delta_gap = s['gap_mean'] - baseline['gap_mean']
            delta_dp = s['dp_mean'] - baseline['dp_mean']
            delta_eo = s['eo_mean'] - baseline['eo_mean']
            
            print(f"{names[ablation]:<25}")
            print(f"  ΔF1:  {delta_f1:+.4f}  (removing hurts F1 by {abs(delta_f1)*100:.2f}%)")
            print(f"  ΔGap: {delta_gap:+.4f}  (gap increases by {delta_gap*100:.2f}%)")
            print(f"  ΔDP:  {delta_dp:+.4f}")
            print(f"  ΔEO:  {delta_eo:+.4f}")
            print()
    
    # Save results
    output_dir = exp_dir / 'analysis'
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'ablation_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
