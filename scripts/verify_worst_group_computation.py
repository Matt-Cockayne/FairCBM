#!/usr/bin/env python3
"""
Verify and correct worst-group F1 and performance gap computation.

Worst-group F1 should be: the mean F1 of whichever Fitzpatrick type 
has the worst average performance (not the per-run minimum).
"""

import pandas as pd
import numpy as np

def compute_correct_worst_group_metrics(df, model_type):
    """
    Compute worst-group F1 and performance gap correctly.
    
    Worst-group F1: Mean F1 of the Fitzpatrick type with worst average performance
    Performance gap: Difference between best and worst group mean F1
    """
    model_data = df[df['model_type'] == model_type].copy()
    
    # Compute mean F1 for each Fitzpatrick type across all runs
    fitz_means = {}
    for i in range(1, 7):
        fitz_col = f'fitz_{i}_f1'
        if fitz_col in model_data.columns:
            fitz_means[i] = model_data[fitz_col].mean()
    
    if not fitz_means:
        return None, None, None, None
    
    # Find worst and best groups
    worst_group_idx = min(fitz_means, key=fitz_means.get)
    best_group_idx = max(fitz_means, key=fitz_means.get)
    
    worst_group_mean = fitz_means[worst_group_idx]
    best_group_mean = fitz_means[best_group_idx]
    performance_gap = best_group_mean - worst_group_mean
    
    # Also compute std for worst group
    worst_group_col = f'fitz_{worst_group_idx}_f1'
    worst_group_std = model_data[worst_group_col].std()
    
    return worst_group_mean, worst_group_std, performance_gap, worst_group_idx


def main():
    csv_path = '/home/csc29/projects/SynergyCBM/FairCBM/results/analysis/all_results.csv'
    
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    models = df['model_type'].unique()
    
    print("\n" + "="*80)
    print("CORRECT WORST-GROUP F1 COMPUTATION")
    print("="*80)
    print("\nMethod: Identify Fitzpatrick type with worst average F1, report that type's mean F1\n")
    
    for model in models:
        model_data = df[df['model_type'] == model]
        
        print(f"\n{model.replace('_', ' ').title()} (n={len(model_data)} runs)")
        print("-" * 80)
        
        # Show per-Fitzpatrick means
        print("Per-Fitzpatrick mean F1:")
        fitz_means = {}
        for i in range(1, 7):
            fitz_col = f'fitz_{i}_f1'
            if fitz_col in model_data.columns:
                mean_f1 = model_data[fitz_col].mean()
                std_f1 = model_data[fitz_col].std()
                fitz_means[i] = mean_f1
                print(f"  Type {i}: {mean_f1:.3f} ± {std_f1:.3f}")
        
        # Compute correct worst-group metrics
        wg_mean, wg_std, gap, worst_idx = compute_correct_worst_group_metrics(df, model)
        
        if wg_mean is not None:
            print(f"\n✓ Worst group: Type {worst_idx}")
            print(f"✓ Worst-group F1 (CORRECT): {wg_mean:.3f} ± {wg_std:.3f}")
            print(f"✓ Performance gap (CORRECT): {gap:.3f}")
            
            # Compare with CSV values
            csv_wg = model_data['worst_group_f1'].mean()
            csv_gap = model_data['performance_gap'].mean()
            
            print(f"\nComparison with CSV values:")
            print(f"  CSV worst_group_f1: {csv_wg:.3f}")
            print(f"  Difference: {abs(csv_wg - wg_mean):.6f}")
            print(f"  CSV performance_gap: {csv_gap:.3f}")
            print(f"  Difference: {abs(csv_gap - gap):.6f}")
            
            if abs(csv_wg - wg_mean) > 0.001:
                print(f"  ⚠ WARNING: CSV worst_group_f1 does not match correct computation!")
            else:
                print(f"  ✓ CSV worst_group_f1 matches correct computation")
                
            if abs(csv_gap - gap) > 0.001:
                print(f"  ⚠ WARNING: CSV performance_gap does not match correct computation!")
            else:
                print(f"  ✓ CSV performance_gap matches correct computation")
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
