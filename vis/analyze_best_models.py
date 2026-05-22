"""
Analyze the best models selected during training to understand validation vs test performance.
"""
import json
from pathlib import Path

def main():
    # Load best models info
    with open('results/history/best-models.json') as f:
        best_models = json.load(f)
    
    print("="*80)
    print("BEST MODEL ANALYSIS: Validation Selection vs Test Performance")
    print("="*80)
    print("\nThese are the SINGLE BEST models selected during training (out of 20 runs)")
    print("based on validation composite score = val_f1 × (1 - performance_gap)")
    print()
    
    results = []
    
    for model_type, info in best_models.items():
        exp_name = info['exp_name']
        # Extract run_id from format: multi_run_1517148/run_4 or run_4
        if '/' in exp_name:
            run_id = int(exp_name.split('/')[-1].replace('run_', ''))
        else:
            run_id = int(exp_name.replace('run_', ''))
        
        # Load test results for this run
        history_file = Path(f"results/history/run_{run_id}/{model_type}/history.json")
        
        if not history_file.exists():
            print(f"Warning: {history_file} not found")
            continue
        
        with open(history_file) as f:
            history = json.load(f)
        
        test = history['test'][0]
        test_f1 = test['binary_metrics']['f1']
        test_acc = test['binary_metrics']['accuracy']
        
        if 'binary_fairness' in test and 'worst_group' in test['binary_fairness']:
            wg = test['binary_fairness']['worst_group']
            test_gap = wg['performance_gap']
            test_worst_f1 = wg['worst_group_f1']
        else:
            test_gap = None
            test_worst_f1 = None
        
        results.append({
            'model': model_type,
            'run_id': run_id,
            'val_f1': info['val_f1'],
            'val_gap': info['performance_gap'],
            'val_composite': info['composite_score'],
            'test_f1': test_f1,
            'test_gap': test_gap,
            'test_worst_f1': test_worst_f1,
            'epoch': info['epoch']
        })
    
    # Sort by validation composite score
    results.sort(key=lambda x: x['val_composite'], reverse=True)
    
    print("\nModel Rankings by Validation Composite Score:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Model':<25} {'Val F1':<10} {'Val Gap':<10} {'Val Composite':<15} {'Run':<6}")
    print("-" * 80)
    
    for rank, r in enumerate(results, 1):
        print(f"{rank:<6} {r['model']:<25} {r['val_f1']:<10.3f} {r['val_gap']:<10.3f} {r['val_composite']:<15.3f} {r['run_id']:<6}")
    
    print("\n\nTest Performance of These Best Models:")
    print("-" * 80)
    print(f"{'Model':<25} {'Test F1':<12} {'Test Gap':<12} {'Test Worst-F1':<15} {'Gap Δ':<12}")
    print("-" * 80)
    
    for r in results:
        gap_delta = r['test_gap'] - r['val_gap'] if r['test_gap'] else None
        gap_str = f"{r['test_gap']:.3f}" if r['test_gap'] else "N/A"
        worst_str = f"{r['test_worst_f1']:.3f}" if r['test_worst_f1'] else "N/A"
        delta_str = f"{gap_delta:+.3f}" if gap_delta is not None else "N/A"
        
        print(f"{r['model']:<25} {r['test_f1']:<12.3f} {gap_str:<12} {worst_str:<15} {delta_str:<12}")
    
    print("\n" + "="*80)
    print("KEY OBSERVATIONS:")
    print("="*80)
    
    # Find fair_curriculum_cbm
    fcbm = next((r for r in results if r['model'] == 'fair_curriculum_cbm'), None)
    if fcbm:
        print(f"\n1. Fair Curriculum CBM (Best Single Model - Run {fcbm['run_id']}):")
        print(f"   - Validation: F1={fcbm['val_f1']:.3f}, Gap={fcbm['val_gap']:.3f}")
        print(f"   - Test:       F1={fcbm['test_f1']:.3f}, Gap={fcbm['test_gap']:.3f}, Worst-F1={fcbm['test_worst_f1']:.3f}")
        print(f"   - Gap increased from {fcbm['val_gap']:.3f} → {fcbm['test_gap']:.3f} on test (+{fcbm['test_gap']-fcbm['val_gap']:.3f})")
        print(f"   - This is OVERFITTING on validation fairness")
    
    # Compare to curriculum
    ccbm = next((r for r in results if r['model'] == 'curriculum_cbm'), None)
    if fcbm and ccbm:
        print(f"\n2. Comparison to Curriculum CBM (Best Single Model - Run {ccbm['run_id']}):")
        print(f"   Fair Curriculum: Test F1={fcbm['test_f1']:.3f}, Gap={fcbm['test_gap']:.3f}")
        print(f"   Curriculum:      Test F1={ccbm['test_f1']:.3f}, Gap={ccbm['test_gap']:.3f}")
        if fcbm['test_gap'] < ccbm['test_gap']:
            improvement = ((ccbm['test_gap'] - fcbm['test_gap']) / ccbm['test_gap'] * 100)
            print(f"   → Fair version has {improvement:.1f}% better gap (single best models)")
        else:
            worsening = ((fcbm['test_gap'] - ccbm['test_gap']) / ccbm['test_gap'] * 100)
            print(f"   → Fair version has {worsening:.1f}% WORSE gap (single best models)")
    
    print("\n3. The Paper Reporting Issue:")
    print("   ⚠️  Reporting ONLY the best model is CHERRY-PICKING")
    print("   ⚠️  Scientific papers must report average ± std over ALL runs")
    print("   ⚠️  Your 20-run average shows: F1=0.691±0.158, Gap=0.571±0.323")
    print(f"   ⚠️  Best model shows:         F1={fcbm['test_f1']:.3f}, Gap={fcbm['test_gap']:.3f}")
    print("   → These are DIFFERENT because of high variance across runs!")
    
    print("\n4. Validation vs Test Generalization:")
    avg_gap_increase = sum((r['test_gap'] - r['val_gap']) for r in results if r['test_gap']) / len([r for r in results if r['test_gap']])
    print(f"   Average gap increase from val→test: {avg_gap_increase:+.3f}")
    print("   → Models selected for good validation fairness don't generalize well to test")
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print("For the paper, you should report:")
    print("1. PRIMARY RESULTS: Mean ± std over all 20 runs (what comprehensive_fairness_analysis.py shows)")
    print("2. OPTIONAL SUPPLEMENT: Best model performance as an 'upper bound' with clear disclaimer")
    print("3. NEVER claim the best model performance as representative of the method")
    print("\nThe 20-run average is the honest, scientific way to report results.")
    print("="*80)

if __name__ == '__main__':
    main()
