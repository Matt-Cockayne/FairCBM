import numpy as np, sys
sys.path.insert(0, '.')
from scripts.generate_visualizations import aggregate_across_runs, MULTI_RUN_BASE

for model in ['curriculum_cbm', 'fair_curriculum_cbm']:
    d = aggregate_across_runs(MULTI_RUN_BASE, model)
    print(model)
    print("  LG-F1 =", round(d['final_worst'], 3), "+-", round(d['final_worst_std'], 3))
    print("  Gap   =", round(d['final_gap'], 3),   "+-", round(d['final_gap_std'], 3))
    print("  F1    =", round(d['final_overall'], 3),"+-", round(d['final_overall_std'], 3))
    print("  worst group idx =", d['worst_group_idx'], "(Fitzpatrick Type", d['worst_group_idx']+1, ")")
