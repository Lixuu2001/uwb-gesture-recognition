"""Render all 6 MV-OSR figures end-to-end.

Requires the cached dataset (`cache/X_clutterremoved.npy` etc.) only for
fig6 (confusion matrix); fig1-fig5 read static numbers from the verified
log SUMMARY lines.

Before running fig6 for the first time, generate the case-1 prediction
dumps with:

    python extract_case1_predictions.py --single   # -> results/case1_1view.npz
    python extract_case1_predictions.py            # -> results/case1_3view.npz
"""
import os
import sys
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SCRIPTS = [
    "fig1_per_case_oscr.py",      # per-case OSCR + MV gain (hero)
    "fig2_ablation_heatmap.py",   # pool x normalize 2x2 heatmap
    "fig3_normalisation_diag.py", # 4-mode normalization diagnostic
    "fig4_gap_and_seed.py",       # closed-vs-OSR gap + case 4 seed stability
    "fig5_table_ii.py",           # Eom Table II look-alike (4 sub-tables)
    "fig6_confmat.py",            # Eom Fig 9 / 10 look-alike (confusion matrices)
    "source_data.py",             # write source_data_fig{1..4}.csv
]

for script in SCRIPTS:
    print(f"--- {script} ---")
    try:
        runpy.run_path(os.path.join(HERE, script), run_name="__main__")
    except FileNotFoundError as e:
        print(f"  skipped: {e}  "
              f"(run extract_case1_predictions.py first for fig6)")
print("done.")
