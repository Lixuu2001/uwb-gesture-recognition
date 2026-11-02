"""Write source-data CSV files alongside the four figures.

One CSV per figure, named after the figure (Nature requires
source-data tracebility per figure for reviewer audit).
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Fig 1: per-case OSCR ----
fig1 = [
    ["case", "single_view_OSCR", "three_view_OSCR", "MV_gain"],
    ["Case 1", 87.04, 96.19, 9.15],
    ["Case 2", 84.35, 94.80, 10.45],
    ["Case 3", 85.99, 94.72,  8.73],
    ["Case 4", 73.54, 94.04, 20.50],
    ["Case 5", 83.59, 95.95, 12.36],
    ["Mean",   82.90, 95.14, 12.24],
    ["Std",     4.83,  0.81,  None],
]
with open(os.path.join(HERE, "source_data_fig1.csv"), "w", newline="") as f:
    csv.writer(f).writerows(fig1)

# ---- Fig 2: 2x2 ablation (OSCR mean ± std) ----
fig2 = [
    ["branch", "pool", "normalize", "OSCR_mean_5case", "OSCR_std_5case"],
    ["single_view", "mean",    "per_view_std", 69.71, None],
    ["single_view", "flatten", "per_view_std", 72.42, 4.73],
    ["single_view", "mean",    "abs_std",      83.34, 3.34],
    ["single_view", "flatten", "abs_std",      82.90, 4.83],
    ["three_view",  "mean",    "per_view_std", 92.56, None],
    ["three_view",  "flatten", "per_view_std", 90.63, 2.26],
    ["three_view",  "mean",    "abs_std",      93.63, 2.05],
    ["three_view",  "flatten", "abs_std",      95.14, 0.81],
]
with open(os.path.join(HERE, "source_data_fig2.csv"), "w", newline="") as f:
    csv.writer(f).writerows(fig2)

# ---- Fig 3: normalisation diagnostic (case 0, single-view flatten + inspector) ----
fig3 = [
    ["normalize", "closed_acc", "INSPECTOR_OSCR", "AUROC", "MSP_OSCR"],
    ["none",         87.58, 66.44, 71.94, 66.75],
    ["per_view_std", 95.67, 74.85, 75.95, 72.40],
    ["abs",          97.42, 81.34, 82.71, 76.55],
    ["abs_std",      96.67, 85.15, 86.13, 77.83],
]
with open(os.path.join(HERE, "source_data_fig3.csv"), "w", newline="") as f:
    csv.writer(f).writerows(fig3)

# ---- Fig 4: closed-vs-OSCR gap (3-view) + Case 4 seed stability (1-view) ----
fig4a = [
    ["pool", "normalize", "closed_acc_3v", "OSCR_3v"],
    ["mean",    "per_view_std", 99.08, 92.56],
    ["flatten", "per_view_std", 96.98, 90.63],
    ["mean",    "abs_std",      97.62, 93.63],
    ["flatten", "abs_std",      97.77, 95.14],
]
with open(os.path.join(HERE, "source_data_fig4a.csv"), "w", newline="") as f:
    csv.writer(f).writerows(fig4a)

fig4b = [
    ["seed", "INSPECTOR_OSCR_case4_1v", "MSP_OSCR_case4_1v", "AUROC", "closed_acc"],
    [0, 73.18, 79.36, 76.49, 94.67],
    [1, 71.46, 76.93, 75.52, 94.58],
    [2, 73.40, 73.81, 77.09, 94.33],
    ["mean", 72.68, 76.70, 76.37, 94.53],
    ["std",   0.86,  None,  None,  None],
]
with open(os.path.join(HERE, "source_data_fig4b.csv"), "w", newline="") as f:
    csv.writer(f).writerows(fig4b)

print("wrote source_data_fig{1,2,3,4a,4b}.csv")
