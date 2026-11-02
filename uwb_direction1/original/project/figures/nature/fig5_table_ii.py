"""Figure 5 -- Eom Table II look-alike, drawn as a tabular image.

Claim
-----
For each of the four standard OSR metrics (Accuracy, F1, AUROC, OSCR), the
three-view inspector head dominates every same-backbone single-view
configuration and the MSP-only baseline; the three-view OSCR mean (95.14)
exceeds the Eom 2025 single-view OSCR (88.42) by 6.72 points.

Layout
------
Four sub-tables stacked: (a) Accuracy, (b) F1, (c) AUROC, (d) OSCR.
Rows = schemes (Proposed 3-view, 3-view MSP, 1-view inspector, 1-view
MSP, Eom 2025). Columns = case 1..5 + Mean.

Data sources
------------
Per-case OSCR / closed_acc / inspector AUROC: logs/sweep5_*_flat_absstd.log
(verified per case). Per-case Acc / F1 and per-case MSP AUROC are not
emitted by main.py, so those per-case cells show "--" and only the
all-case Mean column is filled. Eom 2025 row shows the single OSCR=88.42
value quoted in the text and "--" elsewhere.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _style import PALETTE, save_pub

HERE = os.path.dirname(os.path.abspath(__file__))

# ===== data (per-case OSCR + AUROC verified from log SUMMARY lines) =====

NAN_CELL = "--"

# Each entry: (case1, case2, case3, case4, case5, mean ± std)
# When per-case is unavailable, use [None]*5 and fill mean only.

schemes = [
    "Proposed (3-view inspector)",
    "3-view MSP",
    "Ours, 1-view inspector",
    "1-view MSP",
    "Eom 2025 (reported)",
]

# Accuracy (%): only Mean column from SUMMARY
acc_per_case = {
    "Proposed (3-view inspector)": [None] * 5 + ["92.29 ± 1.27"],
    "3-view MSP":                  [None] * 5 + ["78.41 ± 5.70"],
    "Ours, 1-view inspector":      [None] * 5 + ["78.03 ± 3.77"],
    "1-view MSP":                  [None] * 5 + ["72.37 ± 4.94"],
    "Eom 2025 (reported)":         [None] * 5 + [NAN_CELL],
}

# F1 (%): same -- only Mean
f1_per_case = {
    "Proposed (3-view inspector)": [None] * 5 + ["92.54 ± 1.20"],
    "3-view MSP":                  [None] * 5 + ["82.62 ± 4.17"],
    "Ours, 1-view inspector":      [None] * 5 + ["80.84 ± 3.10"],
    "1-view MSP":                  [None] * 5 + ["77.63 ± 3.04"],
    "Eom 2025 (reported)":         [None] * 5 + [NAN_CELL],
}

# AUROC (%): per-case for inspector heads; MSP gets only Mean
auroc_per_case = {
    "Proposed (3-view inspector)": [96.62, 98.72, 95.14, 98.09, 97.02, "97.12 ± 1.24"],
    "3-view MSP":                  [None] * 5 + ["83.34 ± 7.25"],
    "Ours, 1-view inspector":      [87.73, 88.45, 86.60, 77.11, 84.72, "84.92 ± 4.10"],
    "1-view MSP":                  [None] * 5 + ["78.49 ± 5.27"],
    "Eom 2025 (reported)":         [None] * 5 + [NAN_CELL],
}

# OSCR (%): per-case for all 4 of our rows; Eom mean only
oscr_per_case = {
    "Proposed (3-view inspector)": [96.19, 94.80, 94.72, 94.04, 95.95, "95.14 ± 0.81"],
    "3-view MSP":                  [82.64, 81.15, 90.74, 86.39, 68.34, "81.85 ± 7.53"],
    "Ours, 1-view inspector":      [87.04, 84.35, 85.99, 73.54, 83.59, "82.90 ± 4.83"],
    "1-view MSP":                  [82.57, 78.43, 73.75, 81.15, 69.21, "77.02 ± 4.93"],
    "Eom 2025 (reported)":         [None] * 5 + ["88.42"],
}

panels = [
    ("(a)  Accuracy (%)",  acc_per_case),
    ("(b)  F1-score (%)",  f1_per_case),
    ("(c)  AUROC (%)",     auroc_per_case),
    ("(d)  OSCR (%)",      oscr_per_case),
]


def _cell_text(v):
    if v is None:
        return NAN_CELL
    if isinstance(v, str):
        return v
    return f"{v:.2f}"


def draw_table(ax, title, data_dict, highlight_rows=("Proposed (3-view inspector)",)):
    """Draw one Eom-style sub-table on a matplotlib axes."""
    ax.set_axis_off()
    cols = ["Scheme", "case 1", "case 2", "case 3", "case 4", "case 5", "Mean"]
    rows = [[name] + [_cell_text(v) for v in data_dict[name]] for name in schemes]

    n_rows = len(rows) + 1
    n_cols = len(cols)

    # column widths: Scheme is wider
    col_widths = np.array([0.30, 0.105, 0.105, 0.105, 0.105, 0.105, 0.175])
    col_widths /= col_widths.sum()

    # row geometry
    row_h = 1.0 / n_rows
    x_edges = np.concatenate([[0.0], np.cumsum(col_widths)])

    # Title at top of this panel
    ax.text(0.5, 1.06, title, ha="center", va="bottom", fontsize=8.5,
             fontweight="bold", transform=ax.transAxes,
             color=PALETTE["text"])

    # Top double rule
    for off, lw in ((1.00, 1.2), (0.998 - row_h, 0.6)):
        ax.plot([0, 1], [off, off], color=PALETTE["text"], lw=lw,
                transform=ax.transAxes)

    # Bottom double rule
    ax.plot([0, 1], [0.0, 0.0], color=PALETTE["text"], lw=1.2,
            transform=ax.transAxes)

    # Header text
    y_header = 1.0 - row_h / 2
    for j, col_name in enumerate(cols):
        x_center = (x_edges[j] + x_edges[j + 1]) / 2
        ax.text(x_center, y_header, col_name, ha="center", va="center",
                fontsize=7.0, fontweight="bold", transform=ax.transAxes,
                color=PALETTE["text"])

    # Body rows
    for i, row in enumerate(rows):
        y_top = 1.0 - row_h * (i + 1)
        y_center = y_top - row_h / 2
        is_hl = row[0] in highlight_rows
        if is_hl:
            ax.add_patch(Rectangle((0, y_top - row_h), 1, row_h,
                                    transform=ax.transAxes,
                                    color=PALETTE["three_lt"], alpha=0.20,
                                    zorder=0))
        for j, cell in enumerate(row):
            x_center = (x_edges[j] + x_edges[j + 1]) / 2
            ha = "left" if j == 0 else "center"
            x = x_edges[j] + 0.008 if j == 0 else x_center
            weight = "bold" if (is_hl and j == n_cols - 1) else "normal"
            color = PALETTE["three"] if (is_hl and j == n_cols - 1) else PALETTE["text"]
            ax.text(x, y_center, cell, ha=ha, va="center", fontsize=6.6,
                    transform=ax.transAxes, color=color, fontweight=weight)

    # column separator vertical (faint) above Mean column to set it apart
    ax.plot([x_edges[-2], x_edges[-2]], [0.0, 1.0 - row_h],
            color="#CCCCCC", lw=0.4, transform=ax.transAxes)


fig, axes = plt.subplots(4, 1, figsize=(7.0, 9.2),
                          gridspec_kw=dict(hspace=0.55))

for ax, (title, data) in zip(axes, panels):
    draw_table(ax, title, data)

# Note at bottom
fig.text(0.5, -0.005,
         '"--" indicates a per-case value not emitted by the run script '
         '(per-case Acc / F1 and per-case MSP AUROC are only available as the '
         '5-case mean; the engine computes them but main.py prints only '
         'OSCR / AUROC per case).',
         ha="center", va="top", fontsize=5.8, color=PALETTE["text"], wrap=True)

save_pub(fig, os.path.join(HERE, "fig5_table_ii"))
print("[fig5] wrote fig5_table_ii.{png,pdf,svg}")
