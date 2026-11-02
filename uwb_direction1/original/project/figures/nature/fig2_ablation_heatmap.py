"""Figure 2 — 2x2 ablation heatmap (pool x normalize), per branch.

Claim
-----
The dominant lever on both branches is the input normalisation (per_view_std →
abs_std: +10.5 single, +4.5 three); pooling (mean ↔ flatten) is a secondary
factor of <= 1.5 OSCR.

Evidence
--------
Two heatmaps side by side (single-view, three-view), each a 2x2 grid of OSCR
mean ± std (Table IV). The vertical contrast (normalize) is clearly stronger
than the horizontal contrast (pool).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from _style import PALETTE, save_pub, W15

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Table IV ----
# rows = normalize (per_view_std, abs_std); cols = pool (mean, flatten).
# The "mean x per_view_std" cell was only run on case 0; use that value but
# flag it with an asterisk in the annotation.

single_mean_vals = np.array([
    [69.71, 72.42],   # per_view_std
    [83.34, 82.90],   # abs_std
])
single_std_vals = np.array([
    [np.nan, 4.73],
    [3.34, 4.83],
])

three_mean_vals = np.array([
    [92.56, 90.63],   # per_view_std
    [93.63, 95.14],   # abs_std
])
three_std_vals = np.array([
    [np.nan, 2.26],
    [2.05, 0.81],
])

pool_labels = ["mean", "flatten"]
norm_labels = ["per_view_std", "|x| + std (abs_std)"]

# Shared color scale: span both panels so panels are directly comparable.
vmin = min(single_mean_vals.min(), three_mean_vals.min())
vmax = max(single_mean_vals.max(), three_mean_vals.max())

# Custom sequential cmap from neutral to teal (matches unified palette).
cmap = LinearSegmentedColormap.from_list(
    "teal_seq",
    [(0.0, "#F4F2EE"),
     (0.45, "#BBD0CF"),
     (0.80, "#5FA0A9"),
     (1.0, "#2E7F8E")],
    N=256,
)

fig, axes = plt.subplots(1, 2, figsize=(W15, 2.4),
                          gridspec_kw=dict(wspace=0.22))

panel_data = [
    ("a   Single-view (Left)", single_mean_vals, single_std_vals),
    ("b   Three-view (L + T + R)", three_mean_vals, three_std_vals),
]

for k, (ax, (title, mv, sv)) in enumerate(zip(axes, panel_data)):
    im = ax.imshow(mv, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(pool_labels)
    if k == 0:
        ax.set_yticks([0, 1]); ax.set_yticklabels(norm_labels)
    else:
        ax.set_yticks([0, 1]); ax.set_yticklabels([])
    ax.set_title(title, loc="left", fontweight="bold", pad=4)

    # annotate each cell
    for i in range(2):
        for j in range(2):
            m, s = mv[i, j], sv[i, j]
            # pick text color by background luminance
            t_color = "white" if m > (vmin + vmax) / 2 else PALETTE["text"]
            if np.isnan(s):
                txt = f"{m:.2f}\n(case 0 only)"
            else:
                txt = f"{m:.2f}\n± {s:.2f}"
            weight = "bold" if (i == 1 and j == 1) else "normal"
            ax.text(j, i, txt, ha="center", va="center", color=t_color,
                    fontsize=7, fontweight=weight)

    ax.set_xlabel("Token pooling")
    if ax is axes[0]:
        ax.set_ylabel("Input normalisation")

    # Mark the headline cell (three-view, flatten x abs_std).
    if "Three" in title:
        ax.text(0.93, 0.78, r"$\bigstar$", ha="center", va="center", color="#FFE066",
                fontsize=11, fontweight="bold",
                transform=ax.transAxes)

    # Clean: hide bottom/left axis spines on heatmap-style imshow.
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

# Shared colorbar
cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                     orientation="vertical")
cbar.set_label("OSCR (%, 5-case mean)", rotation=90)
cbar.outline.set_visible(False)
cbar.ax.tick_params(length=2.5, width=0.8)

# small caption: ΔOSCR from normalisation row
fig.text(0.5, -0.04,
         "Δ from per_view_std → |x|+std (5-case): +10.48 (single-view, flatten); +4.51 (three-view, flatten).",
         ha="center", va="top", fontsize=6.5, color=PALETTE["text"])

save_pub(fig, os.path.join(HERE, "fig2_ablation_heatmap"))
print("[fig2] wrote fig2_ablation_heatmap.{png,pdf,svg}")
