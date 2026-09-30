"""Figure 1 — Per-case OSCR for single-view vs three-view (MV-OSR headline).

Claim
-----
Three-view fusion lifts open-set OSCR uniformly across the five Eom test cases
(+12.24 mean), with the largest gain on the hardest case (Case 4, +20.50).

Evidence
--------
Grouped bars (5 cases x {single, three}); ΔOSCR annotations; reference line at
Eom paper's 88.42; horizontal mean ± std bands for our 1-view (82.90 ± 4.83)
and 3-view (95.14 ± 0.81) headline rows of Table II.

Data source
-----------
Table V of tables.md, verified against logs/sweep5_single_flat_absstd.log and
logs/sweep5_three_flat_absstd.log.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _style import PALETTE, save_pub, W15

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- data (Table V, OSCR columns) ----
cases  = ["Case 1", "Case 2", "Case 3", "Case 4", "Case 5"]
single = np.array([87.04, 84.35, 85.99, 73.54, 83.59])
three  = np.array([96.19, 94.80, 94.72, 94.04, 95.95])
gain   = three - single  # [+9.15, +10.45, +8.73, +20.50, +12.36]

single_mean, single_std = 82.90, 4.83
three_mean,  three_std  = 95.14, 0.81
eom_baseline = 88.42

# ---- figure ----
fig, ax = plt.subplots(figsize=(W15, 3.2))

x = np.arange(len(cases))
bw = 0.34
b1 = ax.bar(x - bw/2, single, bw,
            color=PALETTE["single"], edgecolor=PALETTE["single_dk"],
            linewidth=0.6, label="Single-view (Left)")
b2 = ax.bar(x + bw/2, three, bw,
            color=PALETTE["three"], edgecolor="white", linewidth=0.6,
            label="Three-view (L + T + R)")

# Δ-gain annotations between each pair.
for xi, (s, t, g) in enumerate(zip(single, three, gain)):
    y_top = max(s, t) + 1.0
    ax.annotate(f"+{g:.2f}", xy=(xi, y_top + 0.5), ha="center", va="bottom",
                fontsize=6.5, color=PALETTE["gain"], fontweight="bold")
    # short connector line above the higher bar
    ax.plot([xi - bw/2, xi + bw/2], [y_top, y_top], color=PALETTE["gain"],
            linewidth=0.6, solid_capstyle="round")

# bar value labels (just below top of each bar, inside)
for rect, val in zip(b1, single):
    ax.text(rect.get_x() + rect.get_width()/2, val - 2.5, f"{val:.1f}",
            ha="center", va="top", fontsize=5.8, color=PALETTE["single_dk"])
for rect, val in zip(b2, three):
    ax.text(rect.get_x() + rect.get_width()/2, val - 2.5, f"{val:.1f}",
            ha="center", va="top", fontsize=5.8, color="white", fontweight="bold")

# horizontal reference: Eom single-view OSCR = 88.42
ax.axhline(eom_baseline, color=PALETTE["eom"], linewidth=0.9, linestyle="--",
           alpha=0.9, zorder=1)
ax.text(-0.5, eom_baseline + 0.6,
        f"Eom 2025 single-view OSCR = {eom_baseline:.2f}",
        color=PALETTE["eom"], fontsize=6, ha="left", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))

# headline mean ± std bands (faint horizontal stripes)
ax.axhspan(single_mean - single_std, single_mean + single_std,
           color=PALETTE["single"], alpha=0.20, zorder=0)
ax.axhspan(three_mean - three_std, three_mean + three_std,
           color=PALETTE["three_lt"], alpha=0.30, zorder=0)
ax.text(len(cases) - 0.5, single_mean - 0.4,
        f"1-view 5-case mean = {single_mean:.2f} ± {single_std:.2f}",
        color=PALETTE["single_dk"], fontsize=5.8, va="top", ha="right",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
ax.text(len(cases) - 0.5, three_mean + 0.4,
        f"3-view 5-case mean = {three_mean:.2f} ± {three_std:.2f}",
        color=PALETTE["three"], fontsize=5.8, va="bottom", ha="right",
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))

# axes & labels
ax.set_xticks(x)
ax.set_xticklabels(cases)
ax.set_ylabel("Open-set classification rate, OSCR (%)")
ax.set_xlabel("Eom Table I test case")
ax.set_ylim(60, 102)
ax.set_yticks(np.arange(60, 101, 5))
ax.tick_params(axis="both", colors=PALETTE["text"])
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(PALETTE["text"])

# legend at top
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3,
                handlelength=1.2, handletextpad=0.5, columnspacing=1.2)

# small mean-of-gain caption (left-aligned, bottom inside the plot)
ax.text(-0.5, 61.5,
        f"ΔOSCR mean (5 cases) = +{gain.mean():.2f}",
        color=PALETTE["gain"], fontsize=6.5, ha="left", va="bottom",
        fontweight="bold")

save_pub(fig, os.path.join(HERE, "fig1_per_case_oscr"))
print("[fig1] wrote fig1_per_case_oscr.{png,pdf,svg}")
