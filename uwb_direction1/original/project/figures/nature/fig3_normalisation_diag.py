"""Figure 3 — Normalisation diagnostic (case 0, single-view, flatten + inspector).

Claim
-----
Taking the amplitude (|x|) of the clutter-removed CIR before per-view std-
normalisation ("abs_std") is the best of four normalisation options on every
OSR metric.

Evidence
--------
Four normalisation modes on the x-axis. For each, three bars: INSPECTOR OSCR,
AUROC, MSP OSCR (Table VI). The abs_std cluster dominates the other three on
INSPECTOR OSCR and AUROC; MSP OSCR continues to lag, reinforcing the value of
the prototype-distance inspector head.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from _style import PALETTE, save_pub, W1

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Table VI ----
modes = ["none\n(signed CIR)", "per_view_std\n(legacy)",
         "abs\n(|x| only)", "abs_std\n(|x| + std)"]
insp_oscr = np.array([66.44, 74.85, 81.34, 85.15])
auroc     = np.array([71.94, 75.95, 82.71, 86.13])
msp_oscr  = np.array([66.75, 72.40, 76.55, 77.83])

fig, ax = plt.subplots(figsize=(W1, 3.0))

x = np.arange(len(modes))
bw = 0.26

b1 = ax.bar(x - bw, insp_oscr, bw, color=PALETTE["three"],
            edgecolor="white", linewidth=0.5, label="INSPECTOR OSCR")
b2 = ax.bar(x,        auroc, bw, color=PALETTE["three_lt"],
            edgecolor="white", linewidth=0.5, label="AUROC")
b3 = ax.bar(x + bw, msp_oscr, bw, color=PALETTE["msp"],
            edgecolor="white", linewidth=0.5, label="MSP OSCR")

for group in (b1, b2, b3):
    for rect in group:
        v = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, v + 0.5, f"{v:.1f}",
                ha="center", va="bottom", fontsize=5.5,
                color=PALETTE["text"])

# Highlight the abs_std winner
ax.add_patch(plt.Rectangle((x[-1] - 1.5*bw, 60), 3*bw, 30,
                           fill=False, edgecolor=PALETTE["gain"],
                           linewidth=1.0, linestyle=":", zorder=0))
ax.text(x[-1], 90.5, r"$\bigstar$ best", ha="center", va="bottom",
        color=PALETTE["gain"], fontsize=6.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(modes)
ax.set_ylabel("Score (%, case 0)")
ax.set_ylim(60, 95)
ax.set_yticks(np.arange(60, 95.1, 5))
ax.tick_params(axis="both", colors=PALETTE["text"])
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(PALETTE["text"])

leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.16), ncol=3,
                handlelength=1.0, handletextpad=0.4, columnspacing=0.9)

# trailing caption
fig.text(0.5, -0.10,
         "Δ INSPECTOR OSCR from `none` → `abs_std`: +18.71 pts.",
         ha="center", va="top", fontsize=6.2, color=PALETTE["text"])

save_pub(fig, os.path.join(HERE, "fig3_normalisation_diag"))
print("[fig3] wrote fig3_normalisation_diag.{png,pdf,svg}")
