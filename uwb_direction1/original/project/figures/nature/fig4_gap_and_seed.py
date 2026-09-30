"""Figure 4 — closed-vs-OSR gap (three-view) + Case-4 seed stability (single-view).

Claim
-----
(a) Closed-set accuracy is saturated across all four pool x normalize configurations
on three views (97-99%); the meaningful spread appears in OSCR, where the
abs_std rows pull ahead. The "headline gap" between closed-set and OSCR is what
the manuscript should defend, not closed-set alone.

(b) On Case 4 single-view, INSPECTOR < MSP reproduces across three random
seeds (Table VII). This is a *reproducible* failure mode of the single-view
prototype-distance inspector on "near-relative" unknown gestures, and is
exactly what the three-view variance / disagreement signal repairs (Fig. 1,
Case 4 gain = +20.50).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from _style import PALETTE, save_pub, W15, W2

HERE = os.path.dirname(os.path.abspath(__file__))

# ============= panel (a) closed vs OSCR for 3-view, 4 configs =============
configs = ["mean ×\nper_view_std\n(case 0)",
           "flatten ×\nper_view_std",
           "mean ×\nabs_std",
           "flatten ×\nabs_std $\\bigstar$"]
configs_short = ["m / pvs", "f / pvs", "m / abs", "f / abs $\\bigstar$"]
closed3 = np.array([99.08, 96.98, 97.62, 97.77])
oscr3   = np.array([92.56, 90.63, 93.63, 95.14])

# ============= panel (b) Case-4 seed stability ============================
seeds = ["seed 0", "seed 1", "seed 2", "mean"]
insp = np.array([73.18, 71.46, 73.40, 72.68])
msp  = np.array([79.36, 76.93, 73.81, 76.70])

# -------- figure --------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.0),
                          gridspec_kw=dict(wspace=0.28, width_ratios=[1.1, 1.0]))

# ---------- (a) ----------
ax = axes[0]
x = np.arange(len(configs))
bw = 0.36
ax.bar(x - bw/2, closed3, bw, color=PALETTE["single"],
       edgecolor=PALETTE["single_dk"], linewidth=0.5, label="Closed-set acc.")
ax.bar(x + bw/2, oscr3, bw, color=PALETTE["three"],
       edgecolor="white", linewidth=0.5, label="OSCR")
for xi, (c, o) in enumerate(zip(closed3, oscr3)):
    ax.text(xi - bw/2, c + 0.4, f"{c:.2f}", ha="center", va="bottom",
            fontsize=5.8, color=PALETTE["single_dk"])
    ax.text(xi + bw/2, o + 0.4, f"{o:.2f}", ha="center", va="bottom",
            fontsize=5.8, color=PALETTE["three"], fontweight="bold")
    # gap annotation
    ax.annotate("", xy=(xi + bw/2, o), xytext=(xi + bw/2, c),
                arrowprops=dict(arrowstyle="-|>,head_width=0.15,head_length=0.25",
                                color=PALETTE["gain"], lw=0.7),
                annotation_clip=False)
    ax.text(xi + bw/2 + 0.06, (c + o)/2,
            f"-{c-o:.1f}", color=PALETTE["gain"], fontsize=5.5,
            va="center", ha="left")

ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=6.0)
ax.set_xlabel("pool / normalize     ( m=mean, f=flatten, pvs=per_view_std, abs=abs_std )",
              fontsize=6.0)
ax.set_ylabel("Score (%, three-view, 5-case mean)")
ax.set_title("a   Closed-set vs OSR gap (3-view)", loc="left", fontweight="bold", pad=4)
ax.set_ylim(85, 102)
ax.set_yticks(np.arange(85, 100.1, 3))
ax.tick_params(axis="both", colors=PALETTE["text"])
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(PALETTE["text"])
ax.legend(loc="lower left", handlelength=1.0, handletextpad=0.4)

# ---------- (b) ----------
ax = axes[1]
x = np.arange(len(seeds))
bw = 0.34
b1 = ax.bar(x - bw/2, insp, bw, color=PALETTE["three"],
            edgecolor="white", linewidth=0.5, label="INSPECTOR OSCR")
b2 = ax.bar(x + bw/2, msp, bw, color=PALETTE["msp"],
            edgecolor="white", linewidth=0.5, label="MSP OSCR")

# annotate which one is higher per seed (always MSP for case 4 1-view)
for xi, (a, b) in enumerate(zip(insp, msp)):
    ax.text(xi - bw/2, a + 0.3, f"{a:.2f}", ha="center", va="bottom",
            fontsize=5.8, color=PALETTE["three"])
    ax.text(xi + bw/2, b + 0.3, f"{b:.2f}", ha="center", va="bottom",
            fontsize=5.8, color=PALETTE["msp"])
    # gap annotation
    if b > a:
        ax.annotate(f"MSP > INSP\nby {b-a:.2f}",
                    xy=(xi, max(a, b) + 1.5),
                    ha="center", va="bottom", fontsize=5.5,
                    color=PALETTE["eom"])

# emphasize "mean" group
ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, color="#F7E9DC", alpha=0.6, zorder=0)

ax.set_xticks(x); ax.set_xticklabels(seeds)
ax.set_ylabel("OSCR (%, Case 4, single-view)")
ax.set_title("b   Case 4 reproducibility (single-view, flatten × abs_std)",
             loc="left", fontweight="bold", pad=4)
ax.set_ylim(65, 90)
ax.set_yticks(np.arange(65, 90.1, 5))
ax.tick_params(axis="both", colors=PALETTE["text"])
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(PALETTE["text"])
ax.legend(loc="upper right", handlelength=1.0, handletextpad=0.4)

# caption
fig.text(0.5, -0.04,
         "(a) Closed-set is largely saturated; ranking is set by OSCR. "
         "(b) MSP > INSPECTOR on Case 4 is reproduced across 3 seeds — the failure mode that the three-view disagreement signal repairs (Fig. 1, +20.50).",
         ha="center", va="top", fontsize=6.2, color=PALETTE["text"])

save_pub(fig, os.path.join(HERE, "fig4_gap_and_seed"))
print("[fig4] wrote fig4_gap_and_seed.{png,pdf,svg}")
