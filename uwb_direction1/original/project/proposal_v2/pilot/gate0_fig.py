"""Gate 0 figure for the block-two technical-route report.

Reads the frozen simulation output logs/w3_gate0_sim.npz (produced by
tools/w3_gate0_sim.py) and draws two panels:

  left  : true wrap rate vs the FIT model risk score R^C, one point per level,
          with the identity line and the level 6->7 inversion marked;
  right : the Viterbi/B-spline fit RMS per level, which is what breaks at the
          top level.

Nothing here is a criterion.  The verdict lives in the log; this only draws it.
Run:  python proposal_v2/pilot/gate0_fig.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("logs/w3_gate0_sim.npz")
wrap = d["wrap_pct"]
rc = d["rc_level"] * 100.0
rms = d["rms"]                               # (8, 10), already in mm
strokes = d["strokes"]

fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.3))

# ---- left: ordering ------------------------------------------------------
a = ax[0]
a.plot([0, 95], [0, 95], color="0.75", lw=1.0, ls="--", zorder=1)
a.plot(wrap[:7], rc[:7], "-o", color="#1f4e79", lw=1.6, ms=6, zorder=3,
       label=r"levels 0--6")
a.plot(wrap[6:], rc[6:], "-o", color="#c00000", lw=1.6, ms=6, zorder=3,
       label=r"level 6$\to$7 (inversion)")
for i, (x, y, s) in enumerate(zip(wrap, rc, strokes)):
    dy = -7 if i == 7 else 4
    a.annotate(f"L{i}\n{s:g}", (x, y), textcoords="offset points",
               xytext=(6, dy), fontsize=7.5, color="0.25")
a.set_xlabel("true wrap rate of the known path  [%]")
a.set_ylabel(r"FIT model risk score  $R^C$  [%]")
a.set_title(r"Gate 0: Spearman $\rho=+0.9762$  (PASS, $\geq 0.7$)", fontsize=10)
a.set_xlim(-5, 100)
a.set_ylim(-5, 100)
a.grid(alpha=0.3)
a.legend(fontsize=8, loc="upper left")

# ---- right: what breaks at the top level ---------------------------------
b = ax[1]
med = np.median(rms, axis=1)
b.boxplot([rms[i] for i in range(8)], positions=np.arange(8), widths=0.55,
          medianprops=dict(color="#c00000"))
b.plot(np.arange(8), med, color="#1f4e79", lw=1.2, alpha=0.6, zorder=1)
b.set_xticks(np.arange(8))
b.set_xticklabels([f"L{i}\n{s:g}" for i, s in enumerate(strokes)], fontsize=8)
b.set_xlabel("level (number of strokes at fixed amplitude 0.15 m)")
b.set_ylabel("three-view fit RMS  [mm]")
b.set_title("the top level is where range tracking degrades", fontsize=10)
b.grid(alpha=0.3, axis="y")

fig.suptitle("Gate 0 ordering gate, 8 levels $\\times$ 10 renders, single scattering centre",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = "proposal_v2/pilot/w3_gate0_sim.png"
fig.savefig(out, dpi=150)
print("[saved]", out)
