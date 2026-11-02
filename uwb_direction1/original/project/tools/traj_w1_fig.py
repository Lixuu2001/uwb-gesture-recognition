"""W1 closing: the replacement self-consistency metric + the 12-gesture figure.

Leave-one-radar-out turned out to be ill-posed (see tools/traj_w1_obs.py: removing a
radar always leaves a blind direction that the held-out view looks 77-83% along).
The valid self-consistency test that keeps all three views observable is a TEMPORAL
holdout: drop contiguous 6-frame blocks from ALL views, fit, predict the held-out
frames.  Reported in mm and as a fraction of the range excursion.
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from config import GESTURE_NAMES                                        # noqa: E402
from uwb_osr.trajectory import (viterbi_track, bspline_basis, fit_trajectory,  # noqa: E402
                                reproject)

CACHE = r"D:\uwb_gestures\cache"
K_CTRL, LAM, MU = 30, 0.05, 0.05
SHORT = {0: "G1 L>R", 1: "G2 R>L", 2: "G3 U>D", 3: "G4 D>U", 4: "G5 diag LR-UD",
         5: "G6 diag LR-DU", 6: "G7 diag RL-UD", 7: "G8 diag RL-DU",
         8: "G9 clockwise", 9: "G10 anti-clock", 10: "G11 push", 11: "G12 empty"}


def tracks_of(unit):
    tr = [viterbi_track(unit[v]) for v in range(3)]
    return np.stack([t[0] for t in tr]), np.stack([t[1] for t in tr])


def main():
    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    subj = np.load(f"{CACHE}/subj_clutterremoved.npy")
    B = bspline_basis(90, K_CTRL)

    mask = np.zeros(90, bool)
    for s in range(0, 90, 20):
        mask[s:s + 6] = True

    print("=== temporal block holdout (6-frame blocks, 30 of 90 frames, all 3 views) ===")
    print(f"{'gesture':22s}{'mm':>8s}{'% of excursion':>16s}{'excursion mm':>14s}")
    allmm, allrel = [], []
    for g in range(12):
        mm, rel, amp = [], [], []
        for s in range(1, 9):
            for i in np.where((y == g) & (subj == s))[0][:8]:
                r, c = tracks_of(np.asarray(X[i]))
                c2 = c.copy(); c2[:, mask] = 0.0
                out = fit_trajectory(r, c2, (0, 1, 2), B=B, lam=LAM, mu=MU)
                pred = np.stack([reproject(out["C"], B, v) for v in range(3)])
                e = np.sqrt(np.mean((pred[:, mask] - r[:, mask]) ** 2))
                a = np.median(np.ptp(r, axis=1))
                mm.append(e * 1000); rel.append(100 * e / (a + 1e-9)); amp.append(a * 1000)
        if g != 11:
            allmm += mm; allrel += rel
        print(f"{SHORT[g]:22s}{np.median(mm):8.1f}{np.median(rel):16.1f}{np.median(amp):14.1f}")
    print(f"{'ALL (G12 excluded)':22s}{np.median(allmm):8.1f}{np.median(allrel):16.1f}")

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))
    for g, ax in enumerate(axes.ravel()):
        for i in np.where((y == g) & (subj == 1))[0][:5]:
            r, c = tracks_of(np.asarray(X[i]))
            p = fit_trajectory(r, c, (0, 1, 2), B=B, lam=LAM, mu=MU)["p"] * 1000
            x, yy = p[:, 0] - p[:, 0].mean(), p[:, 1] - p[:, 1].mean()
            ax.plot(x, yy, lw=1.3, alpha=0.85)
            ax.plot(x[0], yy[0], "o", ms=4, color="k", alpha=0.6)
        ax.set_title(SHORT[g], fontsize=10)
        ax.set_xlim(-450, 450); ax.set_ylim(-450, 450)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        if g % 4 == 0:
            ax.set_ylabel("y (mm)")
        if g >= 8:
            ax.set_xlabel("x (mm)")
    fig.suptitle("Reconstructed hand trajectories, all 12 gestures (subject HV1, 5 repetitions each)\n"
                 "Viterbi range tracking + nonlinear B-spline trilateration, no training",
                 fontsize=12)
    fig.tight_layout()
    out = "proposal_v2/pilot/traj_w1_12gestures.png"
    fig.savefig(out, dpi=120)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
