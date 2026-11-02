"""Why leave-one-radar-out reprojection fails: an observability check.

Three radars measure three ranges; the hand has three degrees of freedom.  Removing
one radar leaves 2 measurements for 3 unknowns, so there is ALWAYS a one-dimensional
unobservable direction n = null([u_a; u_b]).  The reprojection error of the held-out
view v is then governed by |u_v . n|: the more the held-out view looks along the
direction the remaining pair cannot see, the less predictable it is.

Consequence for the protocol: leave-one-radar-out is not a clean test for a 3-radar /
3-DOF system.  It becomes one if the trajectory is constrained to 2 DOF (fix the
standoff z), which is physically reasonable for the in-plane gestures.  This script
measures both.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from config import GESTURE_NAMES                                        # noqa: E402
from uwb_osr.trajectory import (viterbi_track, bspline_basis, fit_trajectory,  # noqa: E402
                                reproject, readouts, dprime, RADAR_POS, Z_REF)

CACHE = r"D:\uwb_gestures\cache"
VIEWS = ["Left", "Top", "Right"]
K_CTRL, LAM = 30, 0.05


def unit_vectors(p):
    d = p - RADAR_POS
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def main():
    p0 = np.array([0.0, 0.0, Z_REF])
    U = unit_vectors(p0)
    print("=== analytic observability at the nominal hand position (0, 0, %.2f) ===" % Z_REF)
    for v in range(3):
        keep = [u for u in range(3) if u != v]
        n = np.cross(U[keep[0]], U[keep[1]])
        n /= np.linalg.norm(n)
        print(f"  hold out {VIEWS[v]:>5s}: remaining pair {VIEWS[keep[0]]}+{VIEWS[keep[1]]}"
              f"  blind direction n = ({n[0]:+.2f}, {n[1]:+.2f}, {n[2]:+.2f})"
              f"   |u . n| = {abs(U[v] @ n):.3f}")
    print("  (|u.n| near 1 = the held-out view points almost exactly along the blind"
          " direction => its range is unpredictable in principle, not by bad fitting)")

    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    subj = np.load(f"{CACHE}/subj_clutterremoved.npy")
    B = bspline_basis(90, K_CTRL)

    print("\n=== empirical: 3-DOF fit vs 2-DOF fit (standoff pinned) ===")
    print(f"{'gesture':22s}{'loo3 mm':>9s}{'loo3 %':>8s}{'loo2 mm':>9s}{'loo2 %':>8s}"
          f"{'rms3':>7s}{'rms2':>7s}")
    agg = {"3": [], "2": [], "3r": [], "2r": []}
    read2 = {}
    for g in range(12):
        rows3, rows2, rel3, rel2, r3, r2 = [], [], [], [], [], []
        read2[g] = []
        for s in range(1, 9):
            for i in np.where((y == g) & (subj == s))[0][:6]:
                unit = np.asarray(X[i])
                tr = [viterbi_track(unit[v]) for v in range(3)]
                r = np.stack([t[0] for t in tr]); c = np.stack([t[1] for t in tr])
                f3 = fit_trajectory(r, c, (0, 1, 2), B=B, lam=LAM, mu=0.05)
                f2 = fit_trajectory(r, c, (0, 1, 2), B=B, lam=LAM, mu=200.0)
                r3.append(f3["rms"] * 1000); r2.append(f2["rms"] * 1000)
                read2[g].append(readouts(f2["p"]))
                for v in range(3):
                    keep = tuple(u for u in range(3) if u != v)
                    for mu, mm, rel in ((0.05, rows3, rel3), (200.0, rows2, rel2)):
                        out = fit_trajectory(r, c, keep, B=B, lam=LAM, mu=mu)
                        res = np.sqrt(np.mean((reproject(out["C"], B, v) - r[v]) ** 2))
                        mm.append(res * 1000)
                        rel.append(100 * res / (np.ptp(r[v]) + 1e-9))
        agg["3"] += rows3; agg["2"] += rows2; agg["3r"] += rel3; agg["2r"] += rel2
        print(f"{'G'+str(g+1)+' '+GESTURE_NAMES.get(g,''):22s}"
              f"{np.median(rows3):9.1f}{np.median(rel3):8.1f}"
              f"{np.median(rows2):9.1f}{np.median(rel2):8.1f}"
              f"{np.median(r3):7.1f}{np.median(r2):7.1f}")
    print(f"{'ALL':22s}{np.median(agg['3']):9.1f}{np.median(agg['3r']):8.1f}"
          f"{np.median(agg['2']):9.1f}{np.median(agg['2r']):8.1f}")

    print("\n=== do the physical readouts survive the 2-DOF (pinned) model? ===")

    def col(g, k):
        return np.array([d[k] for d in read2[g]])

    for name, ga, gb, key in [("horizontal G1/G2", 0, 1, "asym_x"),
                              ("vertical  G4/G3", 3, 2, "asym_y"),
                              ("chirality G10/G9", 9, 8, "area")]:
        a, b = col(ga, key), col(gb, key)
        flip = (a.mean() < b.mean())
        aa, bb = (-a, -b) if flip else (a, b)
        acc = 100 * (np.mean(aa > 0) + np.mean(bb < 0)) / 2
        print(f"  {name:20s} acc={acc:6.2f}  d'={dprime(aa, bb):6.3f}  (n={len(a)}+{len(b)})")
    print("\n  ratio sd_x/sd_y under the 2-DOF model:")
    for g in (0, 1, 2, 3, 8, 9, 10, 11):
        rr = col(g, "sd_x") / col(g, "sd_y")
        print(f"    G{g+1:<3} {np.median(rr):5.2f}")


if __name__ == "__main__":
    main()
