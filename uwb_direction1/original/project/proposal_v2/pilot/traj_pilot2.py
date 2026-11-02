"""Pilot round 2: better readouts on the reconstructed trajectory.

Round 1 found: chirality (loop signed area) separates CW/CCW at d'=2.74 with zero
training, but NET start-to-end displacement is at chance for the swipes.  The
likely reason is that a 4.5 s recording contains the swipe AND the hand coming
back, so net displacement cancels.  So test:
  * axis selectivity      : does a horizontal swipe move mostly in x?  (no sign needed)
  * excursion sign        : sign of the largest excursion from the median position
  * velocity asymmetry    : E[v^3] - the swipe stroke is fast, the reset is slow
  * how many strokes      : zero-crossing count of the velocity, tells us whether
                            the subject repeated the gesture inside the window
"""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\19252\AppData\Local\Temp\claude\D--uwb-gestures\0021ac01-3388-4472-bdc8-d381e152d862\scratchpad")
from traj_pilot import (range_track, solve_dp, smooth, signed_area, dprime,
                        CACHE, BIN_M, FPS)


def readouts(dp):
    x, y = dp[:, 0], dp[:, 1]
    vx, vy = np.gradient(x) * FPS, np.gradient(y) * FPS
    out = {}
    out["sx"], out["sy"] = x.std(), y.std()
    for nm, a, v in (("x", x, vx), ("y", y, vy)):
        c = a - np.median(a)
        out[f"exc_{nm}"] = c[np.argmax(np.abs(c))]      # signed largest excursion
        out[f"asym_{nm}"] = np.mean(v ** 3)             # fast stroke vs slow reset
        out[f"nz_{nm}"] = int(np.sum(np.diff(np.sign(smooth(v, 7))) != 0))
    out["area"] = signed_area(x - x.mean(), y - y.mean())
    out["vmax"] = np.max(np.hypot(vx, vy))
    return out


def main():
    z0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    subj = np.load(f"{CACHE}/subj_clutterremoved.npy")
    want = [0, 1, 2, 3, 8, 9, 10, 11]
    per_g = 30
    F = {g: [] for g in want}
    S = {g: [] for g in want}
    for g in want:
        for s in range(1, 9):
            for i in np.where((y == g) & (subj == s))[0][:per_g]:
                unit = np.asarray(X[i])
                rr = np.stack([smooth(range_track(unit[v])[0], 5) for v in range(3)])
                F[g].append(readouts(solve_dp(rr - rr[:, :1], z0)))
                S[g].append(s)
    keys = list(F[want[0]][0].keys())
    A = {g: {k: np.array([f[k] for f in F[g]]) for k in keys} for g in want}
    S = {g: np.array(S[g]) for g in want}

    names = {0: "G1 L->R", 1: "G2 R->L", 2: "G3 U->D", 3: "G4 D->U",
             8: "G9 CW", 9: "G10 CCW", 10: "G11 push", 11: "G12 empty"}
    print(f"z0={z0} m   n={len(F[0])} per gesture\n")
    print("--- amplitude of the reconstructed motion (mm) and stroke count ---")
    print(f"{'gesture':10s} {'sd_x':>7s} {'sd_y':>7s} {'vmax(m/s)':>10s} {'nzero_x':>8s} {'nzero_y':>8s}")
    for g in want:
        print(f"{names[g]:10s} {A[g]['sx'].mean()*1000:7.1f} {A[g]['sy'].mean()*1000:7.1f} "
              f"{A[g]['vmax'].mean():10.2f} {A[g]['nz_x'].mean():8.1f} {A[g]['nz_y'].mean():8.1f}")

    def rule(ga, gb, k, flip=False):
        a, b = A[ga][k], A[gb][k]
        if flip:
            a, b = -a, -b
        acc = 100 * (np.mean(a > 0) + np.mean(b < 0)) / 2
        per = []
        for s in range(1, 9):
            aa, bb = a[S[ga] == s], b[S[gb] == s]
            per.append(100 * (np.mean(aa > 0) + np.mean(bb < 0)) / 2)
        return acc, dprime(a, b), np.mean(per), np.std(per)

    print("\n--- zero-threshold sign rules, no training (acc%, d', per-subject acc) ---")
    tests = [(0, 1, "exc_x", "horizontal: largest excursion"),
             (0, 1, "asym_x", "horizontal: velocity asymmetry E[v^3]"),
             (3, 2, "exc_y", "vertical: largest excursion"),
             (3, 2, "asym_y", "vertical: velocity asymmetry E[v^3]"),
             (9, 8, "area", "chirality: loop signed area")]
    for ga, gb, k, nm in tests:
        acc, d, pm, ps = rule(ga, gb, k)
        acc2, d2, pm2, ps2 = rule(ga, gb, k, flip=True)
        if acc2 > acc:
            acc, d, pm, ps, nm = acc2, -d2, pm2, ps2, nm + " [sign flipped]"
        print(f"  {nm:42s} acc={acc:6.2f}  d'={d:6.3f}  per-subj={pm:6.2f}+-{ps:5.2f}")

    print("\n--- axis selectivity: sd_x/sd_y ratio (horizontal swipes should be >1) ---")
    for g in want:
        r = A[g]["sx"] / (A[g]["sy"] + 1e-9)
        print(f"  {names[g]:10s} median ratio {np.median(r):6.2f}")


if __name__ == "__main__":
    main()
