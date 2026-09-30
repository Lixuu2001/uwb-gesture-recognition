"""W2 follow-up 3: is the residual gap a per-view RANGE OFFSET?

`w2_body2.py` ruled out the three obvious extended-target mechanisms: every setting
that reproduces the real three-view residual (49.7 mm) also destroys the trajectory
(RMS error 100-500 mm), whereas real reconstructions stay usable (the W1 readouts
score 84-94%).  So the missing physics must inflate the FIT residual without
breaking the TRACK - which is exactly what a static per-view range bias does
(antenna/cable delay, per-radar standoff error, subject standing position).

Test: give each recording three free per-view range offsets, estimated by coordinate
descent (the optimal offset for a fixed trajectory is the mean residual of that view).
Guard against trivially absorbing the residual by scoring on HELD-OUT frames: the
offsets and the trajectory are fitted on 60 of 90 frames and scored on five 6-frame
blocks that were never seen.

If the offsets are also CONSISTENT across recordings and subjects, they are a
calibration constant of the dataset rather than a per-recording nuisance.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from uwb_osr.trajectory import (viterbi_track, bspline_basis, fit_trajectory,  # noqa: E402
                                reproject, RADAR_POS)

CACHE = r"D:\uwb_gestures\cache"
K_CTRL, LAM, MU = 30, 0.05, 0.05
VIEWS = ["Left", "Top", "Right"]


def fit_with_offsets(r, c, B, n_iter=6, mask=None):
    """Alternate: fit trajectory -> update per-view offsets = mean residual."""
    w = c.copy()
    if mask is not None:
        w = w.copy()
        w[:, mask] = 0.0
    b = np.zeros(3)
    out = None
    for _ in range(n_iter):
        out = fit_trajectory(r - b[:, None], w, (0, 1, 2), B=B, lam=LAM, mu=MU)
        pred = np.stack([reproject(out["C"], B, v) for v in range(3)])
        resid = r - pred
        wt = w / (w.sum(axis=1, keepdims=True) + 1e-12)
        b = (resid * wt).sum(axis=1)
    return out, b


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) * 1000)


def main():
    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    subj = np.load(f"{CACHE}/subj_clutterremoved.npy")
    B = bspline_basis(90, K_CTRL)

    mask = np.zeros(90, bool)
    for s in range(0, 90, 20):
        mask[s:s + 6] = True

    rng = np.random.default_rng(0)
    rows = []
    for s in range(1, 9):
        idx = np.where((subj == s) & (y < 11))[0]
        for i in rng.choice(idx, 8, replace=False):
            unit = np.asarray(X[i])
            tr = [viterbi_track(unit[v]) for v in range(3)]
            r = np.stack([t[0] for t in tr])
            c = np.stack([t[1] for t in tr])

            base = fit_trajectory(r, c, (0, 1, 2), B=B, lam=LAM, mu=MU)
            pred0 = np.stack([reproject(base["C"], B, v) for v in range(3)])
            off_out, b = fit_with_offsets(r, c, B)
            pred1 = np.stack([reproject(off_out["C"], B, v) for v in range(3)]) + b[:, None]

            # held-out-frame scoring
            hb = fit_trajectory(r, np.where(mask, 0.0, c), (0, 1, 2), B=B, lam=LAM, mu=MU)
            ph0 = np.stack([reproject(hb["C"], B, v) for v in range(3)])
            ho, hbias = fit_with_offsets(r, c, B, mask=mask)
            ph1 = np.stack([reproject(ho["C"], B, v) for v in range(3)]) + hbias[:, None]

            rows.append(dict(s=s, g=int(y[i]),
                             full0=rms(r - pred0), full1=rms(r - pred1),
                             hold0=rms((r - ph0)[:, mask]), hold1=rms((r - ph1)[:, mask]),
                             b=b))
    R = {k: np.array([q[k] for q in rows]) for k in rows[0]}

    print("=" * 74)
    print("per-view range offsets: does the residual gap close?")
    print("=" * 74)
    print(f"  n = {len(rows)} recordings, 8 per subject\n")
    print(f"{'':22s}{'all frames':>13s}{'held-out frames':>18s}")
    print(f"{'no offsets':22s}{np.median(R['full0']):13.1f}{np.median(R['hold0']):18.1f}")
    print(f"{'+ per-view offsets':22s}{np.median(R['full1']):13.1f}{np.median(R['hold1']):18.1f}")
    d_full = 100 * (1 - np.median(R['full1']) / np.median(R['full0']))
    d_hold = 100 * (1 - np.median(R['hold1']) / np.median(R['hold0']))
    print(f"{'improvement':22s}{d_full:12.1f}%{d_hold:17.1f}%")

    print("\n  estimated offsets (mm), median over recordings:")
    b = np.stack(R["b"]) * 1000
    for v in range(3):
        print(f"    {VIEWS[v]:>6s}: {np.median(b[:, v]):8.1f}   "
              f"(p10 {np.percentile(b[:, v], 10):7.1f}, p90 {np.percentile(b[:, v], 90):7.1f}, "
              f"sd {b[:, v].std():6.1f})")

    print("\n  per-subject median offsets (mm) - a dataset constant would repeat:")
    print("          " + "".join(f"{'HV'+str(s):>9s}" for s in range(1, 9)))
    for v in range(3):
        print(f"    {VIEWS[v]:>6s}" + "".join(
            f"{np.median(b[R['s'] == s, v]):9.1f}" for s in range(1, 9)))

    # how much of the spread is between-subject vs within-subject?
    print("\n  variance split of the offsets:")
    for v in range(3):
        mu_s = np.array([np.median(b[R["s"] == s, v]) for s in range(1, 9)])
        within = np.mean([b[R["s"] == s, v].std() for s in range(1, 9)])
        print(f"    {VIEWS[v]:>6s}: between-subject sd {mu_s.std():6.1f} mm | "
              f"within-subject sd {within:6.1f} mm")


if __name__ == "__main__":
    main()
