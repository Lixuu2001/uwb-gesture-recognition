"""W2 follow-up 4: do the three views share a common phase centre?

Every model tried so far (isotropic cloud, rigid / articulated forearm, RCS
fluctuation, torso, per-view range offset) assumed the three radars observe the SAME
scatterer cloud with the SAME weights, and none of them reproduced the real residual
(47-50 mm) without destroying the track.

But the aspect differs per view: at 0.45 m standoff the Left radar sees the hand edge
-on, the Top radar sees its back, the Right radar the other edge.  The dominant
scattering centre can therefore be a DIFFERENT physical point on the hand/forearm in
each view - the views would not share a phase centre.  That inflates the fit residual
(no single point explains all three ranges) while leaving each view able to track
relative motion (every candidate point moves with the hand), which is exactly the
pattern observed.

Model: view v measures || p(t) + delta_v - P_v ||, i.e. an equivalent per-view
scatterer offset delta_v (3 params each).  This is identical to displacing radar v by
-delta_v, so it reuses `fit_trajectory(pos=...)`.  Gauge fix: mean(delta) = 0, since a
common offset is the same as translating the trajectory.

Scored on HELD-OUT frames (five 6-frame blocks) so the 9 extra parameters cannot just
absorb the residual.
"""
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, ".")
from uwb_osr.trajectory import (viterbi_track, bspline_basis, fit_trajectory,  # noqa: E402
                                reproject, RADAR_POS)

CACHE = r"D:\uwb_gestures\cache"
K_CTRL, LAM, MU = 30, 0.05, 0.05
VIEWS = ["Left", "Top", "Right"]


def fit_phase_centres(r, w, B, n_iter=5):
    """Alternate: trajectory | per-view equivalent offsets (mean-zero gauge)."""
    delta = np.zeros((3, 3))
    out = None
    for _ in range(n_iter):
        pos = RADAR_POS - delta
        out = fit_trajectory(r, w, (0, 1, 2), B=B, lam=LAM, mu=MU, pos=pos)
        p = out["p"]

        def resid(d):
            d = d.reshape(3, 3)
            d = d - d.mean(axis=0, keepdims=True)          # gauge fix
            e = []
            for v in range(3):
                rng_ = np.linalg.norm(p + d[v] - RADAR_POS[v], axis=1)
                e.append(w[v] * (rng_ - r[v]))
            return np.concatenate(e)

        sol = least_squares(resid, delta.ravel(), method="lm", max_nfev=200)
        delta = sol.x.reshape(3, 3)
        delta = delta - delta.mean(axis=0, keepdims=True)
    return out, delta


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
            cm = np.where(mask, 0.0, c)                     # train weights

            base = fit_trajectory(r, c, (0, 1, 2), B=B, lam=LAM, mu=MU)
            p0 = np.stack([reproject(base["C"], B, v) for v in range(3)])

            pc, d = fit_phase_centres(r, c, B)
            p1 = np.stack([np.linalg.norm(pc["p"] + d[v] - RADAR_POS[v], axis=1)
                           for v in range(3)])

            hb = fit_trajectory(r, cm, (0, 1, 2), B=B, lam=LAM, mu=MU)
            h0 = np.stack([reproject(hb["C"], B, v) for v in range(3)])
            hp, hd = fit_phase_centres(r, cm, B)
            h1 = np.stack([np.linalg.norm(hp["p"] + hd[v] - RADAR_POS[v], axis=1)
                           for v in range(3)])

            rows.append(dict(s=s, g=int(y[i]),
                             full0=rms(r - p0), full1=rms(r - p1),
                             hold0=rms((r - h0)[:, mask]), hold1=rms((r - h1)[:, mask]),
                             d=d, dn=np.linalg.norm(d, axis=1)))
    R = {k: np.array([q[k] for q in rows]) for k in rows[0]}

    print("=" * 74)
    print("per-view equivalent phase centres (9 params, mean-zero gauge)")
    print("=" * 74)
    print(f"  n = {len(rows)} recordings\n")
    print(f"{'':26s}{'all frames':>13s}{'held-out frames':>18s}")
    print(f"{'common phase centre':26s}{np.median(R['full0']):13.1f}{np.median(R['hold0']):18.1f}")
    print(f"{'per-view phase centres':26s}{np.median(R['full1']):13.1f}{np.median(R['hold1']):18.1f}")
    print(f"{'improvement':26s}"
          f"{100*(1-np.median(R['full1'])/np.median(R['full0'])):12.1f}%"
          f"{100*(1-np.median(R['hold1'])/np.median(R['hold0'])):17.1f}%")

    dn = np.stack(R["dn"]) * 1000
    print("\n  |delta_v| (mm), median over recordings:")
    for v in range(3):
        print(f"    {VIEWS[v]:>6s}: {np.median(dn[:, v]):7.1f}"
              f"   (p10 {np.percentile(dn[:, v],10):6.1f}, p90 {np.percentile(dn[:, v],90):6.1f})")

    d = np.stack(R["d"]) * 1000
    print("\n  mean delta_v vector (mm) - a systematic aspect effect would repeat:")
    for v in range(3):
        m = d[:, v, :].mean(axis=0)
        sd = d[:, v, :].std(axis=0)
        print(f"    {VIEWS[v]:>6s}: ({m[0]:6.1f}, {m[1]:6.1f}, {m[2]:6.1f})"
              f"   sd ({sd[0]:5.1f}, {sd[1]:5.1f}, {sd[2]:5.1f})")

    print("\n  per-subject median |delta| (mm):")
    print("          " + "".join(f"{'HV'+str(s):>8s}" for s in range(1, 9)))
    for v in range(3):
        print(f"    {VIEWS[v]:>6s}" + "".join(
            f"{np.median(dn[R['s'] == s, v]):8.1f}" for s in range(1, 9)))


if __name__ == "__main__":
    main()
