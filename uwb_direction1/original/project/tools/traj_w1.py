"""W1 of research point 2.1: formal trajectory reconstruction + evaluation.

Upgrades over the先导 (`proposal_v2/pilot/traj_pilot*.py`):
  * Viterbi range tracking instead of a greedy per-frame centroid
  * nonlinear solve on a B-spline trajectory with temporal regularisation, instead of
    a linearised per-frame closed form
  * LEAK-FREE leave-one-radar-out reprojection (the held-out view touches nothing:
    not the fit, not the weights, not the initialisation)
  * all 12 gestures, per-subject folds
  * a model-based frame-alignment test (shift one view, refit, see which shift wins)

Hyper-parameters K=30, lam=0.05 were selected on a CONTIGUOUS-BLOCK temporal holdout
(fit on 60 of 90 frames, predict five held-out 6-frame blocks), which is a different
axis from the headline leave-one-view metric, so the selection is not circular.

Usage:  python tools/traj_w1.py [--reps 20] [--out logs/traj_w1.npz]
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from config import GESTURE_NAMES                                   # noqa: E402
from uwb_osr.trajectory import (viterbi_track, bspline_basis, fit_trajectory,   # noqa: E402
                                reproject, readouts, dprime, FPS)

CACHE = r"D:\uwb_gestures\cache"
VIEWS = ["Left", "Top", "Right"]
K_CTRL, LAM, MU = 30, 0.05, 0.05


def sign_rule(a, b, flip=False):
    """Zero-threshold two-way accuracy: a should be positive, b negative."""
    a, b = (-np.asarray(a), -np.asarray(b)) if flip else (np.asarray(a), np.asarray(b))
    return 100 * (np.mean(a > 0) + np.mean(b < 0)) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20, help="repetitions per gesture per subject")
    ap.add_argument("--out", default="logs/traj_w1.npz")
    ap.add_argument("--sync-n", type=int, default=150, help="samples for the alignment test")
    args = ap.parse_args()

    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    subj = np.load(f"{CACHE}/subj_clutterremoved.npy")
    B = bspline_basis(90, K_CTRL)
    print(f"K={K_CTRL} lam={LAM} mu={MU}  reps={args.reps}/gesture/subject", flush=True)

    rows = []
    t0 = time.time()
    for g in range(12):
        for s in range(1, 9):
            for i in np.where((y == g) & (subj == s))[0][:args.reps]:
                unit = np.asarray(X[i])
                tr = [viterbi_track(unit[v]) for v in range(3)]
                r = np.stack([t[0] for t in tr])
                c = np.stack([t[1] for t in tr])

                full = fit_trajectory(r, c, (0, 1, 2), B=B, lam=LAM, mu=MU)
                rd = readouts(full["p"])

                # leave-one-radar-out: the held-out view enters nothing
                loo_mm, loo_rel = [], []
                for v in range(3):
                    keep = tuple(u for u in range(3) if u != v)
                    out = fit_trajectory(r, c, keep, B=B, lam=LAM, mu=MU)
                    pred = reproject(out["C"], B, v)
                    res = np.sqrt(np.mean((pred - r[v]) ** 2))
                    amp = np.ptp(r[v])
                    loo_mm.append(res * 1000)
                    loo_rel.append(100 * res / (amp + 1e-9))

                rows.append(dict(g=g, s=s, i=int(i), rms=full["rms"] * 1000,
                                 z=float(full["p"][:, 2].mean()),
                                 loo_mm=loo_mm, loo_rel=loo_rel, **rd))
        print(f"  G{g+1} done  ({time.time()-t0:.0f}s)", flush=True)

    R = {k: np.array([row[k] for row in rows]) for k in rows[0]}

    # ---------------------------------------------------------------- tables
    print("\n=== A. fit quality ===")
    print(f"  3-view fit rms          : {np.median(R['rms']):6.1f} mm (median)")
    print(f"  solved standoff z       : {np.median(R['z']):6.3f} m")
    print(f"  leave-one-view residual : {np.median(R['loo_mm']):6.1f} mm  "
          f"= {np.median(R['loo_rel']):5.1f}% of that view's range excursion")
    for v in range(3):
        print(f"     held-out {VIEWS[v]:>5s}      : {np.median(R['loo_mm'][:, v]):6.1f} mm  "
              f"= {np.median(R['loo_rel'][:, v]):5.1f}%")

    print("\n=== B. axis selectivity (median over samples) ===")
    print(f"{'gesture':22s}{'sd_x(mm)':>10s}{'sd_y(mm)':>10s}{'ratio':>8s}{'path(mm)':>10s}")
    for g in range(12):
        m = R["g"] == g
        print(f"{'G'+str(g+1)+' '+GESTURE_NAMES.get(g,''):22s}"
              f"{np.median(R['sd_x'][m])*1000:10.1f}{np.median(R['sd_y'][m])*1000:10.1f}"
              f"{np.median(R['sd_x'][m]/R['sd_y'][m]):8.2f}"
              f"{np.median(R['path_len'][m])*1000:10.1f}")

    print("\n=== C. zero-threshold sign rules, per-subject folds ===")
    tests = [("horizontal G1/G2 (E[vx^3])", 0, 1, "asym_x"),
             ("vertical  G4/G3 (E[vy^3])", 3, 2, "asym_y"),
             ("chirality G10/G9 (loop area)", 9, 8, "area")]
    summary = {}
    for name, ga, gb, key in tests:
        A = R[key][R["g"] == ga]; Bb = R[key][R["g"] == gb]
        flip = sign_rule(A, Bb, True) > sign_rule(A, Bb, False)
        pooled = sign_rule(A, Bb, flip)
        per = []
        for s in range(1, 9):
            a = R[key][(R["g"] == ga) & (R["s"] == s)]
            b = R[key][(R["g"] == gb) & (R["s"] == s)]
            per.append(sign_rule(a, b, flip))
        per = np.array(per)
        d = dprime(A, Bb) * (-1 if flip else 1)
        summary[name] = (pooled, d, per)
        print(f"  {name:30s} pooled={pooled:6.2f}  d'={d:6.3f}  "
              f"per-subj={per.mean():6.2f}+-{per.std():5.2f}  "
              f"folds>75%: {int((per > 75).sum())}/8")
        print(f"      {'  '.join(f'HV{s}:{p:5.1f}' for s, p in zip(range(1,9), per))}")

    # ------------------------------------------------- D. alignment test
    print("\n=== D. frame alignment: shift one view, refit, which shift wins? ===")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(rows), min(args.sync_n, len(rows)), replace=False)
    shifts = np.arange(-10, 11)
    best_shift = {v: [] for v in range(3)}
    for n, ridx in enumerate(pick):
        i = R["i"][ridx]
        unit = np.asarray(X[i])
        tr = [viterbi_track(unit[v]) for v in range(3)]
        r = np.stack([t[0] for t in tr]); c = np.stack([t[1] for t in tr])
        for v in range(3):
            errs = []
            for d in shifts:
                rr, cc = r.copy(), c.copy()
                rr[v] = np.roll(r[v], d); cc[v] = np.roll(c[v], d)
                if d > 0:
                    cc[v, :d] = 0.0
                elif d < 0:
                    cc[v, d:] = 0.0
                errs.append(fit_trajectory(rr, cc, (0, 1, 2), B=B, lam=LAM, mu=MU)["rms"])
            best_shift[v].append(shifts[int(np.argmin(errs))])
    for v in range(3):
        bs = np.array(best_shift[v])
        print(f"  shifting {VIEWS[v]:>5s}: best shift median={np.median(bs):5.1f} "
              f"mode={np.bincount(bs+10).argmax()-10:3d}  "
              f"|shift|<=1 in {100*np.mean(np.abs(bs)<=1):5.1f}%  sd={bs.std():5.2f}")

    np.savez(args.out, **{k: v for k, v in R.items()},
             best_shift=np.array([best_shift[v] for v in range(3)]))
    print(f"\nsaved -> {args.out}   ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
