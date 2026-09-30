"""Pilot X3 - is "how imaginable is a held-out view" predictable from LOCAL geometry?

The worry that motivates this: UWB-Gestures has only THREE fixed radar positions, so a
leave-one-view-out study looks like it has only three configurations, which is far too
few to claim anything about sensor placement.

But the radars are fixed and the HAND MOVES.  The Jacobian that decides what is
observable is evaluated at the hand's position, so every sample - in fact every frame -
sits at a different geometry.  If the imaginability of a held-out view tracks a
geometric quantity computed at the hand's own position, the effective number of
configurations is the number of samples, not three.

This script dumps, per held-out-view instance:
  * the geometry at the hand's mean position: the null direction of the two remaining
    views, the share of that null direction the standoff prior can pin (its z^2), and
    the alignment of the dropped view's line of sight with it
  * the measured error: held-out range error (mm) and envelope correlation of the
    physically rendered held-out view
and reports Spearman correlations between them.

Usage:  python proposal_v2/pilot/xview_geom.py [--reps 5]
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from uwb_osr.forward import clutter_filter, render
from uwb_osr.trajectory import (BIN_M, M_BINS, RADAR_POS, bspline_basis,
                                fit_trajectory, reproject, viterbi_track)

CACHE = r"D:\uwb_gestures\cache"
VIEWS = ["Left", "Top", "Right"]
GAINS = (0.876, 1.043, 1.081)


def analytic_env(x):
    n = x.shape[-1]
    h = np.zeros(n); h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0; h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(np.fft.fft(x, axis=-1) * h, axis=-1))


def env_corr(pred, real):
    ep, er = analytic_env(pred.astype(np.float64)), analytic_env(real.astype(np.float64))
    a, b = ep.ravel() - ep.mean(), er.ravel() - er.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def geometry(p, v, pos=RADAR_POS):
    """Local observability geometry at hand position p when view v is dropped."""
    U = np.array([(p - P) / np.linalg.norm(p - P) for P in pos])
    others = [u for u in range(3) if u != v]
    n = np.cross(U[others[0]], U[others[1]])
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return dict(z_share=np.nan, cos_los=np.nan, cond=np.nan, gram=np.nan)
    n = n / nn
    J2 = U[others]                                  # (2,3) the two kept views
    s = np.linalg.svd(J2, compute_uv=False)
    return dict(
        z_share=float(n[2] ** 2),                   # what the standoff prior can pin
        cos_los=float(abs(n @ U[v])),               # the naive predictor from doc 13
        cond=float(s[0] / (s[1] + 1e-12)),          # conditioning of the kept pair
        gram=float(abs(U[others[0]] @ U[others[1]])),  # how parallel the kept pair is
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="logs/xview_geom.npz")
    args = ap.parse_args()

    X = np.load(os.path.join(CACHE, "X_clutterremoved.npy"), mmap_mode="r")
    y = np.load(os.path.join(CACHE, "y_clutterremoved.npy"))
    subj = np.load(os.path.join(CACHE, "subj_clutterremoved.npy"))

    d = np.load("logs/w3_width_gap.npz")
    tmpl = (d["t_wide"], M_BINS // 2)

    rng = np.random.default_rng(0)
    picks = []
    for g in np.unique(y):
        if g == 11:                       # G12 = empty gesture, outlier class
            continue
        for s in np.unique(subj):
            idx = np.where((y == g) & (subj == s))[0]
            if len(idx):
                picks.extend(rng.choice(idx, min(args.reps, len(idx)), replace=False).tolist())
    picks = np.array(sorted(picks))
    print(f"[pick] {len(picks)} samples (G12 excluded)\n")

    B = bspline_basis(90)
    rows = []
    t0 = time.time()
    for n_i, i in enumerate(picks):
        Xi = np.asarray(X[i], dtype=np.float64)
        r = np.empty((3, 90)); conf = np.empty((3, 90))
        for v in range(3):
            r[v], conf[v] = viterbi_track(Xi[v])
        fit_all = fit_trajectory(r, conf, views=(0, 1, 2), B=B)
        p_bar = fit_all["p"].mean(axis=0)             # hand's mean position, 3-view fit
        span = float(np.linalg.norm(fit_all["p"].std(axis=0)))

        for v in range(3):
            others = [u for u in range(3) if u != v]
            fit = fit_trajectory(r, conf, views=others, B=B)
            err = float(np.median(np.abs(reproject(fit["C"], B, v) - r[v]) * 1000.0))
            pred = clutter_filter(render(fit["p"], template=tmpl, gains=GAINS))[v]
            c = env_corr(pred, Xi[v])
            g = geometry(p_bar, v)
            rows.append((int(y[i]), int(subj[i]), v, p_bar[0], p_bar[1], p_bar[2],
                         span, g["z_share"], g["cos_los"], g["cond"], g["gram"], err, c))
        if (n_i + 1) % 40 == 0:
            print(f"  {n_i+1}/{len(picks)}  ({time.time()-t0:.0f}s)")

    A = np.array(rows)
    cols = ["y", "subj", "v", "px", "py", "pz", "span",
            "z_share", "cos_los", "cond", "gram", "err_mm", "corr"]
    C = {c: A[:, k] for k, c in enumerate(cols)}
    print(f"\n[n] {len(A)} held-out-view instances\n")

    # ---- how much geometric diversity is actually in this dataset? ----------
    print("=" * 72)
    print("X3a  do the three FIXED radars really give only three configurations?")
    print("=" * 72)
    print(f"  hand mean position spread (m): "
          f"x {C['px'].std():.3f}  y {C['py'].std():.3f}  z {C['pz'].std():.3f}")
    for name in ("z_share", "cos_los", "cond", "gram"):
        q = np.nanpercentile(C[name], [5, 50, 95])
        print(f"  {name:>8s}: 5%/50%/95% = {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}"
              f"   range {np.nanmax(C[name])-np.nanmin(C[name]):.3f}")

    # ---- does geometry predict imaginability, WITHIN a fixed dropped view? ---
    print("\n" + "=" * 72)
    print("X3b  Spearman(geometry, measured error) -- computed WITHIN each dropped")
    print("     view, so the three-view contrast cannot drive the correlation")
    print("=" * 72)
    print(f"{'dropped':>8s} {'n':>5s} {'rho(z_share,err)':>18s} {'rho(cos_los,err)':>18s} "
          f"{'rho(cond,err)':>15s} {'rho(z_share,corr)':>19s}")
    for v in range(3):
        m = C["v"] == v
        def rho(a, b):
            r_, p_ = spearmanr(a[m], b[m])
            return f"{r_:+.3f}{'*' if p_ < 0.01 else ' '}"
        print(f"{VIEWS[v]:>8s} {int(m.sum()):5d} {rho(C['z_share'], C['err_mm']):>18s} "
              f"{rho(C['cos_los'], C['err_mm']):>18s} {rho(C['cond'], C['err_mm']):>15s} "
              f"{rho(C['z_share'], C['corr']):>19s}")

    print("\n  pooled over all instances (three-view contrast INCLUDED):")
    for name in ("z_share", "cos_los", "cond", "gram", "span"):
        r_e, p_e = spearmanr(C[name], C["err_mm"])
        r_c, p_c = spearmanr(C[name], C["corr"])
        print(f"    {name:>8s}  vs err  rho={r_e:+.3f} (p={p_e:.1e})"
              f"   vs corr  rho={r_c:+.3f} (p={p_c:.1e})")

    os.makedirs("logs", exist_ok=True)
    np.savez(args.out, A=A, cols=np.array(cols))
    print(f"\nsaved -> {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
