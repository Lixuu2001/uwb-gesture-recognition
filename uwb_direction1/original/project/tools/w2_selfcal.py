"""W2: sensor self-calibration.

The nominal geometry comes from two sentences in the dataset paper (left-right 1.1 m,
midpoint-to-top 0.55 m); heights and standoff were never published.  The closed loop
showed the price of getting it wrong: a 5 cm error in one radar's position costs
~22 mm of trajectory error.  So estimate the geometry from the recordings themselves.

Gauge.  Ranges fix the scale, and the sensor plane defines z=0, so only three
parameters are free once the Left radar and the x-axis are pinned:
    Right = (rx, 0, 0),  Top = (tx, ty, 0),  Left = (-0.55, 0, 0)
An optional shared range offset b absorbs a constant delay (cable / window start).

Objective: the three-view fit residual, with each recording's trajectory as nuisance
parameters (alternating minimisation).  Validated first on synthetic data where the
true geometry is known by construction.
"""
import argparse
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, ".")
from uwb_osr.trajectory import (viterbi_track, bspline_basis, fit_trajectory,  # noqa: E402
                                RADAR_POS, Z_REF)
from uwb_osr.forward import render, clutter_filter, load_template, calibrate_gains_noise  # noqa: E402

CACHE = r"D:\uwb_gestures\cache"
K_CTRL, LAM, MU = 30, 0.05, 0.05


def geometry(theta):
    rx, tx, ty = theta[:3]
    return np.array([[-0.55, 0.0, 0.0], [tx, ty, 0.0], [rx, 0.0, 0.0]])


def cost(theta, tracks, B, with_offset=False):
    pos = geometry(theta)
    b = theta[3] if with_offset else 0.0
    tot = []
    for r, c in tracks:
        out = fit_trajectory(r - b, c, (0, 1, 2), B=B, lam=LAM, mu=MU, pos=pos)
        tot.append(out["rms"])
    return float(np.mean(tot))


def tracks_from(frames):
    tr = [viterbi_track(frames[v]) for v in range(3)]
    return np.stack([t[0] for t in tr]), np.stack([t[1] for t in tr])


def calibrate(tracks, B, theta0, with_offset=False, verbose=True):
    f = lambda th: cost(th, tracks, B, with_offset)                    # noqa: E731
    # Nelder-Mead stops prematurely on this objective (verified: it left a 4 cm error
    # on a synthetic case that Powell recovers to 1.2 cm), so use Powell.
    res = minimize(f, theta0, method="Powell",
                   options=dict(xtol=1e-4, ftol=1e-7, maxiter=4000))
    if verbose:
        print(f"    start cost {f(theta0)*1000:6.2f} mm -> final {res.fun*1000:6.2f} mm "
              f"({res.nit} iters)")
    return res.x, res.fun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=48)
    args = ap.parse_args()
    B = bspline_basis(90, K_CTRL)
    nominal = np.array([0.55, 0.0, 0.55])          # rx, tx, ty  (= RADAR_POS)

    # ---------------------------------------------------------- 1. synthetic check
    print("=== 1. recovery check on synthetic data (true geometry known) ===")
    tmpl = load_template()
    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    subj = np.load(f"{CACHE}/subj_clutterremoved.npy")
    gains, sigmas = calibrate_gains_noise(X, y, subj)
    g_render = gains / gains.max()
    noise = float(np.median(sigmas / gains))

    rng = np.random.default_rng(0)
    true_theta = np.array([0.60, -0.04, 0.50])     # displaced from nominal by 5, 4, 5 cm
    pos_true = geometry(true_theta)
    syn = []
    for k in range(16):
        t = np.linspace(0, 1, 90)
        a, b_, c_ = rng.uniform(0.12, 0.30, 3)
        ph = rng.uniform(0, 2 * np.pi, 2)
        p = np.stack([a * np.sin(2 * np.pi * 2 * t + ph[0]),
                      b_ * np.sin(2 * np.pi * 2 * t + ph[1]),
                      Z_REF + 0.03 * np.sin(2 * np.pi * t)], axis=1)
        cr = clutter_filter(render(p, template=tmpl, pos=pos_true,
                                   gains=g_render, noise=noise, rng=rng))
        syn.append(tracks_from(cr))
    print(f"  true theta      = {np.round(true_theta, 3)}")
    print(f"  nominal theta   = {np.round(nominal, 3)}   "
          f"(error {np.abs(true_theta-nominal).max()*100:.1f} cm)")
    est, _ = calibrate(syn, B, nominal.copy())
    print(f"  estimated theta = {np.round(est, 3)}   "
          f"residual error {np.abs(est-true_theta).max()*100:.2f} cm")

    # ---------------------------------------------------------- 2. real data
    print("\n=== 2. calibration on real recordings ===")
    rng = np.random.default_rng(1)
    pick = []
    for g in range(12):
        for s in range(1, 9):
            ids = np.where((y == g) & (subj == s))[0]
            if len(ids) and g != 11:
                pick.append(rng.choice(ids))
    pick = rng.choice(pick, min(args.n_real, len(pick)), replace=False)
    real = [tracks_from(np.asarray(X[i])) for i in pick]
    print(f"  {len(real)} recordings")
    base = cost(nominal, real, B)
    print(f"  nominal geometry residual : {base*1000:.2f} mm")
    est_r, fr = calibrate(real, B, nominal.copy())
    print(f"  calibrated theta = {np.round(est_r, 3)}  (nominal {np.round(nominal,3)})")
    print(f"  residual {base*1000:.2f} -> {fr*1000:.2f} mm  "
          f"({100*(base-fr)/base:+.1f}%)")

    est_o, fo = calibrate(real, B, np.append(nominal, 0.0), with_offset=True)
    print(f"  + shared range offset b = {est_o[3]*1000:+.1f} mm -> {fo*1000:.2f} mm  "
          f"({100*(base-fo)/base:+.1f}% vs nominal)")

    # ---------------------------------------------------------- 3. held-out check
    print("\n=== 3. does the calibrated geometry generalise? (held-out recordings) ===")
    pick2 = []
    for g in range(11):
        ids = np.where((y == g) & (subj == 3))[0]
        pick2 += list(ids[:4])
    hold = [tracks_from(np.asarray(X[i])) for i in pick2[:40]]
    c0, c1 = cost(nominal, hold, B), cost(est_r, hold, B)
    print(f"  held-out residual: nominal {c0*1000:.2f} mm -> calibrated {c1*1000:.2f} mm "
          f"({100*(c0-c1)/c0:+.1f}%)")
    np.savez("logs/w2_selfcal.npz", theta=est_r, theta_offset=est_o,
             base=base, final=fr, holdout=(c0, c1))
    print("\nsaved -> logs/w2_selfcal.npz")


if __name__ == "__main__":
    main()
