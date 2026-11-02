"""W2: simulation closed loop -- the only route to an accuracy number in mm.

Take a trajectory, render its three-view echoes with the data-calibrated forward
model, push the synthetic frames through the SAME W1 pipeline, and compare the
recovered trajectory against the one we rendered.  Since the ground truth is known by
construction, this gives the mm-level accuracy that neither leave-one-radar-out (ill
posed, see proposal_v2/04) nor temporal holdout (extrapolation) can give.

Trajectories used:
  * REAL   - reconstructions of real recordings (realistic speeds and shapes)
  * SYNTH  - analytic swipes and circles (clean, no reconstruction bias)

Also reports how the error scales with the additive noise level and whether the
per-frame geometry is recoverable when the radar positions are perturbed.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from config import GESTURE_NAMES                                       # noqa: E402
from uwb_osr.trajectory import (viterbi_track, bspline_basis, fit_trajectory,  # noqa: E402
                                RADAR_POS, Z_REF)
from uwb_osr.forward import (render, clutter_filter, load_template,    # noqa: E402
                             calibrate_gains_noise)

CACHE = r"D:\uwb_gestures\cache"
K_CTRL, LAM, MU = 30, 0.05, 0.05


def reconstruct(frames, B, pos=RADAR_POS, mu=MU):
    tr = [viterbi_track(frames[v]) for v in range(3)]
    r = np.stack([t[0] for t in tr])
    c = np.stack([t[1] for t in tr])
    return fit_trajectory(r, c, (0, 1, 2), B=B, lam=LAM, mu=mu, pos=pos)


def err_mm(a, b):
    return np.linalg.norm(a - b, axis=1) * 1000


def synth_traj(kind, T=90, rng=None):
    t = np.linspace(0, 1, T)
    z = Z_REF + 0.03 * np.sin(2 * np.pi * t)
    if kind == "swipe_h":
        x = 0.35 * np.sin(2 * np.pi * 2 * t)
        yy = 0.05 * np.sin(2 * np.pi * t)
    elif kind == "swipe_v":
        x = 0.05 * np.sin(2 * np.pi * t)
        yy = 0.30 * np.sin(2 * np.pi * 2 * t)
    elif kind == "circle_cw":
        x, yy = 0.25 * np.cos(2 * np.pi * 2 * t), -0.25 * np.sin(2 * np.pi * 2 * t)
    elif kind == "circle_ccw":
        x, yy = 0.25 * np.cos(2 * np.pi * 2 * t), 0.25 * np.sin(2 * np.pi * 2 * t)
    elif kind == "diag":
        x, yy = 0.25 * np.sin(2 * np.pi * 2 * t), 0.25 * np.sin(2 * np.pi * 2 * t)
    else:
        raise ValueError(kind)
    return np.stack([x, yy, z], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()

    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    subj = np.load(f"{CACHE}/subj_clutterremoved.npy")
    B = bspline_basis(90, K_CTRL)
    tmpl = load_template()
    gains, sigmas = calibrate_gains_noise(X, y, subj)
    print(f"per-view gain (relative)  : {gains / gains.max()}")
    print(f"per-view noise sigma      : {sigmas}")
    # express the render gain so that the peak echo matches the real per-view peak
    g_render = gains / gains.max()
    noise_lvl = float(np.median(sigmas / gains))
    print(f"noise/peak ratio used     : {noise_lvl:.4f}\n")

    # ---------------------------------------------------------------- 1. synthetic
    print("=== 1. analytic trajectories: rendered -> reconstructed ===")
    print(f"{'trajectory':14s}{'median err':>12s}{'p90':>8s}{'max':>8s}"
          f"{'x':>8s}{'y':>8s}{'z':>8s}")
    for kind in ("swipe_h", "swipe_v", "circle_cw", "circle_ccw", "diag"):
        p = synth_traj(kind)
        raw = render(p, template=tmpl, gains=g_render, noise=noise_lvl)
        out = reconstruct(clutter_filter(raw), B)
        e = err_mm(out["p"], p)
        d = np.abs(out["p"] - p).mean(axis=0) * 1000
        print(f"{kind:14s}{np.median(e):12.1f}{np.percentile(e,90):8.1f}{e.max():8.1f}"
              f"{d[0]:8.1f}{d[1]:8.1f}{d[2]:8.1f}")

    # ---------------------------------------------------------------- 2. noise sweep
    print("\n=== 2. error vs noise level (circle_ccw) ===")
    p = synth_traj("circle_ccw")
    for nl in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1):
        errs = []
        for s in range(3):
            raw = render(p, template=tmpl, gains=g_render, noise=nl,
                         rng=np.random.default_rng(s))
            errs.append(np.median(err_mm(reconstruct(clutter_filter(raw), B)["p"], p)))
        print(f"  noise/peak={nl:5.3f}   median err = {np.mean(errs):7.1f} mm")

    # ---------------------------------------------------------------- 3. real shapes
    print("\n=== 3. real reconstructed trajectories, re-rendered and re-reconstructed ===")
    print(f"{'gesture':22s}{'median err mm':>15s}{'p90':>8s}")
    allerr = []
    for g in (0, 2, 4, 8, 9, 10):
        errs = []
        for s in range(1, 9):
            for i in np.where((y == g) & (subj == s))[0][:args.reps // 4 + 1]:
                p_true = reconstruct(np.asarray(X[i]), B)["p"]
                raw = render(p_true, template=tmpl, gains=g_render, noise=noise_lvl)
                p_hat = reconstruct(clutter_filter(raw), B)["p"]
                errs.append(np.median(err_mm(p_hat, p_true)))
        allerr += errs
        print(f"{'G'+str(g+1)+' '+GESTURE_NAMES.get(g,''):22s}"
              f"{np.median(errs):15.1f}{np.percentile(errs,90):8.1f}")
    print(f"{'ALL':22s}{np.median(allerr):15.1f}{np.percentile(allerr,90):8.1f}")

    # ---------------------------------------------------------------- 4. geometry
    print("\n=== 4. sensitivity to a wrong sensor geometry (render true, solve perturbed) ===")
    p = synth_traj("circle_ccw")
    raw = render(p, template=tmpl, gains=g_render, noise=noise_lvl)
    cr = clutter_filter(raw)
    for dx in (0.0, 0.02, 0.05, 0.10):
        pos = RADAR_POS.copy()
        pos[0, 0] -= dx                       # left radar displaced by dx
        e = err_mm(reconstruct(cr, B, pos=pos)["p"], p)
        print(f"  left radar off by {dx*100:5.1f} cm -> median err {np.median(e):7.1f} mm")


if __name__ == "__main__":
    main()
