"""W2 follow-up 2b: corrected error decomposition + the RCS-taper variable.

Two fixes over `tools/w2_body.py` / `proposal_v2/08`:

  1. The bias/shape split was computed as median(|e_t|) against |mean(e_t)|, which is
     not an identity - the "shape" term clipped to zero on skewed runs.  The valid
     decomposition uses RMS:   RMS^2 = |mean offset|^2 + mean||e_t - mean e||^2 .
  2. `08` reported that the RCS taper along the forearm "was not actually exercised"
     (the implementation collapsed).  It is exercised here, and it turns out to be
     the variable that decides whether the echo width is reproduced at all.

Targets from real data: echo width 154.5 mm, three-view fit residual 49.7 mm.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from uwb_osr.trajectory import bspline_basis                             # noqa: E402
from uwb_osr.forward import (render, clutter_filter, load_template,      # noqa: E402
                             calibrate_gains_noise)
from w2_extended import echo_width_mm, reconstruct, circle               # noqa: E402
from w2_body import arm_points, rcs_series, render_cloud, ELBOW          # noqa: E402

CACHE = r"D:\uwb_gestures\cache"


def decompose(p_hat, p_true):
    """RMS error split into a constant offset and a time-varying part (identity)."""
    e = (p_hat - p_true) * 1000.0
    mu = e.mean(axis=0)
    bias = float(np.linalg.norm(mu))
    var = float(np.sqrt(np.mean(np.sum((e - mu) ** 2, axis=1))))
    rms = float(np.sqrt(np.mean(np.sum(e ** 2, axis=1))))
    return rms, bias, var


def rigid_arm(traj, n=6, length=0.30, taper=1.0):
    """The `08` model: scatterers at FIXED offsets behind the hand."""
    d = np.linspace(0, length, n)
    off = np.stack([np.zeros(n), -0.3 * d, d], axis=1)
    pts = traj[None, :, :] + off[:, None, :]
    w = taper ** np.arange(n)
    return pts, w / w.sum()


def main():
    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    subj = np.load(f"{CACHE}/subj_clutterremoved.npy")
    B = bspline_basis(90, 30)
    tmpl = load_template()
    gains, sig = calibrate_gains_noise(X, y, subj)
    g = gains / gains.max()
    noise = float(np.median(sig / gains))
    p_true = circle("ccw")
    T = p_true.shape[0]

    print(f"{'model':36s}{'width':>8s}{'fit_rms':>9s}"
          f"{'RMS err':>9s}{'bias':>7s}{'time-var':>10s}")

    def report(name, points, weights, amp_t=None, body_w=0.0, seeds=3):
        ws, rs, dec = [], [], []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            pts, wts, at = points, weights, amp_t
            if body_w > 0:
                sway = np.zeros((1, T, 3))
                sway[0, :, 2] = 0.95 + 0.010 * np.sin(2 * np.pi * 0.3 * np.arange(T) / 20.0)
                sway[0, :, 1] = -0.10
                pts = np.concatenate([pts, sway], axis=0)
                wts = np.concatenate([wts, [body_w * wts.sum()]])
                if at is not None:
                    at = np.concatenate([at, np.ones((1, T))], axis=0)
            cr = clutter_filter(render_cloud(pts, wts, tmpl, g, noise, amp_t=at, rng=rng))
            ws.append(np.mean([echo_width_mm(cr[v]) for v in range(3)]))
            out = reconstruct(cr, B)
            rs.append(out["rms"] * 1000)
            dec.append(decompose(out["p"], p_true))
        dec = np.array(dec).mean(axis=0)
        print(f"{name:36s}{np.mean(ws):8.1f}{np.mean(rs):9.1f}"
              f"{dec[0]:9.1f}{dec[1]:7.1f}{dec[2]:10.1f}")

    report("hand only (point target)", p_true[None], np.array([1.0]))

    print("  -- rigid arm (the `08` model), RCS taper swept --")
    for tp in (1.0, 0.8, 0.6):
        report(f"rigid arm 30cm, taper={tp:.1f}", *rigid_arm(p_true, taper=tp))

    print("  -- articulated arm (elbow fixed, forearm rotates) --")
    for tp in (1.0, 0.8, 0.6):
        report(f"articulated arm, taper={tp:.1f}", *arm_points(p_true, taper=tp))

    print("  -- articulated + flat RCS + fluctuation --")
    for s_db in (4.0, 8.0):
        ap, aw = arm_points(p_true, taper=1.0)
        report(f"articulated flat + RCS {s_db:.0f}dB", ap, aw,
               amp_t=rcs_series(len(aw), T, sigma_db=s_db, rng=np.random.default_rng(7)))

    print("  -- articulated + flat RCS + torso of varying strength --")
    for bw in (0.1, 0.3, 1.0):
        ap, aw = arm_points(p_true, taper=1.0)
        report(f"articulated flat + torso x{bw:.1f}", ap, aw, body_w=bw)

    print(f"\n  TARGET (real data): width 154.5 mm, fit_rms 49.7 mm")


if __name__ == "__main__":
    main()
