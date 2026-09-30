"""W2 follow-up 2: which unmodelled physics closes the REST of the sim-to-real gap?

`proposal_v2/08` left a specific target.  A rigid "hand + 30 cm forearm" reproduces
the measured echo width (148 vs 154 mm) but only half the three-view fit residual
(23.7 vs 49.7 mm).  Three candidate mechanisms are tested here, each switchable, so
the contribution of each can be read off rather than tuned:

  ARM   articulated forearm - the elbow stays put while the hand moves, so the
        forearm ROTATES and its scattering centres move non-rigidly relative to the
        hand (the rigid offset used in `08` cannot do this)
  RCS   time-varying scattering amplitude (Swerling-like fluctuation with a
        correlation time), instead of the static weights used so far
  BODY  a torso reflector that is strong but nearly static, so most of it is removed
        by the loopback clutter filter and only its sway survives

Target to hit:  echo width 154.5 mm  AND  three-view fit residual 49.7 mm,
without wrecking the trajectory error beyond what `08` already reports.

Usage: python tools/w2_body.py
"""
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from uwb_osr.trajectory import bspline_basis, Z_REF                      # noqa: E402
from uwb_osr.forward import (render, clutter_filter, load_template,      # noqa: E402
                             calibrate_gains_noise)
from w2_extended import echo_width_mm, reconstruct, circle               # noqa: E402

CACHE = r"D:\uwb_gestures\cache"
ELBOW = np.array([0.05, -0.28, 0.72])     # roughly: behind and below the hand


def arm_points(traj, n=6, elbow=ELBOW, taper=0.6):
    """Per-frame scattering centres along the forearm elbow -> hand.

    Returns (N, T, 3) absolute positions and (N,) weights.  The hand itself is the
    first centre; the rest walk back towards a FIXED elbow, so the geometry changes
    with the hand position - that is the articulation the rigid model lacked.
    """
    f = np.linspace(0.0, 1.0, n)          # 0 = hand, 1 = elbow
    pts = traj[None, :, :] * (1 - f)[:, None, None] + elbow[None, None, :] * f[:, None, None]
    w = taper ** (f * (n - 1))
    return pts, w / w.sum()


def rcs_series(n, T, rho=0.85, sigma_db=4.0, rng=None):
    """Correlated log-normal amplitude fluctuation, one series per scatterer."""
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.zeros((n, T))
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    return 10 ** (sigma_db * x / 20.0)


def render_cloud(points, weights, tmpl, gains, noise, amp_t=None, rng=None):
    """Sum per-scatterer renders; `points` is (N,T,3), `amp_t` optional (N,T)."""
    rng = np.random.default_rng(0) if rng is None else rng
    out = None
    for i in range(points.shape[0]):
        f = render(points[i], template=tmpl, gains=gains, noise=0.0)
        if amp_t is not None:
            f = f * amp_t[i][None, :, None]
        f = f * weights[i]
        out = f if out is None else out + f
    out = out + rng.normal(0.0, noise, out.shape)
    return out


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

    print(f"{'model':34s}{'width':>8s}{'fit_rms':>9s}{'traj':>8s}"
          f"{'bias':>7s}{'shape':>8s}")

    def report(name, points, weights, amp_t=None, body=False):
        ws, rs, es, bs = [], [], [], []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            pts, wts = points, weights
            at = amp_t
            if body:
                # torso: strong reflector, nearly static (breathing / sway ~1 cm)
                sway = np.zeros((1, T, 3))
                sway[0, :, 2] = 0.95 + 0.010 * np.sin(2 * np.pi * 0.3 * np.arange(T) / 20.0)
                sway[0, :, 1] = -0.10
                pts = np.concatenate([pts, sway], axis=0)
                wts = np.concatenate([wts, [3.0 * wts.sum()]])
                if at is not None:
                    at = np.concatenate([at, np.ones((1, T))], axis=0)
            raw = render_cloud(pts, wts, tmpl, g, noise, amp_t=at, rng=rng)
            cr = clutter_filter(raw)
            ws.append(np.mean([echo_width_mm(cr[v]) for v in range(3)]))
            out = reconstruct(cr, B)
            rs.append(out["rms"] * 1000)
            d = out["p"] - p_true
            es.append(np.median(np.linalg.norm(d, axis=1)) * 1000)
            bs.append(np.linalg.norm(d.mean(axis=0)) * 1000)
        e, b = np.mean(es), np.mean(bs)
        print(f"{name:34s}{np.mean(ws):8.1f}{np.mean(rs):9.1f}{e:8.1f}{b:7.1f}"
              f"{max(e ** 2 - b ** 2, 0) ** 0.5:8.1f}")

    # baselines
    hand = (p_true[None, :, :], np.array([1.0]))
    report("hand only (point)", *hand)

    ap, aw = arm_points(p_true)
    report("ARM articulated", ap, aw)

    rng = np.random.default_rng(7)
    report("ARM + RCS fluctuation", ap, aw, amp_t=rcs_series(len(aw), T, rng=rng))
    report("ARM + BODY", ap, aw, body=True)
    rng = np.random.default_rng(7)
    report("ARM + RCS + BODY", ap, aw,
           amp_t=rcs_series(len(aw), T, rng=rng), body=True)

    # how strong does the fluctuation have to be on its own?
    for s_db in (2.0, 6.0, 10.0):
        rng = np.random.default_rng(7)
        report(f"ARM + RCS sigma={s_db:.0f}dB", ap, aw,
               amp_t=rcs_series(len(aw), T, sigma_db=s_db, rng=rng))

    # arm length sensitivity under articulation
    for elb_z in (0.60, 0.72, 0.85):
        e = ELBOW.copy(); e[2] = elb_z
        ap2, aw2 = arm_points(p_true, elbow=e)
        report(f"ARM articulated, elbow z={elb_z:.2f}", ap2, aw2)

    print(f"\n  TARGET (real data): width 154.5 mm, fit_rms 49.7 mm")


if __name__ == "__main__":
    main()
