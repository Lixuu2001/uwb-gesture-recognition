"""W2 follow-up: does an EXTENDED-TARGET model close the sim-to-real residual gap?

`proposal_v2/05_W2结果` §5 left the forward model with one unexplained fact:

    three-view fit residual   synthetic (single scatterer) =  8.7 mm
                              real data                    = 35-54 mm

and named the prime suspect: the hand is not a point.  The measured echo is 159 mm
wide at -6 dB while the renderer places one effective scatterer.  This script tests
that suspicion by rendering N scattering centres spread over an extent E and asking
whether ONE configuration reproduces BOTH independent observables (echo width AND
fit residual).  Matching two observables with one knob is a real test; matching one
is curve fitting.

Part B tests the other mechanism named in `proposal_v2/07` §2: that the recorded
chirality failure (RESULTS 49.5, d'=0.10) is Doppler-phase ALIASING rather than
geometry.  With ground truth available we can sweep the hand speed across the
lambda/4 wrap threshold and watch the Doppler readout die while the range readout
survives.

Usage: python tools/w2_extended.py
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from uwb_osr.trajectory import (viterbi_track, bspline_basis, fit_trajectory,  # noqa: E402
                                BIN_M, RADAR_POS, Z_REF, FPS)
from uwb_osr.forward import (render, clutter_filter, load_template,           # noqa: E402
                             calibrate_gains_noise)
from uwb_osr.data import doppler_map                                          # noqa: E402

CACHE = r"D:\uwb_gestures\cache"
K_CTRL, LAM, MU = 30, 0.05, 0.05
LAMBDA_MM = 41.1                      # X4 low band, see 05_W2 §2
WRAP_MM = LAMBDA_MM / 4               # 10.28 mm per frame


def analytic(x):
    n = x.shape[-1]
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(np.fft.fft(x, axis=-1) * h, axis=-1)


def echo_width_mm(frames, db=-6.0):
    """Width of the range profile at `db` below the peak, on the strongest frame."""
    e = np.abs(analytic(frames.astype(np.float64)))
    t = (e ** 2).sum(axis=1).argmax()
    prof = e[t] / e[t].max()
    thr = 10 ** (db / 20)
    return float((prof >= thr).sum() * BIN_M * 1000)


def reconstruct(frames, B, pos=RADAR_POS):
    tr = [viterbi_track(frames[v]) for v in range(3)]
    r = np.stack([t[0] for t in tr])
    c = np.stack([t[1] for t in tr])
    return fit_trajectory(r, c, (0, 1, 2), B=B, lam=LAM, mu=MU, pos=pos)


def circle(kind, T=90, amp=0.25, strokes=2.0):
    t = np.linspace(0, 1, T)
    sgn = -1.0 if kind == "cw" else 1.0
    return np.stack([amp * np.cos(2 * np.pi * strokes * t),
                     sgn * amp * np.sin(2 * np.pi * strokes * t),
                     Z_REF + 0.03 * np.sin(2 * np.pi * t)], axis=1)


def hand_cloud(n, extent_m, seed=0):
    """n scattering centres spread over `extent_m` (rms), one pinned at the centre."""
    if n <= 1:
        return None
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, extent_m / 2, (n, 3))
    pts[0] = 0.0
    return pts


def dprime(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float((a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-12))


def signed_area(x, y):
    x, y = x - np.mean(x), y - np.mean(y)
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def doppler_chirality(frames):
    """Velocity-plane loop area from the Doppler centroids (the RESULTS 49.5 route)."""
    d = doppler_map(frames)                       # (V,T,M) signed radial velocity map
    w = np.abs(frames) ** 2
    v = (d * w).sum(axis=2) / (w.sum(axis=2) + 1e-12)      # (V,T) per-view centroid
    v_h = (v[2] - v[0]) / 2.0
    v_v = v[1]
    return signed_area(v_h, v_v)


def main():
    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    subj = np.load(f"{CACHE}/subj_clutterremoved.npy")
    B = bspline_basis(90, K_CTRL)
    tmpl = load_template()
    gains, sigmas = calibrate_gains_noise(X, y, subj)
    g_render = gains / gains.max()
    noise = float(np.median(sigmas / gains))

    # reference numbers measured on REAL data with THIS code
    rng = np.random.default_rng(0)
    idx = rng.choice(np.where(y < 11)[0], 60, replace=False)
    real_w, real_res = [], []
    for i in idx:
        unit = np.asarray(X[i])
        real_w.append(np.mean([echo_width_mm(unit[v]) for v in range(3)]))
        real_res.append(reconstruct(unit, B)["rms"] * 1000)
    print("=" * 76)
    print("REFERENCE (real data, measured by this script)")
    print("=" * 76)
    print(f"  echo width @-6dB : {np.median(real_w):7.1f} mm   (05_W2 reported 159)")
    print(f"  3-view fit rms   : {np.median(real_res):7.1f} mm   (05_W2 reported 35-54)")

    # ---------------------------------------------------------------- A
    print("\n" + "=" * 76)
    print("A. extended target sweep: can ONE config match BOTH observables?")
    print("=" * 76)
    print(f"{'N':>3s}{'extent':>9s}{'width_mm':>10s}{'fit_rms':>9s}"
          f"{'traj_err':>10s}{'centre_bias':>12s}")
    p_true = circle("ccw")
    for n in (1, 3, 5, 8):
        for ext in (0.0, 0.05, 0.10, 0.15, 0.20):
            if n == 1 and ext > 0:
                continue
            ws, rs, es, bs = [], [], [], []
            for seed in range(3):
                sc = hand_cloud(n, ext, seed=seed)
                raw = render(p_true, template=tmpl, gains=g_render, noise=noise,
                             scatterers=sc, rng=np.random.default_rng(seed))
                cr = clutter_filter(raw)
                ws.append(np.mean([echo_width_mm(cr[v]) for v in range(3)]))
                out = reconstruct(cr, B)
                rs.append(out["rms"] * 1000)
                es.append(np.median(np.linalg.norm(out["p"] - p_true, axis=1)) * 1000)
                bs.append(np.linalg.norm((out["p"] - p_true).mean(axis=0)) * 1000)
            print(f"{n:>3d}{ext*100:8.0f}cm{np.mean(ws):10.1f}{np.mean(rs):9.1f}"
                  f"{np.mean(es):10.1f}{np.mean(bs):12.1f}")
    print(f"\n  target: width ~ {np.median(real_w):.0f} mm AND fit_rms ~ {np.median(real_res):.0f} mm")

    # ---------------------------------------------------------------- B
    print("\n" + "=" * 76)
    print("B. aliasing: Doppler vs range chirality readout across the wrap threshold")
    print("=" * 76)
    print(f"  wrap threshold = lambda/4 = {WRAP_MM:.2f} mm of radial motion per frame")
    print(f"{'amp_cm':>8s}{'strokes':>9s}{'max|dr|':>10s}{'%wrapped':>10s}"
          f"{'dop d(prime)':>14s}{'range d(prime)':>15s}")
    for amp, st in ((0.05, 1.0), (0.10, 1.0), (0.15, 2.0), (0.25, 2.0), (0.35, 3.0)):
        dop, rng_, wraps, mx = {}, {}, [], []
        for kind in ("cw", "ccw"):
            dv, rv = [], []
            for seed in range(4):
                p = circle(kind, amp=amp, strokes=st)
                raw = render(p, template=tmpl, gains=g_render, noise=noise,
                             rng=np.random.default_rng(seed))
                cr = clutter_filter(raw)
                dv.append(doppler_chirality(cr))
                out = reconstruct(cr, B)
                rv.append(signed_area(out["p"][:, 0], out["p"][:, 1]))
                if kind == "cw" and seed == 0:
                    d = np.abs(np.diff(np.linalg.norm(p[:, None, :] - RADAR_POS[None], axis=2),
                                       axis=0)) * 1000
                    wraps.append(np.mean(d > WRAP_MM) * 100)
                    mx.append(d.max())
            dop[kind], rng_[kind] = dv, rv
        print(f"{amp*100:8.0f}{st:9.1f}{np.mean(mx):10.1f}{np.mean(wraps):10.1f}"
              f"{dprime(dop['ccw'], dop['cw']):14.2f}{dprime(rng_['ccw'], rng_['cw']):15.2f}")
    print("\n  RECORDED (RESULTS 49.5, real data): Doppler loop area d'=0.10;")
    print("  W1 range-domain loop area on real data: d'=2.77.")


if __name__ == "__main__":
    main()
