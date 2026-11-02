"""Phase 0b: the forward model's dominant realism gap is the ECHO WIDTH, and it is a
template-calibration gap, not a gesture-dependent one.

Facts established by tools/w3_diag_b0.py and logs/w2_pulse.npz:
  * real -6 dB echo width          ~157 mm  (docs: 154.5 mm; npz width_bins=25 -> 158.8)
  * the CALIBRATED TEMPLATE's own width is 13 bins = 82.5 mm, i.e. half of that,
    even though the stored window W=24 (49 bins = 311 mm) does not truncate it
  * adding rigid multi-scatterers NARROWS the rendered echo (69 -> 45 mm) because
    `render` sums COHERENTLY and 50 mm spacing at lambda=41.1 mm is ~1.2 lambda,
    so the sum speckles instead of broadening

So this script builds a width-matched template by broadening the ENVELOPE while
keeping the measured carrier phase (an "effective extended-target" template), and asks
the only question that matters for planning:

    once the known width defect is fixed, HOW MUCH OF THE REALISM GAP IS LEFT
    for a learned residual model to close?

Usage:  python tools/w3_width_gap.py [--reps 6]
"""
import argparse
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, ".")
from uwb_osr.trajectory import (viterbi_track, bspline_basis, fit_trajectory,   # noqa: E402
                                _analytic, BIN_M, M_BINS)
from uwb_osr.forward import (load_template, render_cr, calibrate_gains_noise,   # noqa: E402
                             AMP_EXP, C_LIGHT)
from tools.w3_diag_b0 import unit_features, c2st, effect_sizes, FEATS, VIEWS    # noqa: E402

CACHE = r"D:\uwb_gestures\cache"
K_CTRL, LAM, MU = 30, 0.05, 0.05


def env_width6(sig):
    """-6 dB (half-amplitude) width in bins of a 1-D real RF signal."""
    e = np.abs(_analytic(np.asarray(sig, float)[None, :]))[0]
    m = int(e.argmax())
    thr = 0.5 * e[m]
    lo = m
    while lo > 0 and e[lo - 1] > thr:
        lo -= 1
    hi = m
    while hi < len(e) - 1 and e[hi + 1] > thr:
        hi += 1
    return hi - lo + 1


def widen_template(full, centre, target_bins, carrier_hz):
    """Broaden the template envelope to `target_bins` -6 dB width, keep the carrier.

    An extended target lowers the EFFECTIVE bandwidth seen at the receiver, which
    widens the envelope.  Re-modulating a smoothed envelope with the measured carrier
    phase avoids the speckle that a coherent multi-scatterer sum produces.
    """
    a = _analytic(full[None, :])[0]
    env = np.abs(a)
    fcs = carrier_hz * 2.0 * BIN_M / C_LIGHT              # cycles per range sample
    n = np.arange(len(full))
    phi0 = float(np.angle(a[centre]) - 2 * np.pi * fcs * centre)

    lo, hi = 0.1, 40.0
    for _ in range(60):
        s = 0.5 * (lo + hi)
        e2 = gaussian_filter1d(env, s, mode="constant")
        w = env_width6((e2 * np.cos(2 * np.pi * fcs * n + phi0)))
        if w < target_bins:
            lo = s
        else:
            hi = s
    s = 0.5 * (lo + hi)
    e2 = gaussian_filter1d(env, s, mode="constant")
    t2 = e2 * np.cos(2 * np.pi * fcs * n + phi0)
    t2 = t2 / np.abs(t2).max()
    return t2, s, fcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6)
    args = ap.parse_args()

    d = np.load("logs/w2_pulse.npz")
    carrier = float(d["carrier"])
    target_bins = float(d["width_bins"])
    full, centre = load_template()

    print(f"stored carrier          : {carrier/1e9:.4f} GHz")
    print(f"stored width_bins       : {target_bins:.1f} bins "
          f"= {target_bins*BIN_M*1000:.1f} mm   (this is the REAL echo width)")
    print(f"template's own -6 dB    : {env_width6(full)} bins "
          f"= {env_width6(full)*BIN_M*1000:.1f} mm   <-- the defect")

    t_wide, sigma, fcs = widen_template(full, centre, target_bins, carrier)
    print(f"carrier as cycles/sample: {fcs:.4f}  (docs say 0.3203)")
    print(f"widened template        : sigma={sigma:.2f} bins -> "
          f"{env_width6(t_wide)} bins = {env_width6(t_wide)*BIN_M*1000:.1f} mm")

    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    subj = np.load(f"{CACHE}/subj_clutterremoved.npy")
    B = bspline_basis(90, K_CTRL)
    gains, noise_sd = calibrate_gains_noise(X, y, subj, n=40)
    g_rel = gains / gains.mean()
    noise_lvl = float(np.mean(noise_sd / gains) * 0.72 ** AMP_EXP)

    paths, Fr = [], []
    for g in (0, 2, 4, 6):                     # one horizontal, one vertical, two diagonals
        for s in range(1, 9):
            for i in np.where((y == g) & (subj == s))[0][:args.reps]:
                unit = np.asarray(X[i])
                tr = [viterbi_track(unit[v]) for v in range(3)]
                r = np.stack([t[0] for t in tr])
                c = np.stack([t[1] for t in tr])
                paths.append(fit_trajectory(r, c, (0, 1, 2), B=B, lam=LAM, mu=MU)["p"])
                Fr.append(unit_features(unit))
    Fr = np.array(Fr)
    print(f"\nprobe set: {len(paths)} samples from G1/G3/G5/G7", flush=True)

    def arm(tmpl):
        F = np.array([unit_features(render_cr(p, template=tmpl, gains=g_rel,
                                              noise=noise_lvl)) for p in paths])
        e = effect_sizes(Fr, F)
        return F, c2st(Fr, F), float(np.linalg.norm(e)), e

    names = [f"{f}[{VIEWS[v]}]" for v in range(3) for f in FEATS]
    print(f"\n{'arm':>26s}{'C2ST':>8s}{'||effect||2':>13s}{'max|d|':>9s}"
          f"{'width(mm)':>11s}")
    res = {}
    wi = [FEATS.index("width6") + 8 * v for v in range(3)]
    for tag, tmpl in [("B0  original template", (full, centre)),
                      ("B0w width-matched tmpl", (t_wide, centre))]:
        F, a, nrm, e = arm(tmpl)
        res[tag] = (F, a, nrm, e)
        print(f"{tag:>26s}{a:8.3f}{nrm:13.1f}{np.abs(e).max():9.2f}"
              f"{F[:,wi].mean()*BIN_M*1000:11.1f}")
    print(f"{'REAL':>26s}{'-':>8s}{'-':>13s}{'-':>9s}"
          f"{Fr[:,wi].mean()*BIN_M*1000:11.1f}")

    e0 = res["B0  original template"][3]
    e1 = res["B0w width-matched tmpl"][3]
    print(f"\ngap closed by fixing the template only: "
          f"||effect||2  {np.linalg.norm(e0):.1f} -> {np.linalg.norm(e1):.1f}  "
          f"({100*(1-np.linalg.norm(e1)/np.linalg.norm(e0)):.0f}% reduction)")

    print("\n=== what is LEFT for a learned residual (top offenders after the fix) ===")
    for k in np.argsort(-np.abs(e1))[:8]:
        print(f"  {names[k]:24s} d = {e1[k]:+7.2f}   real {Fr[:,k].mean():9.3f}  "
              f"synth {res['B0w width-matched tmpl'][0][:,k].mean():9.3f}")

    np.savez("logs/w3_width_gap.npz", t_wide=t_wide, sigma=sigma, fcs=fcs,
             eff_b0=e0, eff_b0w=e1, feat_names=np.array(names),
             auc_b0=res["B0  original template"][1],
             auc_b0w=res["B0w width-matched tmpl"][1])
    print("\nsaved -> logs/w3_width_gap.npz")


if __name__ == "__main__":
    main()
