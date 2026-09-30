"""Phase 0: does the physics renderer explain diagonal (G5-G8) echoes it never saw?

The renderer is a DETERMINISTIC function of the trajectory, so it has no notion of
"seen" vs "unseen" gestures.  Any "zero-shot generation of unseen combined motion"
claim must therefore separate what the physics skeleton gives for free from what a
learned residual would have to add.  Three arms:

  A  G1-G4, rendered from their OWN fitted trajectory      (seen, oracle trajectory)
  B  G5-G8, rendered from their OWN fitted trajectory      (unseen class, oracle traj)
  C  G5-G8, rendered from trajectories COMPOSED out of G1-G4 single-axis profiles
     C-native : source amplitudes kept          -> pure zero-shot, no G5-G8 info at all
     C-scaled : rescaled to the real G5-G8 median sd_x/sd_y (two scalars per class)
                -> isolates scattering/shape realism from the amplitude question

A vs B  : are diagonals intrinsically harder for the renderer?
B vs C  : what does COMPOSING (rather than fitting) the trajectory cost?

Primary metric is a classifier two-sample test (C2ST) AUC between real and synthetic
range profiles on physically interpretable features; 0.5 = indistinguishable.
Per-feature effect sizes say WHICH feature betrays the synthetic data, i.e. exactly
what a residual model would have to fix.

Usage:  python tools/w3_diag_b0.py [--reps 10]
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from config import GESTURE_NAMES                                       # noqa: E402
from uwb_osr.trajectory import (viterbi_track, bspline_basis, fit_trajectory,  # noqa: E402
                                _analytic, BIN_M)
from uwb_osr.forward import (load_template, render_cr, calibrate_gains_noise,  # noqa: E402
                             AMP_EXP)

CACHE = r"D:\uwb_gestures\cache"
K_CTRL, LAM, MU = 30, 0.05, 0.05
VIEWS = ["Left", "Top", "Right"]

# G5..G8 = two-axis combinations of the single-axis swipes G1..G4 (config.py line 90).
# 0-based gesture ids.  (horizontal source, vertical source)
DIAG_SRC = {4: (0, 2),      # G5 LR-UD  <- G1 left->right , G3 up->down
            5: (0, 3),      # G6 LR-DU  <- G1             , G4 down->up
            6: (1, 2),      # G7 RL-UD  <- G2 right->left , G3
            7: (1, 3)}      # G8 RL-DU  <- G2             , G4

FEATS = ["peakbin_sd", "width6", "n_peaks", "amp_iqr",
         "rel_gain", "far_energy", "path_jitter", "spec_cent"]


# --------------------------------------------------------------------------- feats
def unit_features(unit):
    """(3,T,M) clutter-removed RF -> physically interpretable feature vector.

    Amplitude is normalised by the unit's global peak, so absolute gain drops out
    but per-view relative gain and noise-to-peak ratio are preserved.
    """
    env = np.abs(_analytic(unit.astype(np.float64)))
    gmax = env.max() + 1e-30
    env = env / gmax
    out = []
    view_med = [np.median(env[v].max(axis=1)) for v in range(3)]
    gm = np.median(view_med) + 1e-30
    for v in range(3):
        e = env[v]                                        # (T,M)
        pk = e.argmax(axis=1)
        peak = e.max(axis=1)
        # -6 dB width: contiguous run above half the frame peak, around the peak
        w6 = np.empty(len(pk))
        npk = np.empty(len(pk))
        for t, m in enumerate(pk):
            thr = 0.5 * peak[t]
            lo = m
            while lo > 0 and e[t, lo - 1] > thr:
                lo -= 1
            hi = m
            while hi < e.shape[1] - 1 and e[t, hi + 1] > thr:
                hi += 1
            w6[t] = hi - lo + 1
            row = e[t]
            loc = (row[1:-1] > row[:-2]) & (row[1:-1] > row[2:]) & (row[1:-1] > 0.3 * peak[t])
            npk[t] = loc.sum()
        far = env[v][:, 170:].mean() / (peak.mean() + 1e-30)
        sp = np.abs(np.fft.rfft(pk - pk.mean()))
        freq = np.fft.rfftfreq(len(pk))
        cent = float((sp * freq).sum() / (sp.sum() + 1e-30))
        out += [pk.std(),
                np.median(w6),
                np.median(npk),
                (np.percentile(peak, 75) - np.percentile(peak, 25)) / (np.median(peak) + 1e-30),
                view_med[v] / gm,
                far,
                np.median(np.abs(np.diff(pk))),
                cent]
    return np.array(out, dtype=np.float64)


def c2st(Fr, Fs, seed=0):
    """Classifier two-sample test AUC (logistic regression, 5-fold CV)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score
    X = np.vstack([Fr, Fs])
    y = np.r_[np.zeros(len(Fr)), np.ones(len(Fs))]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    sc = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        sc[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, sc))


def effect_sizes(Fr, Fs):
    """Standardised mean difference per feature (synthetic minus real)."""
    m = (Fs.mean(0) - Fr.mean(0))
    s = np.sqrt(0.5 * (Fs.var(0) + Fr.var(0))) + 1e-30
    return m / s


def mahal(Fr, Fs, ridge=1e-3):
    """Mahalanobis distance between the two feature-mean vectors, in real-data units.

    C2ST AUC saturates at 1.0 as soon as the two sets are separable, which destroys
    the A/B/C comparison.  This has unbounded headroom.
    """
    Fr = np.nan_to_num(Fr); Fs = np.nan_to_num(Fs)
    sd = Fr.std(0) + 1e-9
    A, Bm = Fr / sd, Fs / sd
    S = np.cov(A.T) + ridge * np.eye(A.shape[1])
    d = Bm.mean(0) - A.mean(0)
    return float(np.sqrt(max(d @ np.linalg.solve(S, d), 0.0)))


def line_cloud(n, extent, direction):
    """n rigid offsets spread over `extent` metres along a unit `direction`."""
    u = np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    t = np.linspace(-0.5, 0.5, n)[:, None] * extent
    return t * u[None, :]


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="logs/w3_diag_b0.npz")
    args = ap.parse_args()

    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    subj = np.load(f"{CACHE}/subj_clutterremoved.npy")
    B = bspline_basis(90, K_CTRL)
    tmpl = load_template()

    gains, noise_sd = calibrate_gains_noise(X, y, subj, n=40)
    g_rel = gains / gains.mean()
    # peak render amplitude at the population standoff, to put `noise` in the same units
    peak_amp = 0.72 ** AMP_EXP
    noise_lvl = float(np.mean(noise_sd / gains) * peak_amp)
    print(f"per-view relative gains : {np.round(g_rel, 3)}")
    print(f"noise/peak measured     : {np.round(noise_sd / gains, 4)}  "
          f"-> render noise={noise_lvl:.4f}", flush=True)

    t0 = time.time()
    fits, reals = {}, {}
    for g in range(8):
        fits[g], reals[g] = [], []
        for s in range(1, 9):
            for i in np.where((y == g) & (subj == s))[0][:args.reps]:
                unit = np.asarray(X[i])
                tr = [viterbi_track(unit[v]) for v in range(3)]
                r = np.stack([t[0] for t in tr])
                c = np.stack([t[1] for t in tr])
                f = fit_trajectory(r, c, (0, 1, 2), B=B, lam=LAM, mu=MU)
                fits[g].append(dict(s=s, p=f["p"], rms=f["rms"]))
                reals[g].append(unit_features(unit))
        reals[g] = np.array(reals[g])
        print(f"  G{g+1} fitted+featurised ({len(fits[g])} samples, {time.time()-t0:.0f}s)",
              flush=True)

    def render_feats(paths):
        F = []
        for p in paths:
            F.append(unit_features(render_cr(p, template=tmpl, gains=g_rel,
                                             noise=noise_lvl)))
        return np.array(F)

    # ---------------------------------------------------------------- arms A and B
    synth = {}
    for g in range(8):
        synth[g] = render_feats([d["p"] for d in fits[g]])
        print(f"  G{g+1} arm A/B rendered ({time.time()-t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------- arm C
    rng = np.random.default_rng(0)
    comp = {}
    for g, (gx, gy) in DIAG_SRC.items():
        tgt_sdx = np.median([d["p"][:, 0].std() for d in fits[g]])
        tgt_sdy = np.median([d["p"][:, 1].std() for d in fits[g]])
        native, scaled = [], []
        for d in fits[g]:
            s = d["s"]
            cx = [q for q in fits[gx] if q["s"] == s] or fits[gx]
            cy = [q for q in fits[gy] if q["s"] == s] or fits[gy]
            px = cx[rng.integers(len(cx))]["p"]
            py = cy[rng.integers(len(cy))]["p"]
            xs = px[:, 0] - px[:, 0].mean()
            ys = py[:, 1] - py[:, 1].mean()
            zs = 0.5 * (px[:, 2] + py[:, 2])
            native.append(np.c_[xs, ys, zs])
            scaled.append(np.c_[xs / (xs.std() + 1e-12) * tgt_sdx,
                                ys / (ys.std() + 1e-12) * tgt_sdy, zs])
        comp[g] = dict(native=render_feats(native), scaled=render_feats(scaled),
                       sdx=tgt_sdx, sdy=tgt_sdy,
                       src_sdx=np.median([q["p"][:, 0].std() for q in fits[gx]]),
                       src_sdy=np.median([q["p"][:, 1].std() for q in fits[gy]]))
        print(f"  G{g+1} arm C rendered ({time.time()-t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------- report
    print("\n=== A. amplitude: do people do diagonals at single-axis amplitude? ===")
    print(f"{'target':10s}{'real sd_x':>11s}{'src sd_x':>11s}{'real sd_y':>11s}{'src sd_y':>11s}")
    for g in sorted(DIAG_SRC):
        d = comp[g]
        print(f"{'G'+str(g+1):10s}{d['sdx']*1000:11.1f}{d['src_sdx']*1000:11.1f}"
              f"{d['sdy']*1000:11.1f}{d['src_sdy']*1000:11.1f}")

    print("\n=== B. C2ST AUC, real vs physics-rendered (0.5 = indistinguishable) ===")
    rows = {}
    print(f"{'gesture':10s}{'arm A/B':>10s}{'C-native':>10s}{'C-scaled':>10s}")
    for g in range(8):
        a = c2st(reals[g], synth[g])
        if g in comp:
            cn = c2st(reals[g], comp[g]["native"])
            cs = c2st(reals[g], comp[g]["scaled"])
            print(f"{'G'+str(g+1):10s}{a:10.3f}{cn:10.3f}{cs:10.3f}")
            rows[g] = (a, cn, cs)
        else:
            print(f"{'G'+str(g+1):10s}{a:10.3f}{'-':>10s}{'-':>10s}")
            rows[g] = (a, np.nan, np.nan)
    ab_single = np.mean([rows[g][0] for g in range(4)])
    ab_diag = np.mean([rows[g][0] for g in range(4, 8)])
    cn_diag = np.nanmean([rows[g][1] for g in range(4, 8)])
    cs_diag = np.nanmean([rows[g][2] for g in range(4, 8)])
    print(f"\n  arm A  (G1-G4, own traj)      : {ab_single:.3f}")
    print(f"  arm B  (G5-G8, own traj)      : {ab_diag:.3f}   <- A vs B: are diagonals harder?")
    print(f"  arm C-native (G5-G8, composed): {cn_diag:.3f}")
    print(f"  arm C-scaled (G5-G8, composed): {cs_diag:.3f}   <- B vs C: cost of composing")

    # ------------------------------------------- arm A': best physical extension
    print("\n=== B2. best hand-tuned PHYSICAL extension (arm A') ===")
    print("  09 already falsified isotropic clouds as a residual explanation "
          "(20 cm extent -> only 128 mm); here we ask how much of the C2ST gap "
          "the best physical extension closes, i.e. what is LEFT for a learned residual.")
    wi_all = [FEATS.index("width6") + 8 * v for v in range(3)]
    probe = [d["p"] for d in fits[0][:24]] + [d["p"] for d in fits[4][:24]]
    Fr_probe = np.vstack([reals[0][:24], reals[4][:24]])
    real_w = Fr_probe[:, wi_all].mean()
    best = None
    print(f"{'direction':>12s}{'n':>4s}{'extent(mm)':>12s}{'width(mm)':>11s}"
          f"{'C2ST':>8s}{'mahal':>9s}")
    for name, direction in [("depth z", (0, 0, 1)), ("down -y", (0, -1, 0)),
                            ("iso 3D", None)]:
        for extent in (0.10, 0.20, 0.30):
            n = 5
            if direction is None:
                sc = np.array([[0, 0, 0], [extent / 2, 0, 0], [-extent / 2, 0, 0],
                               [0, extent / 2, 0], [0, -extent / 2, 0],
                               [0, 0, extent / 2], [0, 0, -extent / 2]])
            else:
                sc = line_cloud(n, extent, direction)
            F = np.array([unit_features(render_cr(p, template=tmpl, gains=g_rel,
                                                  noise=noise_lvl, scatterers=sc))
                          for p in probe])
            w_mm = F[:, wi_all].mean() * BIN_M * 1000
            a = c2st(Fr_probe, F)
            mh = mahal(Fr_probe, F)
            print(f"{name:>12s}{len(sc):4d}{extent*1000:12.0f}{w_mm:11.1f}{a:8.3f}{mh:9.2f}")
            if best is None or mh < best[0]:
                best = (mh, name, extent, w_mm, a, sc)
    print(f"  real -6 dB width on the same probe set: {real_w*BIN_M*1000:.1f} mm")
    print(f"  best physical extension: {best[1]} extent={best[2]*1000:.0f} mm  "
          f"-> width {best[3]:.1f} mm, C2ST {best[4]:.3f}, mahal {best[5-1] if False else best[0]:.2f}")

    print("\n=== B3. headroom metric (mahalanobis, unbounded) per arm ===")
    mh_a = np.mean([mahal(reals[g], synth[g]) for g in range(4)])
    mh_b = np.mean([mahal(reals[g], synth[g]) for g in range(4, 8)])
    mh_cn = np.mean([mahal(reals[g], comp[g]["native"]) for g in sorted(DIAG_SRC)])
    mh_cs = np.mean([mahal(reals[g], comp[g]["scaled"]) for g in sorted(DIAG_SRC)])
    print(f"  arm A  G1-G4 own traj      : {mh_a:7.2f}")
    print(f"  arm B  G5-G8 own traj      : {mh_b:7.2f}")
    print(f"  arm C-native G5-G8 composed: {mh_cn:7.2f}")
    print(f"  arm C-scaled G5-G8 composed: {mh_cs:7.2f}")
    print(f"  -> cost of COMPOSING (C-scaled - B) : {mh_cs - mh_b:+7.2f}")
    print(f"  -> cost of WRONG AMPLITUDE (Cn - Cs): {mh_cn - mh_cs:+7.2f}")

    print("\n=== B4. C2ST with the saturating width features removed ===")
    keep = np.array([k for k in range(24) if k not in wi_all])
    print(f"  arm A  : {np.mean([c2st(reals[g][:, keep], synth[g][:, keep]) for g in range(4)]):.3f}")
    print(f"  arm B  : {np.mean([c2st(reals[g][:, keep], synth[g][:, keep]) for g in range(4,8)]):.3f}")
    print(f"  C-scaled: {np.mean([c2st(reals[g][:, keep], comp[g]['scaled'][:, keep]) for g in sorted(DIAG_SRC)]):.3f}")

    print("\n=== C. which features betray the synthetic data (std mean diff, pooled) ===")
    Fr = np.vstack([reals[g] for g in range(8)])
    Fs = np.vstack([synth[g] for g in range(8)])
    d = effect_sizes(Fr, Fs)
    order = np.argsort(-np.abs(d))
    names = [f"{f}[{VIEWS[v]}]" for v in range(3) for f in FEATS]
    for k in order[:10]:
        print(f"  {names[k]:24s} d = {d[k]:+7.2f}   real {Fr[:,k].mean():9.3f}  "
              f"synth {Fs[:,k].mean():9.3f}")

    print("\n=== D. the headline physical gap: -6 dB echo width ===")
    wi = [FEATS.index("width6") + 8 * v for v in range(3)]
    for v in range(3):
        print(f"  {VIEWS[v]:>5s}: real {Fr[:,wi[v]].mean():5.2f} bins "
              f"= {Fr[:,wi[v]].mean()*BIN_M*1000:6.1f} mm   |   "
              f"synth {Fs[:,wi[v]].mean():5.2f} bins "
              f"= {Fs[:,wi[v]].mean()*BIN_M*1000:6.1f} mm")

    np.savez(args.out,
             auc=np.array([[rows[g][0], rows[g][1], rows[g][2]] for g in range(8)]),
             eff=d, feat_names=np.array(names),
             real_feats=Fr, synth_feats=Fs,
             sd_table=np.array([[comp[g]["sdx"], comp[g]["src_sdx"],
                                 comp[g]["sdy"], comp[g]["src_sdy"]]
                                for g in sorted(DIAG_SRC)]))
    print(f"\nsaved -> {args.out}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
