"""Pilot X4 - does cross-view imagination transfer to an UNSEEN GESTURE?

X3v2 showed that on stereotyped closed-set gestures the label-conditioned
class-mean prior (0.596) dominates every learned arm (ridge 0.60, kNN 0.56)
and even the known-trajectory oracle only reaches 0.624.  In other words, on
THIS dataset most of the predictable signal at an unobserved view is just the
gesture template.

The natural follow-up, and the one that connects to the project's open-set
identity: hold out an ENTIRE GESTURE.  No class prior exists for it; the only
priors left are physics and cross-view reasoning.  This is the regime where
"imagining an unseen view" has to earn its keep.

LOGO (leave-one-gesture-out), label-free arms only, ds-envelope corr:
  gmean     global mean envelope (gesture-blind floor)
  copy      first observed view
  phys      width-matched physics render from the 2-view fit
  ridge_n   ridge trained on the OTHER 11 gestures (all subjects)
  knn5      5-NN over observed-view envelopes, other 11 gestures
  oracle    physics render from the all-three-view fit

Plus a full-resolution peak-timing addendum (classmean vs phys vs oracle vs
copy) computed by re-extracting envelopes, to show where instance-specific
information actually lives.

Usage: python proposal_v2/pilot/xview_logo.py [--peak]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from uwb_osr.forward import clutter_filter, load_template, render
from uwb_osr.trajectory import bspline_basis, fit_trajectory, viterbi_track

CACHE = r"D:\uwb_gestures\cache"
FEAT_DUMP = "logs/xview_residual_feat.npz"
T_POOL, R_POOL, R_KEEP = 3, 8, 184
D = 30 * (R_KEEP // R_POOL)


def analytic_env(x):
    n = x.shape[-1]
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(np.fft.fft(x, axis=-1) * h, axis=-1))


def ds(env):
    e = env[:90, :R_KEEP].reshape(90, R_KEEP // R_POOL, R_POOL).mean(axis=2)
    return e.reshape(30, T_POOL, R_KEEP // R_POOL).mean(axis=1)


def corr_rows(P, T):
    P = P - P.mean(axis=1, keepdims=True)
    T = T - T.mean(axis=1, keepdims=True)
    return ((P * T).sum(1) /
            (np.linalg.norm(P, axis=1) * np.linalg.norm(T, axis=1) + 1e-30))


def norm_rows(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak", action="store_true",
                    help="also recompute full-res peak-timing errors")
    args = ap.parse_args()

    z = np.load(FEAT_DUMP)
    F, T, REND, ORAC, COPY, M = (z[k] for k in ("F", "T", "REND", "ORAC",
                                                "COPY", "M"))
    n_inst = len(M)
    g_all, s_all, v_all = M[:, 0], M[:, 1], M[:, 2]
    gestures = np.unique(g_all)
    print(f"[cache] {n_inst} instances, {len(gestures)} gestures")

    obsA, obsB, rend = F[:, :D], F[:, D:2 * D], F[:, 2 * D:3 * D]
    scal = F[:, 3 * D:]
    Tn = norm_rows(T)
    obsAn, obsBn = norm_rows(obsA), norm_rows(obsB)
    scaln = (scal - scal.mean(0)) / (scal.std(0) + 1e-9)
    Fn = np.hstack([obsAn, obsBn, norm_rows(rend), scaln])
    Kn = np.hstack([obsAn, obsBn])

    LAMBDAS = [1e-1, 1e0, 1e1, 1e2, 1e3]

    def ridge_fit(Xtr, Ytr, lam):
        d = Xtr.shape[1]
        return np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ Ytr)

    def knn_pred(Kq, Kt, Tt, k=5):
        sim = Kq @ Kt.T
        idx = np.argpartition(-sim, k - 1)[:, :k]
        return np.array([Tt[ix].mean(axis=0) for ix in idx])

    preds = {"ridge_n": np.zeros_like(Tn), "knn5": np.zeros_like(Tn)}
    t0 = time.time()
    for g in gestures:
        te, tr = g_all == g, g_all != g
        # inner lambda pick: leave-one-gesture-out within the 11 training ones
        errs = {l: 0.0 for l in LAMBDAS}
        for g2 in gestures[gestures != g]:
            tr2, te2 = tr & (g_all != g2), tr & (g_all == g2)
            if not te2.sum():
                continue
            for l in LAMBDAS:
                W = ridge_fit(Fn[tr2], Tn[tr2], l)
                errs[l] += np.mean((Fn[te2] @ W - Tn[te2]) ** 2) * te2.sum()
        lam = min(errs, key=errs.get)
        preds["ridge_n"][te] = Fn[te] @ ridge_fit(Fn[tr], Tn[tr], lam)
        preds["knn5"][te] = knn_pred(Kn[te], Kn[tr], Tn[tr], 5)
        print(f"  LOGO gesture {int(g):2d}: lam={lam:g}  "
              f"({time.time()-t0:.0f}s)")

    gmean = np.tile(Tn[tr].mean(axis=0), (n_inst, 1))   # gesture-blind floor
    arms = {"gmean": gmean, "copy": norm_rows(COPY),
            "phys": norm_rows(REND), "ridge_n": preds["ridge_n"],
            "knn5": preds["knn5"], "oracle": norm_rows(ORAC)}
    res = {k: corr_rows(v, Tn) for k, v in arms.items()}

    print("\n" + "=" * 80)
    print("X4  LOGO: imagining views of a NEVER-SEEN gesture "
          f"(ds-envelope corr, n={n_inst})")
    print("=" * 80)
    print(f"{'arm':>9s} {'env corr':>16s} {'per-view L/T/R':>20s}")
    for k, c in res.items():
        pv = " ".join(f"{c[v_all == v].mean():.3f}" for v in range(3))
        print(f"{k:>9s} {c.mean():7.3f} +- {c.std():5.3f}  {pv:>20s}")
    for a, b in (("ridge_n", "phys"), ("ridge_n", "gmean"),
                 ("knn5", "phys"), ("ridge_n", "copy")):
        d_ = res[a] - res[b]
        print(f"  paired {a:>7s} - {b:<7s}: {d_.mean():+.3f}  "
              f"win {np.mean(d_ > 0) * 100:5.1f}%")

    np.savez("logs/xview_logo.npz", meta=M,
             **{f"corr_{k}": v for k, v in res.items()})
    print("saved -> logs/xview_logo.npz")

    # ---------------- full-res peak-timing addendum -------------------------
    if args.peak:
        print("\n[peak] re-extracting full-res envelopes ...")
        X = np.load(os.path.join(CACHE, "X_clutterremoved.npy"),
                    mmap_mode="r")
        y = np.load(os.path.join(CACHE, "y_clutterremoved.npy"))
        subj = np.load(os.path.join(CACHE, "subj_clutterremoved.npy"))
        rng = np.random.default_rng(0)
        picks = []
        for g in np.unique(y):
            for s in np.unique(subj):
                idx = np.where((y == g) & (subj == s))[0]
                if len(idx):
                    picks.extend(rng.choice(idx, min(5, len(idx)),
                                            replace=False).tolist())
        picks = np.array(sorted(picks))
        tmpl_narrow = load_template()
        d = np.load("logs/w3_width_gap.npz")
        tmpl_wide = (d["t_wide"], 189 // 2)
        GAINS = (0.876, 1.043, 1.081)
        B = bspline_basis(90)
        BIN_MM = 6.4

        real_f, phys_f, orac_f, copy_f = [], [], [], []
        meta2 = []
        for i in picks:
            Xi = np.asarray(X[i], dtype=np.float64)
            r = np.empty((3, 90)); conf = np.empty((3, 90))
            for v in range(3):
                r[v], conf[v] = viterbi_track(Xi[v])
            fit_all = fit_trajectory(r, conf, views=(0, 1, 2), B=B)
            ren_all = clutter_filter(render(fit_all["p"], template=tmpl_wide,
                                            gains=GAINS))
            for v in range(3):
                others = [u for u in range(3) if u != v]
                fit = fit_trajectory(r, conf, views=others, B=B)
                ren_v = clutter_filter(render(fit["p"], template=tmpl_wide,
                                              gains=GAINS))[v]
                real_f.append(analytic_env(Xi[v]))
                phys_f.append(analytic_env(ren_v))
                orac_f.append(analytic_env(ren_all[v]))
                copy_f.append(analytic_env(Xi[others[0]]))
                meta2.append((int(y[i]), int(subj[i]), v))
        real_f = np.array(real_f); phys_f = np.array(phys_f)
        orac_f = np.array(orac_f); copy_f = np.array(copy_f)
        meta2 = np.array(meta2)
        m_g, m_s = meta2[:, 0], meta2[:, 1]

        # class-mean full-res, leave-one-instance-out (no self-leak)
        cm_f = np.zeros_like(real_f)
        for g in np.unique(m_g):
            for v in range(3):
                mv = np.where((m_g == g) & (meta2[:, 2] == v))[0]
                if len(mv) < 2:
                    continue
                s = real_f[mv].sum(axis=0)
                for i in mv:
                    cm_f[i] = (s - real_f[i]) / (len(mv) - 1)

        def peak_err(pred):
            pk = np.abs(pred.argmax(axis=2) - real_f.argmax(axis=2))
            return np.median(pk) * BIN_MM

        print("\n  full-res per-frame peak-timing error (median mm):")
        for name, pred in (("classmean", cm_f), ("copy", copy_f),
                           ("phys", phys_f), ("oracle", orac_f)):
            print(f"    {name:>10s}: {peak_err(pred):7.1f}")
        np.savez("logs/xview_peak.npz", meta=meta2,
                 peak_err=np.array([peak_err(cm_f), peak_err(copy_f),
                                    peak_err(phys_f), peak_err(orac_f)]))


if __name__ == "__main__":
    main()
