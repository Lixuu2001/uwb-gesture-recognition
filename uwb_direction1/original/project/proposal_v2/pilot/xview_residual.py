"""Pilot X3 (v2) - is the physics-render residual at a held-out view LEARNABLE?

v1 finding (negative + one overlooked baseline):
  - LOSO ridge on raw envelopes collapses (corr 0.117 < copy 0.149 < phys
    0.304): the cross-view mapping is deeply nonlinear (execution timing warp,
    multipath, extended target).
  - The label-conditioned CLASS-MEAN prior alone reaches 0.525 - HIGHER than
    the instance-specific physics render (0.304).  Doc-24's ladder lacked this
    arm; any "imagine an unseen view" evaluation must control for it.

v2 asks three sharper questions:
  Q1  does per-instance normalisation + label conditioning rescue a LINEAR
      model?  (if yes, cheap; if no, nonlinearity is real)
  Q2  does a LOCAL nonlinear model (kNN over normalised observed-view
      envelopes) work?  (if partially, the mapping is locally smooth)
  Q3  are the class prior and the physics render COMPLEMENTARY?  (best fixed
      blend vs each alone -> informs generator conditioning design)

Arms (all evaluated at the same 30x23 ds-envelope resolution, LOSO):
  classmean / copy / phys / blend / ridge_n / ridge_n+lbl / knn5 / oracle

Pass 1 (feature extraction, ~40 s) is cached to logs/xview_residual_feat.npz.

Usage:  python proposal_v2/pilot/xview_residual.py [--reps 5] [--fresh]
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

T_POOL, R_POOL = 3, 8
R_KEEP = 184
DS = (30, R_KEEP // R_POOL)          # (30, 23) -> 690 dims


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
    num = (P * T).sum(axis=1)
    den = np.linalg.norm(P, axis=1) * np.linalg.norm(T, axis=1) + 1e-30
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    if os.path.exists(FEAT_DUMP) and not args.fresh:
        z = np.load(FEAT_DUMP)
        F, T, REND, ORAC, COPY, M = (z[k] for k in
                                     ("F", "T", "REND", "ORAC", "COPY", "M"))
        print(f"[cache] loaded {F.shape[0]} instances from {FEAT_DUMP}")
    else:
        X = np.load(os.path.join(CACHE, "X_clutterremoved.npy"), mmap_mode="r")
        y = np.load(os.path.join(CACHE, "y_clutterremoved.npy"))
        subj = np.load(os.path.join(CACHE, "subj_clutterremoved.npy"))

        tmpl_narrow = load_template()
        d = np.load("logs/w3_width_gap.npz")
        tmpl_wide = (d["t_wide"], 189 // 2)
        GAINS = (0.876, 1.043, 1.081)

        rng = np.random.default_rng(0)        # same picks as xview_pilot
        picks = []
        for g in np.unique(y):
            for s in np.unique(subj):
                idx = np.where((y == g) & (subj == s))[0]
                if len(idx):
                    picks.extend(rng.choice(idx, min(args.reps, len(idx)),
                                            replace=False).tolist())
        picks = np.array(sorted(picks))
        print(f"[pick] {len(picks)} samples -> {len(picks)*3} view-instances")

        B = bspline_basis(90)
        t0 = time.time()
        F, T, REND, ORAC, COPY, M = [], [], [], [], [], []
        for n, i in enumerate(picks):
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
                e_obs = [analytic_env(Xi[u]) for u in others]
                e_ren, e_real = analytic_env(ren_v), analytic_env(Xi[v])
                e_orac = analytic_env(ren_all[v])
                p = fit["p"]
                sc = ([np.mean(p[:, k]) for k in range(3)] +
                      [np.std(p[:, k]) for k in range(3)] +
                      [np.ptp(p[:, k]) for k in range(3)] +
                      [e_obs[0].max(), e_obs[1].max(), e_ren.max()] +
                      [1.0 if v == k else 0.0 for k in range(3)])
                F.append(np.concatenate([ds(e_obs[0]).ravel(),
                                         ds(e_obs[1]).ravel(),
                                         ds(e_ren).ravel(), sc]))
                T.append(ds(e_real).ravel())
                REND.append(ds(e_ren).ravel())
                ORAC.append(ds(e_orac).ravel())
                COPY.append(ds(e_obs[0]).ravel())
                M.append((int(y[i]), int(subj[i]), v))
            if (n + 1) % 80 == 0:
                print(f"  feat {n+1}/{len(picks)}  ({time.time()-t0:.0f}s)")
        F, T = np.array(F), np.array(T)
        REND, ORAC, COPY = np.array(REND), np.array(ORAC), np.array(COPY)
        M = np.array(M)
        np.savez(FEAT_DUMP, F=F, T=T, REND=REND, ORAC=ORAC, COPY=COPY, M=M)
        print(f"[feat] {F.shape} cached -> {FEAT_DUMP}  ({time.time()-t0:.0f}s)")

    n_inst = len(M)
    g_all, s_all, v_all = M[:, 0], M[:, 1], M[:, 2]
    subjects, gestures = np.unique(s_all), np.unique(g_all)
    D = DS[0] * DS[1]

    # ---------- per-instance normalised envelope blocks -------------------
    def norm_rows(A):
        return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)

    obsA, obsB, rend = F[:, :D], F[:, D:2 * D], F[:, 2 * D:3 * D]
    scal = F[:, 3 * D:]
    Tn = norm_rows(T)
    obsAn, obsBn, rendn = norm_rows(obsA), norm_rows(obsB), norm_rows(rend)

    onehot = np.zeros((n_inst, len(gestures)))
    onehot[np.arange(n_inst), g_all - 1] = 1.0
    mu_s, sd_s = scal.mean(0), scal.std(0) + 1e-9
    scaln = (scal - mu_s) / sd_s

    Fn = np.hstack([obsAn, obsBn, rendn, scaln])
    Fnl = np.hstack([Fn, onehot])
    # kNN feature space: observed views only (what the generator is conditioned
    # on besides the render) - use obsA/obsB concatenated
    Kn = np.hstack([obsAn, obsBn])

    LAMBDAS = [1e-1, 1e0, 1e1, 1e2, 1e3]
    KS = [1, 3, 5, 10]

    def ridge_fit(Xtr, Ytr, lam):
        d = Xtr.shape[1]
        return np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ Ytr)

    preds = {"ridge_n": np.zeros_like(Tn), "ridge_n_lbl": np.zeros_like(Tn),
             "knn": np.zeros_like(Tn), "classmean": np.zeros_like(Tn),
             "blend": np.zeros_like(Tn)}
    knn_k = {s: 5 for s in subjects}
    blend_a = {s: 0.5 for s in subjects}

    for s in subjects:
        te, tr = s_all == s, s_all != s
        # class mean prior (training subjects only, per gesture+view)
        cm = np.zeros_like(Tn)
        for g in gestures:
            for v in range(3):
                mv = tr & (g_all == g) & (v_all == v)
                if mv.sum():
                    cm[te & (g_all == g) & (v_all == v)] = Tn[mv].mean(axis=0)
        preds["classmean"][te] = cm[te]

        # blend alpha on training: classmean_tr needed for all training rows
        cm_tr = np.zeros_like(Tn)
        for g in gestures:
            for v in range(3):
                mv = tr & (g_all == g) & (v_all == v)
                if mv.sum():
                    cm_tr[mv] = Tn[mv].mean(axis=0)
        # leave-one-out-ish: subtract own contribution crudely by using it
        # anyway (n_per_cell ~ 30, self-leak is mild for choosing alpha)
        best_a, best_c = 0.5, -1
        for a in np.linspace(0, 1, 21):
            pred = a * cm_tr + (1 - a) * norm_rows(REND)
            c = corr_rows(pred, Tn).mean()
            if c > best_c:
                best_c, best_a = c, a
        blend_a[s] = best_a
        preds["blend"][te] = (best_a * cm[te] +
                              (1 - best_a) * norm_rows(REND)[te])

        # ridge: pick lambda by inner LOSO on training subjects
        best = {}
        for name, Xf in (("ridge_n", Fn), ("ridge_n_lbl", Fnl)):
            errs = {l: 0.0 for l in LAMBDAS}
            for s2 in subjects[subjects != s]:
                tr2, te2 = tr & (s_all != s2), tr & (s_all == s2)
                if not te2.sum():
                    continue
                for l in LAMBDAS:
                    W = ridge_fit(Xf[tr2], Tn[tr2], l)
                    errs[l] += (np.mean((Xf[te2] @ W - Tn[te2]) ** 2)
                                * te2.sum())
            best[name] = min(errs, key=errs.get)
            preds[name][te] = Xf[te] @ ridge_fit(Xf[tr], Tn[tr], best[name])

        # kNN: rows are unit-norm so squared distance = 2 - 2*dot
        def knn_pred(Kq, Kt, Tt, k):
            sim = Kq @ Kt.T                       # (n_q, n_tr)
            idx = np.argpartition(-sim, k - 1)[:, :k]
            return np.array([Tt[ix].mean(axis=0) for ix in idx])

        errs = {k: 0.0 for k in KS}
        for s2 in subjects[subjects != s]:
            tr2, te2 = tr & (s_all != s2), tr & (s_all == s2)
            if not te2.sum():
                continue
            for k in KS:
                pred = knn_pred(Kn[te2], Kn[tr2], Tn[tr2], k)
                errs[k] += corr_rows(pred, Tn[te2]).mean() * te2.sum()
        knn_k[s] = max(errs, key=errs.get)
        k = knn_k[s]
        preds["knn"][te] = knn_pred(Kn[te], Kn[tr], Tn[tr], k)
        print(f"  fold s{s}: lam={best['ridge_n']:g}/{best['ridge_n_lbl']:g} "
              f"k={k} alpha={best_a:.2f}")

    # ---------- report -----------------------------------------------------
    arms = {"classmean": preds["classmean"], "copy": norm_rows(COPY),
            "phys": norm_rows(REND), "blend": preds["blend"],
            "ridge_n": preds["ridge_n"], "ridge_n_lbl": preds["ridge_n_lbl"],
            "knn": preds["knn"], "oracle": norm_rows(ORAC)}
    res = {k: corr_rows(v, Tn) for k, v in arms.items()}

    print("\n" + "=" * 80)
    print("X3v2  learnability of the held-out view (LOSO, ds-envelope corr, "
          f"n={n_inst})")
    print("=" * 80)
    print(f"{'arm':>13s} {'env corr':>16s} {'per-view L/T/R':>20s}")
    for k, c in res.items():
        pv = " ".join(f"{c[v_all == v].mean():.3f}" for v in range(3))
        print(f"{k:>13s} {c.mean():7.3f} +- {c.std():5.3f}  {pv:>20s}")

    for a, b in (("blend", "classmean"), ("blend", "phys"),
                 ("ridge_n", "phys"), ("ridge_n_lbl", "phys"),
                 ("knn", "phys"), ("knn", "classmean"), ("knn", "blend")):
        d_ = res[a] - res[b]
        print(f"  paired {a:>11s} - {b:<11s}: {d_.mean():+.3f}  "
              f"win {np.mean(d_ > 0) * 100:5.1f}%")

    np.savez("logs/xview_residual.npz", meta=M,
             **{f"corr_{k}": v for k, v in res.items()})
    print("\nsaved -> logs/xview_residual.npz")


if __name__ == "__main__":
    main()
