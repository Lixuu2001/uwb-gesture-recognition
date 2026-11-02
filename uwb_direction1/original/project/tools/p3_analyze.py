"""Post-hoc P3 analysis: e_recon (+) r fusion, simulated-task conformal,
per-view p-value merging -- all from the per-fold npz dumps, no retraining.

Reads the dumps written by `main.py --conformal` (with the P3 extensions:
per-sample e_recon E_*, per-view distances D_*, simulated-task scores
sim_r_cal). Every threshold strategy below is evaluated on the same frozen
score dumps, so a whole lambda sweep costs seconds.

Modes reported (rows):
  global / mondrian / hybrid / eta=3   -- the N2/N3 baselines (SS32/SS33)
  simtask                              -- new-class columns use the conformal
        quantile of the base classes' SIMULATED few-shot heads scored on the
        calib subject (data-free replacement for the falsified shrinkage)
  fuse l=X [glob|sim]                  -- non-conformity = z(r) + lambda*z(e_recon),
        recalibrated on the fused calib scores; class-agnostic e_recon shifts
        all columns equally, so routing (argmin) is untouched and only the
        accept/reject boundary moves -- the axis where 53% of accepted unknowns
        leak through base columns (SS33.4)
  pmerge-glob / pmerge-hyb             -- per-view Fisher merge of calib-ECDF
        p-values (DIAGNOSTIC: the calib set is reused for the ECDF and the
        quantile, so the guarantee is approximate here)

Columns: cov_b/cov_n (coverage), sz (set size), rejU (unknown rejected),
leakB (unknowns accepted through >=1 BASE column), AUROC (min-score known vs
unknown), sAcc_n (new-class singleton accuracy). Plus nearest-prototype
closed accuracies per dump set (the delta-aug comparison number).

Usage:
  python tools/p3_analyze.py "logs/n3_scores_bal_b2e_c0_s*.npz"
  python tools/p3_analyze.py "logs/n3_scores_bal_c0_s*.npz" --alpha 0.1
"""
import argparse
import glob

import numpy as np

LAMBDAS = (0.3, 1.0, 3.0)


def _q(scores, alpha):
    scores = np.asarray(scores, float)
    n = len(scores)
    if n == 0:
        return np.inf
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if rank > n else float(np.sort(scores)[rank - 1])


def _auroc(pos_conf, neg_conf):
    """P(pos > neg) rank statistic; pos = known confidence."""
    x = np.concatenate([pos_conf, neg_conf])
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    ranks = (csum - (cnt - 1) / 2.0)[inv]
    n1, n0 = len(pos_conf), len(neg_conf)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _own(R, ytrue):
    return R[np.arange(len(R)), ytrue]


def _sim_rows(rd):
    """Row indices into the calib set that sim_r_cal was scored on (class-major
    order, mirroring simulate_fewshot_calib's loop)."""
    idx = [np.where(rd["y_cal"] == k)[0] for k in range(rd["n_base"])
           if (rd["y_cal"] == k).any()]
    return np.concatenate(idx) if idx else np.zeros(0, int)


def _metrics(Sb, yb, Sr, yr, Su, q, nb):
    def grp(S, ytrue):
        sets = S <= q[None, :]
        sizes = sets.sum(1)
        if ytrue is None:
            leak = float(sets[:, :nb].any(1).mean()) if len(S) else 0.0
            return dict(rej=float((sizes == 0).mean()) if len(S) else 0.0,
                        size=float(sizes.mean()) if len(S) else 0.0, leak=leak)
        cov = float(_own(sets, ytrue).mean()) if len(S) else 0.0
        sing = sizes == 1
        sa = (float((sets[sing].argmax(1) == ytrue[sing]).mean())
              if sing.sum() else 0.0)
        return dict(cov=cov, size=float(sizes.mean()) if len(S) else 0.0,
                    sing_acc=sa)
    b, r, u = grp(Sb, yb), grp(Sr, yr), grp(Su, None)
    minS = np.concatenate([Sb.min(1), Sr.min(1)])
    au = _auroc(-minS, -Su.min(1))
    return dict(cov_b=b["cov"], cov_n=r["cov"], sz_b=b["size"], sz_n=r["size"],
                rejU=u["rej"], leakB=u["leak"], auroc=au, sAcc_n=r["sing_acc"])


def fold_report(rd, alpha, mode, lam=1.0, eta=3.0, hyb_new=False):
    nb, C = rd["n_base"], rd["n_base"] + rd["n_reg"]
    Rb, Rr, Ru, Rc = rd["R_base"], rd["R_reg"], rd["R_unk"], rd["R_cal"]
    yb, yr, yc = rd["y_base"], rd["y_reg"], rd["y_cal"]
    own_b = _own(Rc, yc)
    qg = _q(own_b, alpha)

    if mode == "eta":
        q = np.full(C, float(eta))
        return _metrics(Rb, yb, Rr, yr, Ru, q, nb)

    if mode in ("global", "mondrian", "hybrid", "simtask"):
        q = np.full(C, qg, float)
        if mode == "mondrian":
            for k in range(nb):
                qk = _q(own_b[yc == k], alpha)
                q[k] = qk if np.isfinite(qk) else qg
        if mode in ("mondrian", "hybrid") and len(rd.get("R_cal_reg", [])):
            Rcr, ycr = rd["R_cal_reg"], rd["y_cal_reg"]
            own_n = _own(Rcr, ycr)
            for c in range(nb, C):
                qc = _q(own_n[ycr == c], alpha)
                q[c] = qc if np.isfinite(qc) else qg
        if mode == "simtask":
            if len(rd.get("sim_r_cal", [])) == 0:
                return None
            q[nb:] = _q(rd["sim_r_cal"], alpha)     # pooled few-shot-head q
        return _metrics(Rb, yb, Rr, yr, Ru, q, nb)

    if mode.startswith("fuse-"):
        if "E_cal" not in rd or len(rd["E_cal"]) == 0:
            return None
        mu_r, sd_r = float(own_b.mean()), float(own_b.std() + 1e-8)
        mu_e, sd_e = (float(rd["E_cal"].mean()),
                      float(rd["E_cal"].std() + 1e-8))
        base_only = mode.startswith("fuse-base")
        hyb_new = hyb_new or mode.endswith("+hyb")

        def fuse(R, E):
            # fuse-base: e_recon penalises acceptance through BASE columns only.
            # Rationale (measured): the base-trained reconstructor puts the
            # registered classes as far off-manifold as true unknowns (median
            # e_recon reg 0.96 > unk 0.90 > base 0.71), so a full-column fusion
            # crushes new-class coverage; but base vs non-base separates, which
            # is exactly the question on a base column -- the 53% leak channel.
            S = (R - mu_r) / sd_r
            ez = lam * ((E - mu_e) / sd_e)[:, None]
            if base_only:
                S[:, :nb] += ez
            else:
                S += ez
            return S
        Sc = fuse(Rc, rd["E_cal"])
        Sb = fuse(Rb, rd["E_base"])
        Sr = fuse(Rr, rd["E_reg"])
        Su = fuse(Ru, rd["E_unk"])
        qf = _q(_own(Sc, yc), alpha)
        q = np.full(C, qf, float)
        if base_only:
            # new columns are un-fused plain scores -> their own plain quantile
            q[nb:] = _q((_own(Rc, yc) - mu_r) / sd_r, alpha)
            if hyb_new and len(rd.get("R_cal_reg", [])):
                # oracle stack: fused base columns + per-class hybrid new q
                Rcr, ycr = rd["R_cal_reg"], rd["y_cal_reg"]
                own_n = (_own(Rcr, ycr) - mu_r) / sd_r
                for c in range(nb, C):
                    qc = _q(own_n[ycr == c], alpha)
                    q[c] = qc if np.isfinite(qc) else q[c]
        if mode == "fuse-sim":
            if len(rd.get("sim_r_cal", [])) == 0:
                return None
            rows = _sim_rows(rd)
            sim_f = ((rd["sim_r_cal"] - mu_r) / sd_r
                     + lam * (rd["E_cal"][rows] - mu_e) / sd_e)
            q[nb:] = _q(sim_f, alpha)
        return _metrics(Sb, yb, Sr, yr, Su, q, nb)

    if mode.startswith("pmerge"):
        if "D_cal" not in rd:
            return None
        # pmerge-*  : ECDF and quantile share the whole calib set (double use ->
        #             approximate; kept as the upper-signal diagnostic).
        # pmerge2-* : split-half calib -- odd rows build the per-(class,view)
        #             ECDF reference, even rows supply the own-score quantile ->
        #             an honest split-conformal calibration of the merged score.
        split = mode.startswith("pmerge2")
        hyb = mode.endswith("hyb")
        V = rd["D_cal"].shape[2]
        rng = np.random.RandomState(0)

        def halves(mask):
            rows = np.where(mask)[0]
            rng.shuffle(rows)
            h = max(1, len(rows) // 2)
            return rows[:h], rows[h:]

        ref, qrows_b = {}, []
        for k in range(nb):
            ra, rb_ = (halves(yc == k) if split
                       else (np.where(yc == k)[0], np.where(yc == k)[0]))
            qrows_b.append(rb_)
            for v in range(V):
                ref[(k, v)] = np.sort(rd["D_cal"][ra, k, v])
        qrows_b = np.concatenate(qrows_b) if qrows_b else np.zeros(0, int)
        has_ncal = len(rd.get("D_cal_reg", [])) > 0
        qrows_n = {}
        if hyb and has_ncal:
            ycr = rd["y_cal_reg"]
            for c in range(nb, C):
                ra, rb_ = (halves(ycr == c) if split
                           else (np.where(ycr == c)[0], np.where(ycr == c)[0]))
                qrows_n[c] = rb_
                for v in range(V):
                    ref[(c, v)] = np.sort(rd["D_cal_reg"][ra, c, v])

        def fisher(D):                                # (N, C, V) -> (N, C)
            out = np.zeros(D.shape[:2])
            for c in range(C):
                for v in range(V):
                    cal = ref.get((c, v))
                    if cal is None or len(cal) == 0:  # no reference: borrow the
                        cal = ref[(c % nb, v)]        # base column's (crude)
                    ge = len(cal) - np.searchsorted(cal, D[:, c, v], "left")
                    p = (1.0 + ge) / (len(cal) + 1.0)
                    out[:, c] += -2.0 * np.log(p)
                                    # higher = further from the class
            return out
        Sc, Sb = fisher(rd["D_cal"]), fisher(rd["D_base"])
        Sr, Su = fisher(rd["D_reg"]), fisher(rd["D_unk"])
        qf = _q(_own(Sc, yc)[qrows_b] if split else _own(Sc, yc), alpha)
        q = np.full(C, qf, float)
        if hyb and has_ncal:
            Scr = fisher(rd["D_cal_reg"])
            own_n = _own(Scr, rd["y_cal_reg"])
            for c in range(nb, C):
                rows_c = (qrows_n[c] if split
                          else np.where(rd["y_cal_reg"] == c)[0])
                qc = _q(own_n[rows_c], alpha)
                q[c] = qc if np.isfinite(qc) else qf
        return _metrics(Sb, yb, Sr, yr, Su, q, nb)

    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="*",
                    default=["logs/n3_scores_bal_c0_s*.npz"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--route-s", type=float, default=1.0,
                    help=("SS40 incumbent handicap: scale the BASE columns by s "
                          "for the nearest-prototype routing line only (base "
                          "heads trained on ~600 samples dominate 5-shot heads; "
                          "s~1.5 trades ~2 pts of acc_base for ~+12 acc_new). "
                          "Conformal rows stay unscaled (acceptance decoupled)."))
    args = ap.parse_args()
    files = sorted(f for p in args.patterns for f in glob.glob(p))
    if not files:
        print("no score files match", args.patterns)
        return
    rds = []
    for f in files:
        rd = dict(np.load(f, allow_pickle=True))
        rd["n_base"], rd["n_reg"] = int(rd["n_base"]), int(rd["n_reg"])
        rds.append(rd)
    has_e = all("E_cal" in r and len(r["E_cal"]) for r in rds)
    has_sim = all(len(r.get("sim_r_cal", [])) for r in rds)
    has_d = all("D_cal" in r for r in rds)
    print(f"{len(files)} folds | e_recon={'yes' if has_e else 'NO'} "
          f"sim_task={'yes' if has_sim else 'NO'} "
          f"per-view={'yes' if has_d else 'NO'} | alpha={args.alpha}")

    # nearest-prototype closed accuracy (the delta-aug headline number).
    # --route-s applies the SS40 incumbent handicap to the routing only.
    def _acc(r):
        nb = r["n_base"]

        def sc(R):
            R = R.copy()
            R[:, :nb] *= args.route_s
            return R
        return (float((sc(r["R_base"]).argmin(1) == r["y_base"]).mean()),
                float((sc(r["R_reg"]).argmin(1) == r["y_reg"]).mean()))
    accs = [_acc(r) for r in rds]
    stag = f" (route-s={args.route_s})" if args.route_s != 1.0 else ""
    print(f"nearest-proto acc{stag}: "
          f"base {100*np.mean([a for a, _ in accs]):.2f} "
          f"+/- {100*np.std([a for a, _ in accs]):.2f} | "
          f"new {100*np.mean([b for _, b in accs]):.2f} "
          f"+/- {100*np.std([b for _, b in accs]):.2f}")

    rows = [("global", "global", 0), ("mondrian", "mondrian", 0),
            ("hybrid (oracle)", "hybrid", 0)]
    if has_sim:
        rows.append(("simtask", "simtask", 0))
    if has_e:
        rows += [(f"fuse l={l} [glob]", "fuse-glob", l) for l in LAMBDAS]
        rows += [(f"fuse l={l} [base]", "fuse-base", l) for l in LAMBDAS]
        rows.append(("fuse l=1 [base]+hyb", "fuse-base+hyb", 1.0))
    if has_e and has_sim:
        rows += [(f"fuse l={l} [sim]", "fuse-sim", l) for l in LAMBDAS]
    if has_d:
        rows += [("pmerge-glob", "pmerge-glob", 0),
                 ("pmerge-hyb", "pmerge-hyb", 0),
                 ("pmerge2-glob", "pmerge2-glob", 0),
                 ("pmerge2-hyb", "pmerge2-hyb", 0)]
    rows.append(("eta=3", "eta", 0))

    keys = ["cov_b", "cov_n", "sz_b", "sz_n", "rejU", "leakB", "auroc",
            "sAcc_n"]
    print(f"\n{'mode':18s}" + "".join(f"{k:>8s}" for k in keys)
          + f"   (target cov {100*(1-args.alpha):.0f})")
    for name, mode, lam in rows:
        vals = [fold_report(rd, args.alpha, mode, lam=lam) for rd in rds]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        m = {k: float(np.nanmean([v[k] for v in vals])) for k in keys}
        print(f"{name:18s}"
              + "".join(f"{m[k]*100:8.1f}" if k not in ("sz_b", "sz_n")
                        else f"{m[k]:8.2f}" for k in keys))


if __name__ == "__main__":
    main()
