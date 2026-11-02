"""Post-hoc conformal analysis for paper-3 N3 (no retraining).

Reads the per-fold score dumps written by `main.py --conformal` and computes,
for several accept/reject threshold strategies, the coverage / set-size / unknown
rejection / singleton metrics, averaged over folds. Because the dumps store the
normalised distances r and the per-class sigma, the sigma-shrinkage correction is
a pure rescale (r* = r * sigma / sigma*), so any lambda is evaluated here without
touching the GPU.

Usage:
  python tools/n3_analyze.py logs/n3_scores_bal_c0_s*.npz
  python tools/n3_analyze.py logs/n2_scores_c0_s*.npz      # the N2 hybrid run
"""
import sys, glob
import numpy as np


def _q(scores, alpha):
    scores = np.asarray(scores, float)
    n = len(scores)
    if n == 0:
        return np.inf
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if rank > n else float(np.sort(scores)[rank - 1])


def _metrics(R, ytrue, q):
    sets = R <= q[None, :]
    sizes = sets.sum(1)
    if ytrue is None:
        return dict(empty=float((sizes == 0).mean()), size=float(sizes.mean()))
    cov = float(sets[np.arange(len(R)), ytrue].mean())
    sing = sizes == 1
    sa = float((sets[sing].argmax(1) == ytrue[sing]).mean()) if sing.sum() else 0.0
    return dict(cov=cov, size=float(sizes.mean()),
                empty=float((sizes == 0).mean()), sing_acc=sa)


def _shrink(R, sigma_bar, n_base, lam):
    """Rescale the new-class columns by sigma/sigma* (sigma-shrinkage toward the
    base-pool sigma). r* = r * sigma / sigma*, sigma*^2=(1-lam) sigma^2+lam pool^2."""
    if R.shape[0] == 0:
        return R
    sig = sigma_bar.copy()
    pool = float(sig[:n_base].mean())
    out = R.copy()
    for c in range(n_base, len(sig)):
        star = np.sqrt((1 - lam) * sig[c] ** 2 + lam * pool ** 2)
        out[:, c] = R[:, c] * sig[c] / (star + 1e-9)
    return out


def q_vector(rd, alpha, mode, eta=3.0):
    C = rd["n_base"] + rd["n_reg"]
    nb = rd["n_base"]
    if mode == "eta":
        return np.full(C, float(eta))
    own_b = rd["R_cal"][np.arange(len(rd["R_cal"])), rd["y_cal"]]
    qg = _q(own_b, alpha)
    q = np.full(C, qg, float)
    if mode in ("hybrid", "mondrian") and "R_cal_reg" in rd and len(rd["R_cal_reg"]):
        Rcr, ycr = rd["R_cal_reg"], rd["y_cal_reg"]
        own_n = Rcr[np.arange(len(Rcr)), ycr]
        for c in range(nb, C):
            qc = _q(own_n[ycr == c], alpha)
            q[c] = qc if np.isfinite(qc) else qg
    if mode == "mondrian":
        for k in range(nb):
            qk = _q(own_b[rd["y_cal"] == k], alpha)
            q[k] = qk if np.isfinite(qk) else qg
    return q


def report(rd, alpha, mode, eta=3.0, lam=0.0):
    nb = rd["n_base"]
    q = q_vector(rd, alpha, "global" if mode == "shrink" else mode, eta)
    Rb, Rr, Ru = rd["R_base"], rd["R_reg"], rd["R_unk"]
    if mode == "shrink":                           # rescale new-class columns
        sig = rd["sigma_bar"]
        Rb = _shrink(Rb, sig, nb, lam)
        Rr = _shrink(Rr, sig, nb, lam)
        Ru = _shrink(Ru, sig, nb, lam)
    b = _metrics(Rb, rd["y_base"], q)
    r = _metrics(Rr, rd["y_reg"], q)
    u = _metrics(Ru, None, q)
    return dict(cov_base=b["cov"], cov_new=r["cov"], size_base=b["size"],
                size_new=r["size"], reject_unknown=u["empty"],
                sing_acc_base=b["sing_acc"], sing_acc_new=r["sing_acc"])


def main():
    pats = sys.argv[1:] or ["logs/n3_scores_bal_c0_s*.npz"]
    files = sorted(f for p in pats for f in glob.glob(p))
    if not files:
        print("no score files match", pats); return
    rds = [dict(np.load(f, allow_pickle=True)) for f in files]
    for r in rds:
        r["n_base"] = int(r["n_base"]); r["n_reg"] = int(r["n_reg"])
    has_sig = all("sigma_bar" in r for r in rds)
    has_newcal = all("R_cal_reg" in r and len(r["R_cal_reg"]) for r in rds)
    print(f"{len(files)} folds | sigma_bar={'yes' if has_sig else 'NO'} "
          f"new-calib={'yes' if has_newcal else 'NO'}")
    alpha = 0.1

    rows = []
    rows.append(("global (N2)", "global", 0.0))
    if has_newcal:
        rows.append(("hybrid (oracle)", "hybrid", 0.0))
    if has_sig:
        for lam in (0.25, 0.5, 0.75, 0.9, 1.0):
            rows.append((f"shrink lam={lam}", "shrink", lam))
    rows.append(("eta=3", "eta", 0.0))

    print(f"\n{'mode':18s} {'cov_base':>8s} {'cov_new':>8s} {'sz_b':>6s} "
          f"{'sz_n':>6s} {'rejU':>6s} {'sAcc_b':>7s} {'sAcc_n':>7s}  (target cov "
          f"{100*(1-alpha):.0f})")
    for name, mode, lam in rows:
        vals = [report(rd, alpha, mode, lam=lam) for rd in rds]
        m = {k: np.mean([v[k] for v in vals]) * 100 for k in vals[0]}
        print(f"{name:18s} {m['cov_base']:8.1f} {m['cov_new']:8.1f} "
              f"{m['size_base']/100:6.2f} {m['size_new']/100:6.2f} "
              f"{m['reject_unknown']:6.1f} {m['sing_acc_base']:7.1f} "
              f"{m['sing_acc_new']:7.1f}")


if __name__ == "__main__":
    main()
