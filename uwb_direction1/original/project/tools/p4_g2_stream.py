"""P4 G2 adaptive runner: dual memory + risk controller on the controlled
stream (design doc SS5.2/5.5/5.6, gate SS8 G2).

Reuses p4_stream.setup_fold (same cached encoders, per-fold-seeded banks,
schedule and calib conformal state as the G0 grid), then runs the ONLINE
loop per (condition x budget):

  * per-condition features: the frozen encoder's f_views + e_recon for
    every stream event, extracted once per condition (SS4.5 compute cache;
    sev=0 events reuse the fold's clean features);
  * sequential decision loop in stream order -- distances are recomputed
    from the feature cache for the remaining events after every plastic
    update (SS4.5 online-head rule), scores are z-fused exactly like the
    frozen recipe, actions come from uwb_osr.risk_control.RiskController;
  * labels: the evaluator answers query events with the hidden label
    (that is the user in deployment); random-slice answers update online
    conformal (known-class events only -- a true-unknown answer has no
    coverage indicator; the spent budget is still charged) and may enter
    the plastic pool at weight 1; uncertainty-slice answers touch plastic
    memory only; adoption feeds the pool at low weight, never thresholds;
  * per-view reliability is v1-proxied by pooled e_recon vs the calib p99
    (views_alert blocks adoption, SS5.4 fallback discipline);
  * prequential evaluation: every event is scored (routing pred + set
    membership) BEFORE any state update at that event.

B in {0, 2, 5, 10} (SS5.5): B=0 = frozen static reference (no queries, no
online state, no adoption -- the G0 system); random-slice slots are
rebuilt per budget from the stream seed (label- and score-free).

Outputs logs/p4_g2_c{ci}_{cond}_B{b}_s{ts}.npz with per-event logs
(action/pred/covered/set size) + pool stats; resume-safe per file.

Usage:
  python tools/p4_g2_stream.py --case 0 --max-folds 2 --deg clean,drop,domain --device cuda --mmap
  python tools/p4_g2_stream.py --case 0 --budgets 0,2,5,10 --device cuda --mmap
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from p4_stream import (G0_FAMILIES, CHUNK, _cond_sfx,        # noqa: E402
                       _degrade_sample, setup_fold)
from config import Config                                     # noqa: E402
from uwb_osr.adaptation import DualMemory                     # noqa: E402
from uwb_osr.data import MVDataset, load_cache                # noqa: E402
from uwb_osr.engine import get_device                         # noqa: E402
from uwb_osr.inspector import extract                         # noqa: E402
from uwb_osr.pretrain import recon_error                      # noqa: E402
from uwb_osr.registration import proto_dist_matrix            # noqa: E402
from uwb_osr.risk_control import (OnlineConformal,            # noqa: E402
                                  RiskController, route)
from uwb_osr.stream import StreamSpec, _random_query_slots    # noqa: E402

ACTION_CODE = {"adopt": 0, "query_rand": 1, "query_unc": 2,
               "accept": 3, "reject": 4}


def condition_features(fa, fam, vt, cfg, device, Xev_raw=None):
    """(n_ev, V, D) frozen features + e_recon for one drift condition.
    sev=0 rows reuse the clean extraction (computed on the first call and
    cached on the fold assets)."""
    if not hasattr(fa, "F_clean"):
        F, E = _extract(fa, fa.Xev, cfg, device)
        fa.F_clean, fa.E_clean_feat = F, E
    if fam == "clean":
        return fa.F_clean, fa.E_clean_feat
    F = fa.F_clean.clone()
    E = fa.E_clean_feat.copy()
    vlist = [0, 1, 2] if vt == "all" else [int(vt)]
    spec = StreamSpec(seed=fa.seed, budget_per_100=5, deg_type=fam,
                      deg_view=vlist[0])
    hot = np.where(fa.ev_sev > 0)[0]
    for a in range(0, len(hot), CHUNK):
        rows = hot[a:a + CHUNK]
        Xd = np.stack([_degrade_sample(
            fa.Xev[i], None if Xev_raw is None else Xev_raw[i], fam,
            vlist, float(fa.ev_sev[i]), fa.seed, int(i), spec)
            for i in rows])
        Fd, Ed = _extract(fa, Xd, cfg, device)
        F[rows], E[rows] = Fd, Ed
    return F, E


def _extract(fa, Xrows, cfg, device):
    n = len(Xrows)
    ds = MVDataset(Xrows, np.zeros(n, np.int64), np.arange(n),
                   list(range(len(cfg.views))), None, cfg.normalize)
    F, _, _ = extract(fa.model, ds, cfg, device)
    E = recon_error(fa.enc0, fa.dec0, ds, cfg, device)
    return F, E


def run_stream(fa, F, E, budget, args, device):
    """One online pass; returns per-event logs + final state summaries."""
    nb, nr, V = fa.nb, fa.nr, fa.V
    n_ev = len(fa.events)
    lam, alpha, s = args.lam, args.alpha, args.s_route

    # frozen conformal recipe pieces (identical to the G0 scorer)
    zr = lambda R: (R - fa.mu_r) / fa.sd_r                    # noqa: E731
    ze = lambda e: (e - fa.mu_e) / fa.sd_e                    # noqa: E731

    # fixed q80 adoption gate from calib true-class fused scores (SS5.5);
    # --gate ablations (SS6.2): "all" opens it, "none" closes adoption.
    own_b = zr(fa.own_base) + lam * ze(fa.E_cal)
    own_n = (zr(fa.R_cal_reg[np.arange(len(fa.R_cal_reg)), fa.y_cal_reg])
             if len(fa.R_cal_reg) else np.zeros(0))
    q80 = float(np.percentile(np.concatenate([own_b, own_n]), 80))
    if args.gate == "all":
        q80 = np.inf                # every new-class singleton adopts
    no_adopt = args.gate == "none"  # confirm-only: adoption intercepted at
    #                                 the action layer so the uncertainty
    #                                 band (q80 < top1 <= q) stays intact
    e99 = float(np.percentile(fa.E_cal, 99))

    b_rand = 0 if budget <= 0 else max(1, int(round(0.2 * budget)))
    slots = (_random_query_slots(
        n_ev, StreamSpec(seed=fa.seed, budget_per_100=budget))
        if budget > 0 else set())

    mem = DualMemory(fa.bank, nb, nr, fa.Fs, fa.ys_sup, device,
                     cap=args.cap, w_auto=args.w_auto)
    oc = OnlineConformal(alpha)
    rc = RiskController(alpha, budget, b_rand, q80, online=oc)
    frozen = budget <= 0                    # SS5.5 B=0: static reference

    R = proto_dist_matrix(mem.bank(), fa.classes, F, V, device)
    act = np.zeros(n_ev, np.int64)
    pred = np.zeros(n_ev, np.int64)
    covered = np.zeros(n_ev, bool)
    setsz = np.zeros(n_ev, np.int64)
    adopt_true = []                          # evaluator: adoption pool truth
    kind, ytrue = fa.kind, fa.ytrue

    for t in range(n_ev):
        S_row = zr(R[t]).astype(float)
        S_row[:nb] += lam * ze(E[t])
        q_eff = oc.q_eff(fa.q)
        member = S_row <= q_eff
        pred[t] = route(R[t], nb, s)
        covered[t] = bool(member[ytrue[t]]) if ytrue[t] >= 0 else False
        setsz[t] = int(member.sum())

        if frozen:
            act[t] = ACTION_CODE["accept" if setsz[t] else "reject"]
            continue
        a, col = rc.decide(t, S_row, fa.q, nb,
                           random_query=(t in slots),
                           views_alert=bool(E[t] > e99))
        if a == "adopt" and no_adopt:
            a = "accept"
        act[t] = ACTION_CODE[a]
        dirty = False
        if a == "adopt":
            mem.add(col, F[t], "auto")
            adopt_true.append(int(kind[t] == 1 and ytrue[t] == col))
            dirty = True
        elif a in ("query_rand", "query_unc"):   # label revealed by the user
            true_c = int(ytrue[t])
            if true_c >= 0:                      # known class: hit indicator
                hit = bool(member[true_c])
                rc.record_confirmation(hit)
                if a == "query_rand":
                    oc.update(hit, from_random_slice=True)
                if true_c >= nb:                 # register class -> plastic
                    mem.add(true_c, F[t], "confirm")
                    dirty = True
        if dirty and t + 1 < n_ev:               # SS4.5 online-head rule
            R[t + 1:] = proto_dist_matrix(mem.bank(), fa.classes,
                                          F[t + 1:], V, device)

    b, r, u = kind == 0, kind == 1, kind == 2
    acc_b = float((pred[b] == ytrue[b]).mean())
    acc_n = float((pred[r] == ytrue[r]).mean())
    n_adopt = len(adopt_true)
    spend = rc.spend_summary()
    return dict(
        act=act, pred=pred, covered=covered, setsz=setsz,
        acc_b=acc_b, acc_n=acc_n,
        H=2 * acc_b * acc_n / (acc_b + acc_n + 1e-9),
        cov_b=float(covered[b].mean()), cov_n=float(covered[r].mean()),
        rej_u=float((setsz[u] == 0).mean()),
        n_adopt=n_adopt,
        contam=(1.0 - float(np.mean(adopt_true)) if n_adopt else np.nan),
        n_rand=sum(v["rand"] for v in spend.values()),
        n_unc=sum(v["unc"] for v in spend.values()),
        n_confirm=len(rc._confirmed), delta=oc.delta, t_oc=oc.t,
        pool={c: st for c, st in mem.pool_stats().items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--deg", default="clean,drop,domain",
                    help="comma list of conditions: fam or fam_vN/fam_vall")
    ap.add_argument("--budgets", default="0,5")
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--s-route", type=float, default=1.5)
    ap.add_argument("--cap", type=int, default=20)
    ap.add_argument("--w-auto", type=float, default=0.25,
                    help="auto-adoption weight (SS6.2 ablation; <=0.25)")
    ap.add_argument("--gate", choices=["q80", "all", "none"], default="q80",
                    help="adoption gate ablation (SS6.2): q80 = strict "
                         "fixed gate (main); all = every new-class "
                         "singleton adopts; none = confirm-only, "
                         "auto-adoption off")
    ap.add_argument("--tag", default="",
                    help="dump-name suffix for ablation variants")
    ap.add_argument("--budget", type=int, default=5)   # setup_fold schedule tag
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mmap", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-folds", type=int, default=-1)
    ap.add_argument("--calib-rotate", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    if args.deg == "all":
        conds = [("clean", None)] + [(f, vt) for vt in (0, 1, 2, "all")
                                     for f in G0_FAMILIES]
    else:
        conds = []
        for tok in args.deg.split(","):
            tok = tok.strip()
            if tok == "clean":
                conds.append(("clean", None))
                continue
            fam, _, vt = tok.partition("_v")
            if fam not in G0_FAMILIES:
                raise SystemExit(f"unknown family {fam!r}")
            conds.append((fam, vt if vt in ("all",)
                          else int(vt) if vt else 0))

    cfg = Config()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    cfg.use_fusion = False
    cfg.pool = "flatten"
    cfg.normalize = "abs_std"
    cfg.use_mae_pretrain = True
    cfg.device = args.device
    device = get_device(cfg)
    from uwb_osr.data import make_synthetic
    if args.smoke:
        cfg.epochs, cfg.mae_epochs = 3, 2
        cfg.inspector_epochs, cfg.n_pos_syn, cfg.n_neg_syn = 5, 400, 300
        X, y, subj = make_synthetic(cfg, n_per_class=80, seed=cfg.seed)
    else:
        X, y, subj = load_cache(cfg, mmap=True)
    X_raw = None
    if any(f == "domain" for f, _ in conds) and not args.smoke:
        import copy
        cfg_raw = copy.copy(cfg)
        cfg_raw.data_type = "Raw"
        X_raw, y_raw, subj_raw = load_cache(cfg_raw, mmap=True)
        if not (np.array_equal(y, y_raw) and np.array_equal(subj, subj_raw)):
            raise SystemExit("Raw cache rows misaligned")
    if args.smoke:
        conds = [c for c in conds if c[0] != "domain"]
    subjects = sorted(set(subj.tolist()))
    folds = subjects[:args.max_folds] if args.max_folds > 0 else subjects
    print(f"== P4 G2 adaptive stream ==  case={args.case} conds={conds} "
          f"budgets={budgets} cap={args.cap} s={args.s_route} "
          f"folds={len(folds)}/{len(subjects)} device={device}", flush=True)

    rows = {}                              # (cond_key, B) -> [fold dicts]
    for ts in folds:
        if args.calib_rotate:
            cs = subjects[(subjects.index(ts) + 1) % len(subjects)]
        else:
            cs = next(s for s in subjects if s != ts)
        fa = setup_fold(cfg, X, y, subj, args.case, ts, cs, device, args)
        Xev_raw = (np.asarray(X_raw[fa.ev_idx], np.float32)
                   if X_raw is not None else None)
        line = f"  subj {ts}:"
        for fam, vt in conds:
            key = fam + ("" if fam == "clean" else _cond_sfx(vt))
            F = E = None
            for B in budgets:
                tag = f"_{args.tag}" if args.tag else ""
                f = os.path.join(
                    "logs", f"p4_g2{tag}_c{args.case}_{key}_B{B}_s{ts}.npz")
                if os.path.exists(f) and not args.overwrite:
                    d = np.load(f, allow_pickle=True)
                    rows.setdefault((key, B), []).append(
                        {k: d[k].item() if d[k].ndim == 0 else d[k]
                         for k in d.files if k != "pool"})
                    continue
                if F is None:
                    F, E = condition_features(fa, fam, vt, cfg, device,
                                              Xev_raw)
                r = run_stream(fa, F, E, B, args, device)
                pool = r.pop("pool")
                np.savez(f, **r, pool=np.array(str(pool)))
                rows.setdefault((key, B), []).append(r)
            got = rows.get((key, budgets[-1]))
            if got:
                line += (f"  [{key}] B{budgets[-1]} "
                         f"H={100*got[-1]['H']:.1f}")
        print(line, flush=True)
        del fa
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n===== G2 SUMMARY (case {args.case}, mean over {len(folds)} "
          f"folds, %) =====")
    print(f"{'condition':13s} {'B':>3s} {'acc_b':>6s} {'acc_n':>6s} "
          f"{'H':>6s} {'cov_b':>6s} {'cov_n':>6s} {'rejU':>5s} "
          f"{'adopt':>6s} {'contam':>7s} {'q_rand':>7s} {'q_unc':>6s}")
    for (key, B), fr in rows.items():
        def m(k):
            return 100 * float(np.nanmean([r[k] for r in fr]))
        na = float(np.mean([r["n_adopt"] for r in fr]))
        qr = float(np.mean([r["n_rand"] for r in fr]))
        qu = float(np.mean([r["n_unc"] for r in fr]))
        print(f"{key:13s} {B:3d} {m('acc_b'):6.1f} {m('acc_n'):6.1f} "
              f"{m('H'):6.1f} {m('cov_b'):6.1f} {m('cov_n'):6.1f} "
              f"{m('rej_u'):5.1f} {na:6.1f} {m('contam'):7.1f} "
              f"{qr:7.1f} {qu:6.1f}")


if __name__ == "__main__":
    main()
