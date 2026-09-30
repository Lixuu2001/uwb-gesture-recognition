"""P4 G0 controlled-stream runner: the FROZEN paper-3 recipe under drift.

Runs the frozen static system (no adaptation, no online updates) over the
deterministic pseudo-time stream of uwb_osr/stream.py (design doc SS4.3-4.5)
and dumps per-event scores for the G0 verdict (SS8 G0: does compound drift
actually break the frozen recipe?).

Frozen recipe (mirrors the Run-A b2e reference + SS40 deployment recipe v2):
  * cached fold encoder (--enc-cache keys, Run-A non-rotated pairing) --
    a cache MISS aborts unless --smoke, so a typo can never trigger a
    silent 30-min retrain inside a scoring run;
  * method-A registration, K=5 subject-balanced supports;
  * acceptance = conformal on UNSCALED fused scores: z(r) everywhere,
    + lam * z(e_recon) on base columns only (SS38 fuse-base lam=1),
    hybrid thresholds (pooled base q + per-class held-out new q, SS39);
  * routing = argmin over R with base columns * s (SS40, s=1.5), decoupled
    from acceptance;
  * singleton-adoption gate (SS39): singleton set on a new-class column =>
    the sample WOULD be silently adopted; G0 only measures the resulting
    pool contamination, it never updates anything.

Stream/pairing discipline:
  * spec.seed = 1000*case + test_subject and the schedule draws depend only
    on (seed, pool sizes), so every degradation family sees the IDENTICAL
    event order -> family files align row-by-row with the 'clean' file;
  * only sev>0 events are re-encoded; sev=0 rows reuse the clean pass
    (SS4.5: one condition materialised at a time, freed after scoring);
  * labels: the runner touches them only inside evaluator blocks (metrics /
    dump); nothing label-dependent feeds back into scores or state.
  * degraded views: --deg-view is a comma list of view slots and/or 'all'
    (every view). One fold visit scores every (family x view) condition
    against the SAME registration and clean pass; v0 keeps the legacy file
    name, other views suffix _v{n} / _vall.
  * 'domain' family (SS4.4 processing-domain switch): the degraded view(s)
    blend toward the row-aligned Raw cache, x = (1-sev)*CR + sev*Raw --
    severity = residual-clutter fraction (an imperfect clutter filter).
    Needs cache/X_raw.npy; unavailable under --smoke (synthetic).

Outputs per (case, family, view, fold): logs/p4_stream_c{ci}_{fam}[_v*]_s{ts}.npz
  R (N,C) sigma-normalised distances, E (N) recon error, per-event meta
  (t/seg/sev/kind/ytrue/random_query), q vector + z-stats, config scalars.
Existing files are skipped (resume) unless --overwrite.

Usage:
  python tools/p4_stream.py --case 0 --max-folds 2 --device cuda --mmap  # smoke
  python tools/p4_stream.py --case 0 --device cuda --mmap               # full
  python tools/p4_stream.py --smoke --device cpu                        # synthetic
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, REG_TEST_CASES, VIEW_ORDER            # noqa: E402
from main import _enc_cache_path                                 # noqa: E402
from uwb_osr.data import (load_cache, make_synthetic, MVDataset,  # noqa: E402
                          register_split_conf)
from uwb_osr.engine import train_one_case, get_device            # noqa: E402
from uwb_osr.inspector import extract                            # noqa: E402
from uwb_osr.model import MultiViewOSRNet, SharedDualPathEncoder  # noqa: E402
from uwb_osr.pretrain import (pretrain_encoder, recon_error,      # noqa: E402
                              MaskedViewDecoder)
from uwb_osr.registration import (_register_method_a, _conformal_q,  # noqa: E402
                                  proto_dist_matrix)
from uwb_osr.stream import (StreamSpec, apply_degradation,        # noqa: E402
                            build_stream)

G0_FAMILIES = ("atten", "noise", "shift", "occlude", "drop", "domain")
CHUNK = 256                       # degraded samples materialised at a time


def _cond_sfx(view_tok):
    """File/key suffix for a degraded-view token; v0 keeps the legacy name."""
    return "" if view_tok in (None, 0, "0") else f"_v{view_tok}"


def _degrade_sample(x, x_raw, fam, view_list, sev, seed, t, spec):
    """Apply one family at one severity to the listed views of one sample."""
    if fam == "domain":
        out = x.copy()
        for v in view_list:
            out[v] = (1.0 - sev) * x[v] + sev * x_raw[v]
        return out
    out = x
    for v in view_list:
        out = apply_degradation(out, v, fam, sev, seed, t,
                                spec.max_shift_bins, spec.occlude_frac)
    return out


def _get_encoder(cfg, args, ci, ts, cs, train_ds, device):
    """Cached fold encoder + Stage-0 pair; MISS aborts outside --smoke."""
    path = _enc_cache_path(cfg, SimpleNamespace(mae_no_init=False), ci, ts, cs)
    if os.path.exists(path):
        ck = torch.load(path, map_location=device)
        model = MultiViewOSRNet(cfg, len(REG_TEST_CASES[ci]["base"])).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        enc0 = SharedDualPathEncoder(cfg).to(device)
        enc0.load_state_dict(ck["mae_enc"])
        dec0 = MaskedViewDecoder(cfg).to(device)
        dec0.load_state_dict(ck["mae_dec"])
        return model, enc0, dec0
    if not args.smoke:
        raise SystemExit(f"enc-cache MISS for c{ci} s{ts} cs{cs}: {path}\n"
                         "G0 must run on the frozen Run-A encoders; refusing "
                         "to retrain inside a scoring run.")
    mae_state, enc0, dec0 = pretrain_encoder(cfg, train_ds, device, False)
    model = train_one_case(cfg, train_ds, len(REG_TEST_CASES[ci]["base"]),
                           device, verbose=False, desc=f"g0 c{ci} s{ts} ",
                           mae_state=mae_state)
    return model, enc0, dec0


def _score_rows(cfg, model, enc0, dec0, bank, classes, V, Xrows, device):
    """Frozen forward on a materialised (n, V, T, M) block -> (R, E)."""
    n = len(Xrows)
    dummy = np.zeros(n, np.int64)
    ds = MVDataset(Xrows, dummy, np.arange(n), list(range(len(cfg.views))),
                   None, cfg.normalize)
    F, _, _ = extract(model, ds, cfg, device)
    R = proto_dist_matrix(bank, classes, F, V, device)
    E = recon_error(enc0, dec0, ds, cfg, device)
    return R, E


def setup_fold(cfg, X, y, subj, ci, ts, cs, device, args):
    """Per-fold frozen assets shared by the G0 grid scorer and the G2
    adaptive runner (tools/p4_g2_stream.py): cached encoder, method-A bank,
    calib conformal state (dumped for post-hoc), the deterministic stream
    schedule, evaluator-side truth arrays and the materialised event
    tensor. Every forward here is eval-mode (RNG-free), so extractions
    added at the end cannot shift the per-fold bank RNG trajectory."""
    rc = REG_TEST_CASES[ci]
    pfx = "p4_stream_smoke" if args.smoke else "p4_stream"
    # bank determinism: inspector training consumes the GLOBAL torch/np RNG,
    # so the registered bank would otherwise depend on how many folds and
    # conditions ran before it in this process (two ignitions of the same
    # fold produced q vectors 0.36-0.59 apart). Re-seed per fold: any visit
    # (smoke, resume back-fill, full regen) rebuilds the same bank.
    fold_seed = 100000 + 1000 * ci + ts
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)
    view_idx = [VIEW_ORDER.index(v) for v in cfg.views]
    sp = register_split_conf(cfg, y, subj, rc["base"], rc["register"],
                             rc["unknown"], ts, cs, args.n_shot, cfg.seed,
                             subject_balanced=True)
    nb, nr = sp["n_base"], sp["n_register"]
    classes = list(range(nb + nr))
    nrm = cfg.normalize

    def ds(idx, lm):
        return MVDataset(X, y, idx, view_idx, lm, nrm)
    train_ds = ds(sp["train_idx"], sp["label_map"])
    model, enc0, dec0 = _get_encoder(cfg, args, ci, ts, cs, train_ds, device)
    support_ds = ds(sp["support_idx"], sp["label_map"])
    bank, V = _register_method_a(model, train_ds, support_ds, cfg, device, nb)

    # frozen conformal state from the held-out calib subject (SS38/SS39):
    # z-stats + pooled base q on fused scores, per-class new q on z(r).
    Fc, _, yc = extract(model, ds(sp["calib_idx"], sp["label_map"]), cfg, device)
    E_cal = recon_error(enc0, dec0, ds(sp["calib_idx"], None), cfg, device)
    R_cal = proto_dist_matrix(bank, classes, Fc, V, device)
    own = R_cal[np.arange(len(R_cal)), yc]
    mu_r, sd_r = float(own.mean()), float(own.std() + 1e-8)
    mu_e, sd_e = float(E_cal.mean()), float(E_cal.std() + 1e-8)
    q = np.full(nb + nr, _conformal_q((own - mu_r) / sd_r, args.alpha))
    q[:nb] = _conformal_q((own - mu_r) / sd_r
                          + args.lam * (E_cal - mu_e) / sd_e, args.alpha)
    Fcr, _, ycr = extract(model, ds(sp["calib_reg_idx"], sp["label_map"]),
                          cfg, device)
    Rcr = np.zeros((0, nb + nr), np.float32)
    if len(Fcr):
        Rcr = proto_dist_matrix(bank, classes, Fcr, V, device)
        own_n = (Rcr[np.arange(len(Rcr)), ycr] - mu_r) / sd_r
        for c in range(nb, nb + nr):
            qc = _conformal_q(own_n[ycr == c], args.alpha)
            if np.isfinite(qc):
                q[c] = qc
    # calibration rows from THIS bank (s_hat post-hoc needs them; the n3
    # dumps come from a different RNG trajectory = a different bank)
    np.savez(os.path.join("logs", f"{pfx}_c{ci}_calib_s{ts}.npz"),
             R_cal=R_cal, y_cal=yc, E_cal=E_cal, R_cal_reg=Rcr, y_cal_reg=ycr,
             q=q, nb=nb, nr=nr, ts=ts, cs=cs, case=ci)

    # stream schedule: pools are the test subject's rows; the schedule and
    # the event order depend only on spec.seed -> identical across families.
    b_idx, r_idx, u_idx = (np.asarray(sp[k]) for k in
                           ("test_base_idx", "test_reg_idx",
                            "test_unknown_idx"))
    seed = 1000 * ci + ts
    # deg_type/deg_view are schedule tags only -- the order and severity
    # draws depend solely on (seed, pool sizes), identical for every family
    events = build_stream(b_idx, r_idx, u_idx,
                          StreamSpec(seed=seed, budget_per_100=args.budget,
                                     deg_type="noise", deg_view=0))
    n_ev = len(events)
    ev_idx = np.array([e.sample_idx for e in events])
    ev_seg = np.array([e.segment for e in events], np.int64)
    ev_sev = np.array([e.severity for e in events])
    ev_rq = np.array([e.random_query for e in events])

    # evaluator-side truth (dump/metrics only, never fed back)
    kind = np.where(np.isin(ev_idx, b_idx), 0,
                    np.where(np.isin(ev_idx, r_idx), 1, 2))
    ytrue = np.array([sp["label_map"].get(int(y[i]), -1) for i in ev_idx])
    ytrue[kind == 2] = -1

    Xev = np.asarray(X[ev_idx], np.float32)
    meta = dict(t=np.arange(n_ev), sample_idx=ev_idx, seg=ev_seg, sev=ev_sev,
                kind=kind, ytrue=ytrue, random_query=ev_rq,
                q=q, mu_r=mu_r, sd_r=sd_r, mu_e=mu_e, sd_e=sd_e,
                nb=nb, nr=nr, s_route=args.s_route, lam=args.lam,
                alpha=args.alpha, seed=seed, ts=ts, cs=cs, case=ci)
    # support features feed the G2 plastic pools (eval forward, RNG-free)
    Fs, _, ys_sup = extract(model, support_ds, cfg, device)
    return SimpleNamespace(
        model=model, enc0=enc0, dec0=dec0, bank=bank, V=V, sp=sp,
        nb=nb, nr=nr, classes=classes, q=q, mu_r=mu_r, sd_r=sd_r,
        mu_e=mu_e, sd_e=sd_e, E_cal=E_cal, own_base=own,
        R_cal_reg=Rcr, y_cal_reg=ycr, events=events, ev_idx=ev_idx,
        ev_seg=ev_seg, ev_sev=ev_sev, ev_rq=ev_rq, kind=kind, ytrue=ytrue,
        Xev=Xev, meta=meta, seed=seed, pfx=pfx, Fs=Fs, ys_sup=ys_sup)


def fold_stream(cfg, X, y, subj, ci, ts, cs, device, args, families, views,
                X_raw=None):
    fa = setup_fold(cfg, X, y, subj, ci, ts, cs, device, args)
    model, enc0, dec0, bank = fa.model, fa.enc0, fa.dec0, fa.bank
    nb, nr, classes, V = fa.nb, fa.nr, fa.classes, fa.V
    Xev, ev_sev, seed, meta, pfx = fa.Xev, fa.ev_sev, fa.seed, fa.meta, fa.pfx
    n_ev = len(fa.events)

    # clean pass: one frozen forward over the whole stream, in event order
    # (resume: reuse the clean dump so family back-fills skip the forward)
    fclean = os.path.join("logs", f"{pfx}_c{ci}_clean_s{ts}.npz")
    if os.path.exists(fclean) and not args.overwrite:
        d = np.load(fclean)
        R_clean, E_clean = d["R"], d["E"]
    else:
        R_clean = np.zeros((n_ev, nb + nr), np.float32)
        E_clean = np.zeros(n_ev, np.float32)
        for a in range(0, n_ev, CHUNK):
            z = slice(a, min(a + CHUNK, n_ev))
            R_clean[z], E_clean[z] = _score_rows(cfg, model, enc0, dec0, bank,
                                                 classes, V, Xev[z], device)

    Xev_raw = (np.asarray(X_raw[fa.ev_idx], np.float32)
               if X_raw is not None else None)
    conds = [("clean", None)] + [(fam, vt) for vt in views
                                 for fam in families]
    out = {}
    for fam, vt in conds:
        key = fam + _cond_sfx(vt)
        f = os.path.join("logs", f"{pfx}_c{ci}_{key}_s{ts}.npz")
        if os.path.exists(f) and not args.overwrite:
            d = np.load(f)
            out[key] = _fold_metrics(d["R"], d["E"], meta)
            continue
        if fam == "clean":
            R, E = R_clean, E_clean
        else:
            if fam == "domain" and Xev_raw is None:
                continue
            R, E = R_clean.copy(), E_clean.copy()
            vlist = [0, 1, 2] if vt == "all" else [int(vt)]
            hot = np.where(ev_sev > 0)[0]
            spec = StreamSpec(seed=seed, budget_per_100=args.budget,
                              deg_type=fam, deg_view=vlist[0])
            for a in range(0, len(hot), CHUNK):
                rows = hot[a:a + CHUNK]
                Xd = np.stack([_degrade_sample(
                    Xev[i], None if Xev_raw is None else Xev_raw[i], fam,
                    vlist, float(ev_sev[i]), seed, int(i), spec)
                    for i in rows])
                R[rows], E[rows] = _score_rows(cfg, model, enc0, dec0, bank,
                                               classes, V, Xd, device)
                del Xd
        np.savez(f, R=R, E=E, deg=key, **meta)
        out[key] = _fold_metrics(R, E, meta)
    del model, enc0, dec0, bank, Xev, Xev_raw
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def _fold_metrics(R, E, meta):
    """Evaluator: frozen-recipe decisions + G0 readouts for one fold/family."""
    nb, nr = int(meta["nb"]), int(meta["nr"])
    q, lam, s = meta["q"], float(meta["lam"]), float(meta["s_route"])
    kind, ytrue = meta["kind"], meta["ytrue"]
    S = (R - float(meta["mu_r"])) / float(meta["sd_r"])
    S[:, :nb] += lam * ((E - float(meta["mu_e"])) / float(meta["sd_e"]))[:, None]
    sets = S <= q[None, :]
    sizes = sets.sum(1)
    Rs = R.copy()
    Rs[:, :nb] *= s
    pred = Rs.argmin(1)

    def _slice(m):
        b, r, u = (m & (kind == 0)), (m & (kind == 1)), (m & (kind == 2))
        cov_b = float(sets[b, ytrue[b]].mean()) if b.any() else np.nan
        cov_n = float(sets[r, ytrue[r]].mean()) if r.any() else np.nan
        acc_b = float((pred[b] == ytrue[b]).mean()) if b.any() else np.nan
        acc_n = float((pred[r] == ytrue[r]).mean()) if r.any() else np.nan
        H = 2 * acc_b * acc_n / (acc_b + acc_n + 1e-9)
        rej_u = float((sizes[u] == 0).mean()) if u.any() else np.nan
        return dict(cov_b=cov_b, cov_n=cov_n, acc_b=acc_b, acc_n=acc_n,
                    H=H, rej_u=rej_u, n=int(m.sum()))

    full = _slice(np.ones(len(R), bool))
    hot = _slice(np.asarray(meta["sev"]) > 0)
    # SS39 adoption gate, measurement only: singleton on a new-class column.
    # contam  = uncapped rate over the whole stream (long-run diagnostic);
    # contam5 = the FROZEN recipe's actual pool (first m_adopt=5 singleton
    # hits per class in stream order, SS39) -- the G0 tripwire uses this.
    ad = (sizes == 1) & (sets[:, nb:].any(1))
    n_ad = int(ad.sum())
    col = sets.argmax(1)
    good = ad & (kind == 1) & (ytrue == col)
    contam = float(1 - good.sum() / n_ad) if n_ad else np.nan
    cnt = {c: 0 for c in range(nb, nb + nr)}
    n5 = bad5 = 0
    for i in np.where(ad)[0]:                      # stream order
        c = int(col[i])
        if cnt[c] < 5:
            cnt[c] += 1
            n5 += 1
            bad5 += int(not (kind[i] == 1 and ytrue[i] == c))
    contam5 = float(bad5 / n5) if n5 else np.nan
    return dict(full=full, hot=hot, n_adopt=n_ad, contam=contam,
                n_adopt5=n5, contam5=contam5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--deg", default="all",
                    help=f"comma list of {G0_FAMILIES} or 'all'")
    ap.add_argument("--deg-view", default="0",
                    help="comma list of degraded view slots (0/1/2) and/or "
                         "'all' (= every view at once)")
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--s-route", type=float, default=1.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mmap", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-folds", type=int, default=-1)
    ap.add_argument("--calib-rotate", action="store_true",
                    help=("rotated calib pairing (main.py --calib-rotate); "
                          "required for case 1, whose cached encoders only "
                          "exist under the rotated keys"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    families = (list(G0_FAMILIES) if args.deg == "all"
                else [f.strip() for f in args.deg.split(",")])
    for f in families:
        if f not in G0_FAMILIES:
            raise SystemExit(f"unknown family {f!r}")
    views = []
    for tok in args.deg_view.split(","):
        tok = tok.strip()
        views.append("all" if tok == "all" else int(tok))
    if args.smoke and "domain" in families:
        families.remove("domain")          # no Raw variant of synthetic data

    cfg = Config()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    cfg.use_fusion = False
    cfg.pool = "flatten"
    cfg.normalize = "abs_std"
    cfg.use_mae_pretrain = True
    cfg.device = args.device
    device = get_device(cfg)
    if args.smoke:
        cfg.epochs, cfg.mae_epochs = 3, 2
        cfg.inspector_epochs, cfg.n_pos_syn, cfg.n_neg_syn = 5, 400, 300
        X, y, subj = make_synthetic(cfg, n_per_class=80, seed=cfg.seed)
    else:
        X, y, subj = load_cache(cfg, mmap=True)
    X_raw = None
    if "domain" in families:
        import copy
        cfg_raw = copy.copy(cfg)
        cfg_raw.data_type = "Raw"
        X_raw, y_raw, subj_raw = load_cache(cfg_raw, mmap=True)
        if not (np.array_equal(y, y_raw) and np.array_equal(subj, subj_raw)):
            raise SystemExit("Raw cache rows misaligned with ClutterRemoved")
    subjects = sorted(set(subj.tolist()))
    folds = subjects[:args.max_folds] if args.max_folds > 0 else subjects
    cond_keys = [fam + _cond_sfx(vt) for vt in views for fam in families]
    print(f"== P4 G0 stream (frozen recipe) ==  case={args.case} "
          f"families={families} deg_views={views} B={args.budget} "
          f"s_route={args.s_route} lam={args.lam} alpha={args.alpha} "
          f"folds={len(folds)}/{len(subjects)}  device={device}", flush=True)

    rows = {}                                   # condition key -> [fold dicts]
    for ts in folds:
        if args.calib_rotate:                       # rotated pairing (c1 keys)
            cs = subjects[(subjects.index(ts) + 1) % len(subjects)]
        else:
            cs = next(s for s in subjects if s != ts)   # Run-A pairing

        res = fold_stream(cfg, X, y, subj, args.case, ts, cs, device, args,
                          families, views, X_raw=X_raw)
        for k, r in res.items():
            rows.setdefault(k, []).append(r)
        cl = res["clean"]["full"]
        worst = min((k for k in res if k != "clean"),
                    key=lambda k: res[k]["full"]["H"], default=None)
        line = (f"  subj {ts}: clean H={100*cl['H']:.1f} "
                f"cov {100*cl['cov_b']:.0f}/{100*cl['cov_n']:.0f} "
                f"({len(res)-1} conds")
        if worst:
            line += f", worst {worst} H={100*res[worst]['full']['H']:.1f}"
        print(line + ")", flush=True)

    tgt = 100 * (1 - args.alpha)

    def m(fam, grp, k):
        v = [r[grp][k] for r in rows[fam]]
        return 100 * float(np.nanmean(v))

    def cerr(fam, grp):
        v = [abs(100 * np.nanmean([r[grp][c] for r in rows[fam]]) - tgt)
             for c in ("cov_b", "cov_n")]
        return v                              # [base err, new err]

    print(f"\n===== G0 SUMMARY (case {args.case}, mean over {len(folds)} "
          f"folds, %; target cov {tgt:.0f}) =====")
    print(f"{'condition':13s} {'H_full':>7s} {'dH':>6s} {'H_hot':>7s} "
          f"{'dH_hot':>7s} {'covErr_b':>9s} {'covErr_n':>9s} {'dCovE':>6s} "
          f"{'contam5':>8s} {'contam':>7s} {'trip':>5s}")
    cl_full, cl_hot = m("clean", "full", "H"), m("clean", "hot", "H")
    cl_ce = max(cerr("clean", "full"))
    cl_c5 = 100 * float(np.nanmean([r["contam5"] for r in rows["clean"]]))
    for fam in ["clean"] + [k for k in cond_keys if k in rows]:
        Hf, Hh = m(fam, "full", "H"), m(fam, "hot", "H")
        ce_b, ce_n = cerr(fam, "full")
        dce = max(ce_b, ce_n) - cl_ce
        cont5 = 100 * float(np.nanmean([r["contam5"] for r in rows[fam]]))
        cont = 100 * float(np.nanmean([r["contam"] for r in rows[fam]]))
        trips = []
        if fam != "clean":
            if cl_full - Hf >= 10 or cl_hot - Hh >= 10:
                trips.append("H")
            if dce >= 10:
                trips.append("C")
            if cont5 - cl_c5 >= 10:
                trips.append("A")
        print(f"{fam:13s} {Hf:7.1f} {cl_full-Hf:6.1f} {Hh:7.1f} "
              f"{cl_hot-Hh:7.1f} {ce_b:9.1f} {ce_n:9.1f} {dce:6.1f} "
              f"{cont5:8.1f} {cont:7.1f} {'/'.join(trips) or '-':>5s}")
    print("trip vs clean: H = H-mean drop >=10 (full or degraded subset); "
          "C = max cov err grows >=10; A = SS39-pool (m=5/class) contam "
          "grows >=10 pts (the doc's absolute 10% bar is degenerate: the "
          "clean baseline already exceeds it on calib-lottery folds). "
          "G0 passes when >=2 families trip (SS8 G0).")


if __name__ == "__main__":
    main()
