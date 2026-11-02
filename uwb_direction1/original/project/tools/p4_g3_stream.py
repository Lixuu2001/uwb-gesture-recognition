"""P4 G3 raw-quantity scorer (c-lite pre-registration v0.2, 2026-07-19).

One frozen forward pass per (fold x cell); everything downstream -- the six
S0 systems, S1 localisation and the formal G3 verdict -- is pure numpy
post-processing (tools/p4_g3_verdict.py). Dumps RAW quantities only:

  per cell : D (n, C, V) float32 per-view distances to the frozen bank,
             M (n, V) float32 raw per-masked-view reconstruction MSE
  per fold : calib rows (D_cal / M_cal / y_cal, D_cal_reg / M_cal_reg /
             y_cal_reg), sigma matrix S (C, V), event meta (kind / ytrue /
             sev / seg / random_query)

No thresholds, no scores, no system logic here: each S0 system recomputes
its own mu/sigma/q/s_cal from these raw dumps (spec discipline).

Modes:
  --mode stream  Phase A: the G0 stream schedule per condition (severity
                 follows the per-event timetable) -- comparable to the SS44
                 dumps, feeds the engineering-regression and root-cause
                 gates. Default cells: clean,drop,occlude_vall,domain_vall.
  --mode static  Phase B: fixed severity for EVERY event (B=0 frozen
                 systems are order-free; a fixed-severity cell puts ~1150
                 samples at each grid level instead of the ~57 ramp events).
                 Cells: fam[_vN]_pXX (XX = severity*100) + clean.
                 Default: single-view grid, sevs 25,50,75,100.

Usage:
  python tools/p4_g3_stream.py --case 0 --mode stream --device cuda --mmap
  python tools/p4_g3_stream.py --case 0 --mode static --device cuda --mmap
"""
import argparse
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from config import Config                                        # noqa: E402
from p4_stream import (G0_FAMILIES, _cond_sfx,                   # noqa: E402
                       _degrade_sample, setup_fold)
from uwb_osr.data import MVDataset, load_cache, make_synthetic   # noqa: E402
from uwb_osr.engine import get_device                            # noqa: E402
from uwb_osr.inspector import extract                            # noqa: E402
from uwb_osr.pretrain import recon_error                         # noqa: E402
from uwb_osr.registration import proto_dist_tensor               # noqa: E402
from uwb_osr.stream import StreamSpec                            # noqa: E402

CHUNK = 256
STREAM_CELLS = ("clean", "drop", "occlude_vall", "domain_vall")
STATIC_SEVS = (0.25, 0.5, 0.75, 1.0)


def _raw_rows(cfg, model, enc0, dec0, bank, classes, V, Xrows, device):
    """Frozen forward -> per-view distances D (n, C, V) + raw MSEs M (n, V)."""
    n = len(Xrows)
    ds = MVDataset(Xrows, np.zeros(n, np.int64), np.arange(n),
                   list(range(len(cfg.views))), None, cfg.normalize)
    F, _, _ = extract(model, ds, cfg, device)
    D, S = proto_dist_tensor(bank, classes, F, V, device)
    M = recon_error(enc0, dec0, ds, cfg, device, per_view=True)
    return D.astype(np.float32), M.astype(np.float32), S


def _cells(args):
    if args.deg != "default":
        return [c.strip() for c in args.deg.split(",")]
    if args.mode == "stream":
        return list(STREAM_CELLS)
    cells = ["clean"]
    for fam in G0_FAMILIES:
        for vt in (0, 1, 2):
            for sv in STATIC_SEVS:
                cells.append(f"{fam}{_cond_sfx(vt)}_p{int(sv * 100)}")
    return cells


def _parse_cell(cell):
    """'fam[_vN|_vall][_pXX]' -> (fam, view_token, sev or None)."""
    sev = None
    body = cell
    if "_p" in cell:
        body, p = cell.rsplit("_p", 1)
        sev = int(p) / 100.0
    fam, _, vt = body.partition("_v")
    return fam, (vt if vt == "all" else int(vt) if vt else 0), sev


def fold_g3(cfg, X, y, subj, ci, ts, cs, device, args, cells, X_raw):
    gpfx = "p4_g3_smoke" if args.smoke else "p4_g3"
    fa = setup_fold(cfg, X, y, subj, ci, ts, cs, device, args)
    model, enc0, dec0, bank = fa.model, fa.enc0, fa.dec0, fa.bank
    nb, nr, classes, V = fa.nb, fa.nr, fa.classes, fa.V
    n_ev = len(fa.events)
    sp = fa.sp

    def ds(idx, lm):
        return MVDataset(X, y, idx, [0, 1, 2], lm, cfg.normalize)

    # per-fold calib raw dump (every S0 system recalibrates from this)
    fcal = os.path.join("logs", f"{gpfx}_c{ci}_calib_s{ts}.npz")
    if args.overwrite or not os.path.exists(fcal):
        Fc, _, yc = extract(model, ds(sp["calib_idx"], sp["label_map"]),
                            cfg, device)
        D_cal, S = proto_dist_tensor(bank, classes, Fc, V, device)
        M_cal = recon_error(enc0, dec0, ds(sp["calib_idx"], None), cfg,
                            device, per_view=True)
        Fcr, _, ycr = extract(model, ds(sp["calib_reg_idx"], sp["label_map"]),
                              cfg, device)
        D_cr = (proto_dist_tensor(bank, classes, Fcr, V, device)[0]
                if len(Fcr) else np.zeros((0, nb + nr, V), np.float32))
        M_cr = (recon_error(enc0, dec0, ds(sp["calib_reg_idx"], None), cfg,
                            device, per_view=True)
                if len(sp["calib_reg_idx"]) else np.zeros((0, V), np.float32))
        np.savez(fcal, D_cal=D_cal.astype(np.float32), M_cal=M_cal,
                 y_cal=yc, D_cal_reg=D_cr.astype(np.float32), M_cal_reg=M_cr,
                 y_cal_reg=ycr, S=S, nb=nb, nr=nr, ts=ts, cs=cs, case=ci,
                 kind=fa.kind, ytrue=fa.ytrue, sev=fa.ev_sev, seg=fa.ev_seg,
                 random_query=fa.ev_rq, sample_idx=fa.ev_idx)

    Xev_raw = (np.asarray(X_raw[fa.ev_idx], np.float32)
               if X_raw is not None else None)
    done = 0
    for cell in cells:
        f = os.path.join("logs", f"{gpfx}_c{ci}_{cell}_s{ts}.npz")
        if os.path.exists(f) and not args.overwrite:
            done += 1
            continue
        fam, vt, sev_fix = _parse_cell(cell)
        if fam != "clean" and fam not in G0_FAMILIES:
            raise SystemExit(f"unknown cell {cell!r}")
        if fam == "domain" and Xev_raw is None:
            continue
        vlist = [0, 1, 2] if vt == "all" else [int(vt)]
        sev = (np.zeros(n_ev) if fam == "clean"
               else np.full(n_ev, sev_fix) if sev_fix is not None
               else fa.ev_sev)                      # stream timetable
        spec = StreamSpec(seed=fa.seed, budget_per_100=args.budget,
                          deg_type=fam if fam != "clean" else "noise",
                          deg_view=vlist[0])
        D = np.zeros((n_ev, nb + nr, V), np.float32)
        M = np.zeros((n_ev, V), np.float32)
        S = None
        for a in range(0, n_ev, CHUNK):
            rows = np.arange(a, min(a + CHUNK, n_ev))
            if fam == "clean":
                Xd = fa.Xev[rows]
            else:
                Xd = np.stack([
                    fa.Xev[i] if sev[i] == 0 else _degrade_sample(
                        fa.Xev[i],
                        None if Xev_raw is None else Xev_raw[i],
                        fam, vlist, float(sev[i]), fa.seed, int(i), spec)
                    for i in rows])
            D[rows], M[rows], S = _raw_rows(cfg, model, enc0, dec0, bank,
                                            classes, V, Xd, device)
            del Xd
        np.savez(f, D=D, M=M, S=S, cell=cell, sev=sev.astype(np.float32),
                 nb=nb, nr=nr, ts=ts, case=ci)
        done += 1
    del model, enc0, dec0, bank, fa
    if device == "cuda":
        torch.cuda.empty_cache()
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--mode", choices=["stream", "static"], default="stream")
    ap.add_argument("--deg", default="default",
                    help="comma cells (fam[_vN|_vall][_pXX]) or 'default'")
    ap.add_argument("--budget", type=int, default=5)   # schedule tag only
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--s-route", type=float, default=1.5)
    ap.add_argument("--deg-view", default="0")         # setup_fold compat
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mmap", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--calib-rotate", action="store_true")
    ap.add_argument("--max-folds", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cells = _cells(args)

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
    if any(_parse_cell(c)[0] == "domain" for c in cells) and not args.smoke:
        import copy
        cfg_raw = copy.copy(cfg)
        cfg_raw.data_type = "Raw"
        X_raw, y_raw, subj_raw = load_cache(cfg_raw, mmap=True)
        if not (np.array_equal(y, y_raw) and np.array_equal(subj, subj_raw)):
            raise SystemExit("Raw cache rows misaligned")
    subjects = sorted(set(subj.tolist()))
    folds = subjects[:args.max_folds] if args.max_folds > 0 else subjects
    print(f"== P4 G3 raw scorer ==  case={args.case} mode={args.mode} "
          f"cells={len(cells)} folds={len(folds)}/{len(subjects)} "
          f"device={device}", flush=True)
    for ts in folds:
        if args.calib_rotate:
            cs = subjects[(subjects.index(ts) + 1) % len(subjects)]
        else:
            cs = next(s for s in subjects if s != ts)
        n = fold_g3(cfg, X, y, subj, args.case, ts, cs, device, args,
                    cells, X_raw)
        print(f"  subj {ts}: {n}/{len(cells)} cells", flush=True)


if __name__ == "__main__":
    main()
