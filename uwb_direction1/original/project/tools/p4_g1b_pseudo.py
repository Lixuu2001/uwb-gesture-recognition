"""G1b calibration-side pseudo-registration (paper-4 design SS5.3, main method).

The routing-scale question G1a's median ratio could not answer -- where the
argmin boundary between a 600-sample base head and a 5-shot head actually
sits -- is rehearsed here where labels are legal. Per fold, each base class
g in turn plays "new":

  * g's own base head is dropped from the bank;
  * the OTHER five base checkers are retrained on the frozen features with g
    deleted from their real negatives, mixup pool and all_real -- Stage-2 is
    one-vs-rest, so without this retrain every remaining head has explicitly
    learned to reject g and the old-head swallowing is underestimated
    (design review round 2, blocking item 1);
  * a pseudo-new head is trained for g from K subject-balanced support shots
    drawn from the training users (same source as deployed registration),
    negatives likewise exclude g;
  * the 6-column mini-bank routes the CALIBRATION users' samples over the
    pre-registered s grid. Pinned metrics (review round 3): base acc = the
    five remaining classes, new acc = g, base-drop reference = the SAME
    retrained bank at s=1, macro over (held-out class x calib subject).

s_hat = H-max on the grid subject to macro base drop <= 2pts. The fold's
s_hat is then applied to the fold's REAL register bank (rot2 dump) next to
s=1 / s=1.5 / oracle -- the G1 gate readout. Known residual bias: the
encoder saw g (expected conservative, verified on the real vocabularies).

Cost: 6 leave-outs x (5 base + 1 pseudo) x 3 views = 108 checker trainings
per fold; run with --limit-g 1 first as the mandated timing gate.

Usage:
  python tools/p4_g1b_pseudo.py --case 0 --ts 1 --limit-g 1 --device cuda
  python tools/p4_g1b_pseudo.py --case 0 --ts 1 --device cuda
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, REG_TEST_CASES, VIEW_ORDER            # noqa: E402
from main import _enc_cache_path                                 # noqa: E402
from uwb_osr.data import load_cache, MVDataset, register_split_conf  # noqa: E402
from uwb_osr.engine import get_device                            # noqa: E402
from uwb_osr.inspector import extract, train_inspector           # noqa: E402
from uwb_osr.model import MultiViewOSRNet                        # noqa: E402
from uwb_osr.registration import _balanced_shots, proto_dist_matrix  # noqa: E402

S_GRID = np.round(np.arange(1.0, 3.0 + 1e-9, 0.1), 2)
BASE_DROP = 0.02        # pinned: vs the SAME retrained bank at s=1


def _route_at(Rb, yb, Rr, yr, nb, s):
    def acc(R, yt):
        Rs = R.copy(); Rs[:, :nb] *= s
        return float((Rs.argmin(1) == yt).mean())
    ab, an = acc(Rb, yb), acc(Rr, yr)
    return ab, an, 2 * ab * an / (ab + an + 1e-9)


def pseudo_register(g, Ft, yt, tsub, cfg, device, V, args, t_heads):
    """One leave-out: retrain the 5 remaining base heads without g in any
    negative pool, train g's pseudo-new head from balanced support shots.
    Returns (bank keyed 0..5 with pseudo at column 5, remaining classes)."""
    yt_np = np.asarray(yt)
    keep_mask = yt_np != g
    Ft_neg, yt_neg = Ft[keep_mask], torch.as_tensor(yt_np[keep_mask])
    remaining = [k for k in range(int(yt_np.max()) + 1) if k != g]

    rng = np.random.RandomState(cfg.seed + 97 * g + int(args.ts))
    rows_g = np.where(yt_np == g)[0]
    shots = _balanced_shots(rows_g, np.asarray(tsub)[rows_g], args.n_shot, rng)
    assert len(shots) == args.n_shot and set(shots) <= set(rows_g.tolist())

    torch.manual_seed(cfg.seed + 7919 * g + 13 * int(args.ts))
    bank = {}
    for col, k in enumerate(remaining):        # 5 retrained base heads
        pos = Ft[yt_np == k]
        neg = Ft[keep_mask & (yt_np != k)]
        assert not (yt_np[keep_mask & (yt_np != k)] == g).any()
        for v in range(V):
            t0 = time.perf_counter()
            bank[(col, v)] = train_inspector(pos[:, v], neg[:, v], cfg, device,
                                             all_real=Ft_neg[:, v],
                                             all_labels=yt_neg)
            t_heads.append(("base", time.perf_counter() - t0))
    for v in range(V):                          # pseudo-new head for g
        t0 = time.perf_counter()
        bank[(len(remaining), v)] = train_inspector(
            Ft[shots][:, v], Ft_neg[:, v], cfg, device,
            all_real=Ft_neg[:, v], all_labels=yt_neg)
        t_heads.append(("pseudo", time.perf_counter() - t0))
    # deployed degenerate-sigma floor, from the retrained base heads
    for v in range(V):
        floor = float(np.mean([bank[(c, v)][2] for c in range(len(remaining))]))
        insp, phi, s = bank[(len(remaining), v)]
        if s < 1e-3:
            bank[(len(remaining), v)] = (insp, phi, floor)
    return bank, remaining, shots


def fold_g1b(cfg, X, y, subj, ci, ts, device, args):
    rc = REG_TEST_CASES[ci]
    view_idx = [VIEW_ORDER.index(v) for v in cfg.views]
    subjects = sorted(set(subj.tolist()))
    i0 = subjects.index(ts)
    order = [subjects[(i0 + j) % len(subjects)] for j in range(1, len(subjects))]
    cs_list = order[:2]
    cs_tag = "+".join(str(s) for s in cs_list)

    sp = register_split_conf(cfg, y, subj, rc["base"], rc["register"],
                             rc["unknown"], ts, cs_list, args.n_shot, cfg.seed,
                             subject_balanced=True)
    nb = sp["n_base"]

    path = _enc_cache_path(cfg, SimpleNamespace(mae_no_init=False), ci, ts, cs_tag)
    if not os.path.exists(path):
        raise SystemExit(f"enc-cache MISS ({path}); G1b must run on the "
                         f"deployed encoders -- refusing to retrain.")
    ck = torch.load(path, map_location=device)
    model = MultiViewOSRNet(cfg, nb).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    def ds(idx, lm):
        return MVDataset(X, y, idx, view_idx, lm, cfg.normalize)
    Ft, _, yt = extract(model, ds(sp["train_idx"], sp["label_map"]), cfg, device)
    Fc, _, yc = extract(model, ds(sp["calib_idx"], sp["label_map"]), cfg, device)
    V = Ft.size(1)
    tsub = np.asarray(subj)[sp["train_idx"]]
    csub = np.asarray(subj)[sp["calib_idx"]]
    assert not set(tsub.tolist()) & set(csub.tolist()), "train/calib overlap"

    gs = list(range(nb))[:args.limit_g] if args.limit_g > 0 else list(range(nb))
    print(f"fold c{ci} ts={ts} cs={cs_tag}  enc-cache HIT  leave-outs={gs}")

    t_heads = []
    # cells[(g, cs_subj)] = (nb_grid arrays) acc_base(s), acc_new(s)
    ab_cells, an_cells = [], []
    for g in gs:
        bank, remaining, shots = pseudo_register(g, Ft, yt, tsub, cfg, device,
                                                 V, args, t_heads)
        # calib labels -> mini-bank columns (5 remaining base + pseudo at 5)
        col_of = {k: i for i, k in enumerate(remaining)}
        R = proto_dist_matrix(bank, list(range(len(remaining) + 1)), Fc, V,
                              device)
        yc_np = np.asarray(yc)
        for cs0 in cs_list:
            rows = csub == cs0
            is_g = rows & (yc_np == g)
            is_b = rows & (yc_np != g)
            ytrue = np.where(yc_np[rows] == g, len(remaining),
                             [col_of.get(int(l), -1) for l in yc_np[rows]])
            ab_s, an_s = [], []
            for s in S_GRID:
                Rs = R[rows].copy()
                Rs[:, :len(remaining)] *= s
                pred = Rs.argmin(1)
                ab_s.append(float((pred[yc_np[rows] != g]
                                   == ytrue[yc_np[rows] != g]).mean()))
                an_s.append(float((pred[yc_np[rows] == g]
                                   == ytrue[yc_np[rows] == g]).mean()))
            ab_cells.append(ab_s)
            an_cells.append(an_s)
        print(f"  g={g}: support rows from subjects "
              f"{sorted(np.asarray(tsub)[shots].tolist())}, "
              f"{len(ab_cells)} cells so far", flush=True)

    ab = np.asarray(ab_cells).mean(0)          # macro over (g, calib subject)
    an = np.asarray(an_cells).mean(0)
    H = 2 * ab * an / (ab + an + 1e-9)
    feas = ab >= ab[0] - BASE_DROP             # pinned: same-bank s=1 reference
    s_hat = float(S_GRID[np.argmax(np.where(feas, H, -1))])

    print(f"\n  pseudo grid (macro over {len(ab_cells)} cells): "
          f"s=1 acc_b {100*ab[0]:.1f} acc_n {100*an[0]:.1f} H {100*H[0]:.1f}")
    for i, s in enumerate(S_GRID):
        mark = " <- s_hat" if s == s_hat else ""
        if s in (1.0, 1.5, 2.0, 2.5, 3.0) or s == s_hat:
            print(f"    s={s:4.2f}  acc_b {100*ab[i]:5.1f}  acc_n {100*an[i]:5.1f}"
                  f"  H {100*H[i]:5.1f}  {'feas' if feas[i] else 'INFEAS'}{mark}")

    # transfer to the fold's REAL register bank (the G1 gate readout)
    dump = f"logs/n3_scores_bal_b2e_rot2_c{ci}_s{ts}.npz"
    npz = np.load(dump)
    Rb, yb = npz["R_base"], npz["y_base"]
    Rr, yr = npz["R_reg"], npz["y_reg"]
    ab1 = _route_at(Rb, yb, Rr, yr, nb, 1.0)[0]
    cand = [(s, *_route_at(Rb, yb, Rr, yr, nb, s)) for s in S_GRID]
    feas_d = [t for t in cand if t[1] >= ab1 - BASE_DROP]
    s_star = max(feas_d, key=lambda t: t[3])[0] if feas_d else 1.0
    print(f"\n  transfer to real bank ({os.path.basename(dump)}):")
    for name, s in [("s=1", 1.0), ("s=1.5", 1.5), ("G1b s_hat", s_hat),
                    ("oracle-H", s_star)]:
        a, n, h = _route_at(Rb, yb, Rr, yr, nb, s)
        print(f"    {name:10s} s={s:4.2f}  acc_b {100*a:5.1f}  "
              f"acc_n {100*n:5.1f}  H {100*h:5.1f}")

    tb = [t for k, t in t_heads if k == "base"]
    tp = [t for k, t in t_heads if k == "pseudo"]
    n_lo = len(gs)
    per_lo = (np.sum([t for _, t in t_heads]) / n_lo)
    print(f"\n  timing: base head {np.mean(tb):.1f}s, pseudo head "
          f"{np.mean(tp):.1f}s; {per_lo/60:.1f} min per leave-out -> "
          f"{6*per_lo/60:.0f} min/fold, {8*6*per_lo/3600:.1f} h for 8 folds")

    out = f"logs/p4_g1b_c{ci}_s{ts}.npz"
    np.savez(out, s_grid=S_GRID, ab=ab, an=an, H=H, s_hat=s_hat,
             s_oracle=s_star, cells_ab=np.asarray(ab_cells),
             cells_an=np.asarray(an_cells), leave_outs=np.asarray(gs),
             ts=ts, case=ci, sec_base=float(np.mean(tb)),
             sec_pseudo=float(np.mean(tp)))
    print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--ts", type=int, default=-1)
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--limit-g", type=int, default=0,
                    help="leave-out classes (0 = all 6; 1 = timing gate)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = Config()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    cfg.use_fusion = False
    cfg.pool = "flatten"
    cfg.normalize = "abs_std"
    cfg.use_mae_pretrain = True
    cfg.device = args.device
    device = get_device(cfg)
    X, y, subj = load_cache(cfg, mmap=True)
    subjects = sorted(set(subj.tolist()))
    args.ts = subjects[0] if args.ts < 0 else args.ts

    print(f"== G1b calibration-side pseudo-registration ==  case={args.case} "
          f"ts={args.ts} K={args.n_shot} "
          f"limit_g={args.limit_g or 'all'} device={device}")
    fold_g1b(cfg, X, y, subj, args.case, args.ts, device, args)


if __name__ == "__main__":
    main()
