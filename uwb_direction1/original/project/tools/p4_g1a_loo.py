"""G1a estimator audit: checker-level true support-LOO (paper-4 design SS5.3).

The median-ratio routing-scale estimator needs the register classes' TRUE
leave-one-out scores: for each support sample i, retrain that class's
per-view few-shot checkers on the remaining K-1 shots (same negatives, same
augmentation config as the deployed method-A registration) and score the
held-out shot with the fresh heads. Training on all K and only recomputing
the prototype leaks the held-out shot through the MLP weights; that cheap
variant is computed here too but labelled frozen-head LOO and reported as a
diagnostic only.

Everything else comes from the frozen rot2 artifacts so the audit measures
the deployed configuration: the fold encoder is REQUIRED to be an enc-cache
hit (never retrained here), m_base and the oracle-s grid come from the
n3_scores_bal_b2e_rot2 dump of the same fold.

Per fold this is 3 register classes x K shots x 3 views checker trainings
(45 at K=5) plus 9 frozen-head trainings; the first run doubles as the
timing gate the design doc requires before scheduling all 16 folds.

Usage:
  python tools/p4_g1a_loo.py --case 0 --ts 1 --device cuda          # 1-fold
  python tools/p4_g1a_loo.py --case 0 --ts 1 --limit 1 --device cuda  # quick
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

SHRINK_N0 = 10          # pre-registered: rho = n / (n + SHRINK_N0)
CLIP_LO, CLIP_HI = 1.0, 3.0


def _loo_seed(cfg, ts, c, j):
    """Stable per-(class, left-out shot) seed so every retrain is replayable."""
    return int(cfg.seed) + 7919 * (int(c) * 100 + int(j)) + 13 * int(ts)


def _shat(m_new, m_base, n_valid):
    """Pre-registered shrink-and-clip ratio estimator (design SS5.3 G1a)."""
    if not np.isfinite(m_new) or not np.isfinite(m_base) or m_base <= 0:
        return float("nan")
    rho = n_valid / (n_valid + SHRINK_N0)
    return float(np.clip(np.exp(rho * np.log(m_new / m_base)), CLIP_LO, CLIP_HI))


def _route_at(Rb, yb, Rr, yr, nb, s):
    """Routing accuracies of the dumped bank with base columns scaled by s."""
    def acc(R, ytrue):
        Rs = R.copy()
        Rs[:, :nb] *= s
        return float((Rs.argmin(1) == ytrue).mean())
    ab, an = acc(Rb, yb), acc(Rr, yr)
    return ab, an, 2 * ab * an / (ab + an + 1e-9)


@torch.no_grad()
def _score_head(head_v, feat_row, device):
    """r for one sample against one class's per-view heads (mirrors
    _class_dists' d/sigma aggregation, incl. the 1e-6 guard)."""
    ds, ss = [], []
    for v, (insp, phi, sigma) in head_v.items():
        lat = insp(feat_row[None, v].to(device))
        ds.append(float(torch.norm(lat - phi.to(device), dim=1)))
        ss.append(sigma)
    return float(np.mean(ds) / (np.mean(ss) + 1e-6))


def _train_class_head(pos_rows, Ft, yt_t, cfg, device, V, sigma_floor):
    """Few-shot heads for one class, one per view: exactly the deployed
    method-A call (negatives = all base train features, same augmentation),
    plus the deployed degenerate-sigma floor."""
    head, floored = {}, 0
    for v in range(V):
        insp, phi, sig = train_inspector(pos_rows[:, v], Ft[:, v], cfg, device,
                                         all_real=Ft[:, v], all_labels=yt_t)
        if sig < 1e-3:
            sig, floored = float(sigma_floor[v]), floored + 1
        head[v] = (insp, phi, sig)
    return head, floored


def fold_audit(cfg, X, y, subj, ci, ts, device, args):
    rc = REG_TEST_CASES[ci]
    view_idx = [VIEW_ORDER.index(v) for v in cfg.views]
    subjects = sorted(set(subj.tolist()))
    i0 = subjects.index(ts)
    order = [subjects[(i0 + j) % len(subjects)] for j in range(1, len(subjects))]
    cs_list = order[:2]                       # rot2 pairing: 2 pooled calib
    cs_tag = "+".join(str(s) for s in cs_list)

    sp = register_split_conf(cfg, y, subj, rc["base"], rc["register"],
                             rc["unknown"], ts, cs_list, args.n_shot, cfg.seed,
                             subject_balanced=True)
    nb, nr = sp["n_base"], sp["n_register"]

    path = _enc_cache_path(cfg, SimpleNamespace(mae_no_init=False), ci, ts, cs_tag)
    if not os.path.exists(path):
        raise SystemExit(f"enc-cache MISS ({path}); the audit must run on the "
                         f"deployed encoders -- refusing to retrain.")
    ck = torch.load(path, map_location=device)
    model = MultiViewOSRNet(cfg, nb).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    dump = f"logs/n3_scores_bal_b2e_rot2_c{ci}_s{ts}.npz"
    npz = np.load(dump)
    assert int(npz["n_shot"]) == args.n_shot and int(npz["n_base"]) == nb \
        and int(npz["n_reg"]) == nr, f"{dump} disagrees with the fold config"
    R_cal, y_cal = npz["R_cal"], npz["y_cal"]
    m_base = float(np.median(R_cal[np.arange(len(R_cal)), y_cal]))
    own_reg = npz["R_cal_reg"][np.arange(len(npz["R_cal_reg"])), npz["y_cal_reg"]]
    sigma_floor = npz["sigma_mat"][:nb].mean(0)          # deployed base sigmas

    def ds(idx, lm):
        return MVDataset(X, y, idx, view_idx, lm, cfg.normalize)
    Ft, _, yt = extract(model, ds(sp["train_idx"], sp["label_map"]), cfg, device)
    Fs, _, ys = extract(model, ds(sp["support_idx"], sp["label_map"]), cfg, device)
    yt_t = torch.as_tensor(yt)
    V = Ft.size(1)
    sup_subj = np.asarray(subj)[sp["support_idx"]]

    print(f"fold c{ci} ts={ts} cs={cs_tag}  enc-cache HIT  "
          f"support subjects per class: "
          f"{[sorted(sup_subj[ys == c].tolist()) for c in range(nb, nb + nr)]}")

    r_true, r_frozen, cls_of, t_heads = [], [], [], []
    n_floor = 0
    replay = None                              # (c, j, seed, r) determinism probe
    for c in range(nb, nb + nr):
        rows_c = np.where(ys == c)[0]
        K = len(rows_c)
        assert K == args.n_shot, f"class {c}: {K} support shots"
        take = rows_c if args.limit <= 0 else rows_c[:args.limit]

        # -- true checker-level LOO: retrain on K-1, score the held-out shot
        for j, gi in enumerate(take):
            keep = np.array([g for g in rows_c if g != gi])
            torch.manual_seed(_loo_seed(cfg, ts, c, j))
            t0 = time.perf_counter()
            head, fl = _train_class_head(Fs[keep], Ft, yt_t, cfg, device, V,
                                         sigma_floor)
            t_heads.append((time.perf_counter() - t0) / V)
            n_floor += fl
            for v in range(V):                 # leakage guard, per view
                pos = Fs[keep][:, v]
                assert len(pos) == K - 1
                assert not (pos == Fs[gi, v]).all(1).any(), \
                    f"left-out shot present in training pos (c={c} j={j} v={v})"
            r = _score_head(head, Fs[gi], device)
            r_true.append(r)
            cls_of.append(c)
            if replay is None:
                replay = (c, j, gi, keep, r)

        # -- frozen-head LOO (diagnostic only): one head on all K, prototype
        #    recomputed without shot i -- the held-out shot still shaped the
        #    MLP weights, which is exactly the leak the audit must not use.
        torch.manual_seed(_loo_seed(cfg, ts, c, 99))
        head_full, _ = _train_class_head(Fs[rows_c], Ft, yt_t, cfg, device, V,
                                         sigma_floor)
        with torch.no_grad():
            for gi in take:
                keep = np.array([g for g in rows_c if g != gi])
                dsum, ssum = [], []
                for v in range(V):
                    insp = head_full[v][0]
                    lat_keep = insp(Fs[keep][:, v].to(device))
                    phi = lat_keep.mean(0)
                    sig = float(torch.sqrt(((lat_keep - phi) ** 2)
                                           .sum(1).mean()))
                    if sig < 1e-3:
                        sig = float(sigma_floor[v])
                    lat_i = insp(Fs[gi][None, v].to(device))
                    dsum.append(float(torch.norm(lat_i - phi, dim=1)))
                    ssum.append(sig)
                r_frozen.append(float(np.mean(dsum) / (np.mean(ssum) + 1e-6)))

    # -- determinism probe: replay the first LOO with its recorded seed
    c, j, gi, keep, r_first = replay
    torch.manual_seed(_loo_seed(cfg, ts, c, j))
    head, _ = _train_class_head(Fs[keep], Ft, yt_t, cfg, device, V, sigma_floor)
    r_replay = _score_head(head, Fs[gi], device)
    det_ok = abs(r_replay - r_first) <= 1e-4 * max(1.0, abs(r_first))

    r_true, r_frozen = np.asarray(r_true), np.asarray(r_frozen)
    valid = np.isfinite(r_true)
    est = {
        "true": _shat(float(np.median(r_true[valid])), m_base, int(valid.sum())),
        "frozen": _shat(float(np.median(r_frozen)), m_base, len(r_frozen)),
        # proxy = the R_cal_reg stand-in from the 07-10 review, evaluated with
        # the same n so the three rows differ only in where m_new comes from.
        "proxy": _shat(float(np.median(own_reg)), m_base, int(valid.sum())),
    }

    # -- oracle on the dumped bank: H-max on the pre-registered grid subject
    #    to base drop <= 2pts vs the SAME bank at s=1 (design SS5.3)
    Rb, yb = npz["R_base"], npz["y_base"]
    Rr, yr = npz["R_reg"], npz["y_reg"]
    grid = np.round(np.arange(CLIP_LO, CLIP_HI + 1e-9, 0.1), 2)
    ab1 = _route_at(Rb, yb, Rr, yr, nb, 1.0)[0]
    allowed = [(s, *_route_at(Rb, yb, Rr, yr, nb, s)) for s in grid]
    feas = [t for t in allowed if t[1] >= ab1 - 0.02]
    s_star = max(feas, key=lambda t: t[3])[0] if feas else 1.0

    print(f"\n  m_base(dump R_cal own)={m_base:.1f}  "
          f"median LOO: true={np.median(r_true[valid]):.1f} "
          f"frozen={np.median(r_frozen):.1f} proxy={np.median(own_reg):.1f}  "
          f"valid={int(valid.sum())}/{len(r_true)}  sigma-floored={n_floor}")
    print(f"  determinism replay: {'OK' if det_ok else 'FAIL'} "
          f"(r {r_first:.4f} -> {r_replay:.4f})")
    hdr = f"  {'variant':10s} {'s':>6s} {'acc_b':>7s} {'acc_n':>7s} {'H':>7s}"
    print("\n" + hdr)
    for name, s in [("s=1", 1.0), ("s=1.5", 1.5), ("LOO-true", est["true"]),
                    ("frozen", est["frozen"]), ("proxy", est["proxy"]),
                    ("oracle-H", s_star)]:
        if not np.isfinite(s):
            print(f"  {name:10s} {'nan':>6s}")
            continue
        ab, an, h = _route_at(Rb, yb, Rr, yr, nb, s)
        print(f"  {name:10s} {s:6.2f} {100*ab:7.1f} {100*an:7.1f} {100*h:7.1f}")

    mean_t = float(np.mean(t_heads))
    n_loo = len(r_true)
    print(f"\n  timing: {mean_t:.1f}s per head training "
          f"({n_loo * V} LOO heads = {n_loo * V * mean_t / 60:.1f} min this run)")
    print(f"  projection: G1a 16 folds x 45 heads = "
          f"{16 * 45 * mean_t / 3600:.1f} h;  G1b 8 folds x 108 heads "
          f">= {8 * 108 * mean_t / 3600:.1f} h (base heads train on far more "
          f"positives -- treat as a lower bound, time one before scheduling)")

    out = f"logs/p4_g1a_c{ci}_s{ts}.npz"
    np.savez(out, r_true=r_true, r_frozen=r_frozen, cls_of=np.asarray(cls_of),
             m_base=m_base, s_true=est["true"], s_frozen=est["frozen"],
             s_proxy=est["proxy"], s_oracle=s_star, ts=ts, case=ci,
             support_idx=sp["support_idx"], sec_per_head=mean_t,
             sigma_floored=n_floor, det_ok=det_ok)
    print(f"  saved {out}")
    return est, s_star, det_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--ts", type=int, default=-1,
                    help="test subject (default: first in roster)")
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0,
                    help="LOO shots per class (0 = all; use 1 for a quick pass)")
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
    ts = subjects[0] if args.ts < 0 else args.ts

    print(f"== G1a checker-level support-LOO audit ==  case={args.case} ts={ts} "
          f"K={args.n_shot} limit={args.limit or 'all'} device={device}")
    fold_audit(cfg, X, y, subj, args.case, ts, device, args)


if __name__ == "__main__":
    main()
