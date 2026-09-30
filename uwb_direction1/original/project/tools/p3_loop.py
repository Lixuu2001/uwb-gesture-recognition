"""Conformal-gated self-personalization loop (paper-3 deployment simulation).

The deployment story this measures: a gesture is registered CROSS-PERSON
(vendor ships K=5 balanced supports, acc_new ~75); the end user then simply
uses the system. Every stream sample whose conformal prediction set is a
SINGLETON on a new-class column is silently collected as a pseudo-labelled
personal example of that class; once m_adopt are collected, that class's
few-shot head is retrained on the user's own samples -- the measured
personalization regime (RESULTS SS34: acc_new 91.1). The conformal gate is
what makes silent collection defensible: singleton new-class predictions run
at ~73-92% precision (SS36), and the contamination that does slip through is
exactly what this simulation quantifies.

Protocol per LOSO fold (case 0, non-rotated pairing = the Run-A cached
encoders, --enc-cache hits):
  * cold bank  = the frozen recipe (B2-init, balanced K=5, method A);
  * thresholds = frozen conformal recipe (fuse-base lam=1: e_recon fused into
    the BASE columns only, calibrated on the held-out calib subject);
  * the test subject's stream = shuffled(base-test + half of reg-test +
    unknown-test); the other half of reg-test is the held-out EVAL set;
  * v1 simplification: collection scans the stream with the COLD bank and
    adoption happens at the end of the collection window (no mid-stream
    re-scoring); adoption latency is still reported in stream positions.
Reported: adoption rate/latency, pool contamination, eval acc_new cold ->
warm -> oracle-personal ceiling (true-label supports, the SS34 regime), and
unknown rejection cold vs warm (base heads untouched -> should hold).

Usage:
  python tools/p3_loop.py --case 0 --device cuda --mmap        # real (cache hits)
  python tools/p3_loop.py --smoke --device cpu                 # synthetic smoke
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
from uwb_osr.inspector import extract, train_inspector           # noqa: E402
from uwb_osr.model import MultiViewOSRNet, SharedDualPathEncoder  # noqa: E402
from uwb_osr.pretrain import (pretrain_encoder, recon_error,      # noqa: E402
                              MaskedViewDecoder)
from uwb_osr.registration import (_register_method_a,            # noqa: E402
                                  proto_dist_matrix)


def _q(scores, alpha):
    scores = np.asarray(scores, float)
    n = len(scores)
    if n == 0:
        return np.inf
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if rank > n else float(np.sort(scores)[rank - 1])


def _get_encoder(cfg, args, ci, ts, cs, train_ds, device):
    """Load the fold encoder (+ Stage-0 pair for e_recon) from the enc-cache,
    training and caching it when absent (smoke always trains)."""
    path = _enc_cache_path(cfg, SimpleNamespace(mae_no_init=False), ci, ts, cs)
    if not args.smoke and os.path.exists(path):
        ck = torch.load(path, map_location=device)
        model = MultiViewOSRNet(cfg, len(REG_TEST_CASES[ci]["base"])).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        enc0 = SharedDualPathEncoder(cfg).to(device)
        enc0.load_state_dict(ck["mae_enc"])
        dec0 = MaskedViewDecoder(cfg).to(device)
        dec0.load_state_dict(ck["mae_dec"])
        return model, enc0, dec0, True
    mae_state, enc0, dec0 = pretrain_encoder(cfg, train_ds, device, False)
    model = train_one_case(cfg, train_ds, len(REG_TEST_CASES[ci]["base"]),
                           device, verbose=False, desc=f"loop c{ci} s{ts} ",
                           mae_state=mae_state)
    return model, enc0, dec0, False


def fold_loop(cfg, X, y, subj, ci, ts, cs, device, args):
    rc = REG_TEST_CASES[ci]
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
    model, enc0, dec0, hit = _get_encoder(cfg, args, ci, ts, cs, train_ds,
                                          device)

    support_ds = ds(sp["support_idx"], sp["label_map"])
    bank, V = _register_method_a(model, train_ds, support_ds, cfg, device, nb)
    Ft, _, yt = extract(model, train_ds, cfg, device)   # retrain negatives
    yt_t = torch.as_tensor(yt)
    Fs, _, ys = extract(model, support_ds, cfg, device)  # original cold shots
    Fc, _, yc = extract(model, ds(sp["calib_idx"], sp["label_map"]), cfg, device)
    Fb, _, yb = extract(model, ds(sp["test_base_idx"], sp["label_map"]), cfg, device)
    Fr, _, yr = extract(model, ds(sp["test_reg_idx"], sp["label_map"]), cfg, device)
    Fu, _, _ = extract(model, ds(sp["test_unknown_idx"], None), cfg, device)

    E = {k: recon_error(enc0, dec0, ds(sp[i], None), cfg, device)
         for k, i in [("cal", "calib_idx"), ("b", "test_base_idx"),
                      ("r", "test_reg_idx"), ("u", "test_unknown_idx")]}

    # frozen conformal recipe: z(r) everywhere + lam*z(e_recon) on base columns
    R_cal = proto_dist_matrix(bank, classes, Fc, V, device)
    own = R_cal[np.arange(len(R_cal)), yc]
    mu_r, sd_r = float(own.mean()), float(own.std() + 1e-8)
    mu_e, sd_e = float(E["cal"].mean()), float(E["cal"].std() + 1e-8)
    q = np.full(nb + nr, _q((own - mu_r) / sd_r, args.alpha))
    q[:nb] = _q((own - mu_r) / sd_r
                + args.lam * (E["cal"] - mu_e) / sd_e, args.alpha)

    def score(bk, F, e):
        S = (proto_dist_matrix(bk, classes, F, V, device) - mu_r) / sd_r
        S[:, :nb] += args.lam * ((e - mu_e) / sd_e)[:, None]
        return S

    # stream/eval split of the test subject's register samples (per class)
    rng = np.random.RandomState(cfg.seed + 1000 + ts)
    stream_r, eval_r = [], []
    for c in range(nb, nb + nr):
        rows = np.where(yr == c)[0]
        rng.shuffle(rows)
        h = len(rows) // 2
        stream_r += rows[:h].tolist()
        eval_r += rows[h:].tolist()
    eval_r = np.array(eval_r, int)

    items = ([("b", i) for i in range(len(Fb))]
             + [("r", i) for i in stream_r]
             + [("u", i) for i in range(len(Fu))])
    rng.shuffle(items)

    S_cold = {"b": score(bank, Fb, E["b"]), "r": score(bank, Fr, E["r"]),
              "u": score(bank, Fu, E["u"])}
    Fmap = {"b": Fb, "r": Fr, "u": Fu}

    pools, adopt_at, contam = {c: [] for c in range(nb, nb + nr)}, {}, 0
    for t, (kind, i) in enumerate(items):
        row = S_cold[kind][i] <= q
        if row.sum() == 1:
            c = int(np.argmax(row))
            if c >= nb and len(pools[c]) < args.m_adopt:
                pools[c].append((kind, i))
                truth = (kind == "r" and int(yr[i]) == c)
                contam += 0 if truth else 1
                if len(pools[c]) == args.m_adopt:
                    adopt_at[c] = t + 1

    def retrain(rows_feats):
        head = {}
        for v in range(V):
            head[v] = train_inspector(rows_feats[:, v], Ft[:, v], cfg, device,
                                      all_real=Ft[:, v], all_labels=yt_t)
        return head

    # --blend keeps the original cross-person shots in the retrain support.
    # v1 (replace) showed why this matters: the conformal gate only passes
    # samples the COLD head is already sure about, so a personal-only pool is
    # biased toward the user's easy renderings -- clean (contam ~6%) but
    # narrow, and swapping the whole head onto it can LOSE the hard tail the
    # cross-person shots covered (fold regressions 95->79). Blending keeps the
    # cross-person spread and adds the personal mode.
    warm = dict(bank)
    n_adopted = 0
    for c, pool in pools.items():
        if len(pool) >= args.m_adopt:
            feats = torch.stack([Fmap[k][i] for k, i in pool])
            if args.blend:
                feats = torch.cat([Fs[ys == c], feats])
            for v, hv in retrain(feats).items():
                warm[(c, v)] = hv
            n_adopted += 1
    oracle = dict(bank)
    for c in range(nb, nb + nr):
        rows = [i for i in stream_r if int(yr[i]) == c][:args.m_adopt]
        if rows:
            feats = Fr[rows]
            if args.blend:
                feats = torch.cat([Fs[ys == c], feats])
            for v, hv in retrain(feats).items():
                oracle[(c, v)] = hv

    def acc_eval(bk):
        R = proto_dist_matrix(bk, classes, Fr[eval_r], V, device)
        return float((R.argmin(1) == yr[eval_r]).mean())

    def rej_unk(bk):
        return float((score(bk, Fu, E["u"]) > q).all(1).mean())

    n_collected = sum(len(p) for p in pools.values())
    return dict(
        cache="hit" if hit else "MISS",
        acc_cold=acc_eval(bank), acc_warm=acc_eval(warm),
        acc_oracle=acc_eval(oracle),
        rej_cold=rej_unk(bank), rej_warm=rej_unk(warm),
        adopted=n_adopted, collected=n_collected,
        contam=(contam / n_collected if n_collected else 0.0),
        latency=(float(np.mean(list(adopt_at.values())))
                 if adopt_at else float("nan")),
        stream=len(items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--m-adopt", type=int, default=5,
                    help="personal samples per class before the head swaps")
    ap.add_argument("--blend", action="store_true",
                    help=("retrain support = original cross-person shots + "
                          "collected personal pool (v2; guards against the "
                          "gate's easy-sample selection bias)"))
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mmap", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-folds", type=int, default=-1)
    args = ap.parse_args()

    cfg = Config()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)      # inspector init/aug reproducibility
    cfg.use_fusion = False
    cfg.pool = "flatten"
    cfg.normalize = "abs_std"
    cfg.use_mae_pretrain = True
    cfg.device = args.device
    device = get_device(cfg)
    if args.smoke:
        cfg.epochs, cfg.mae_epochs = 3, 2
        cfg.inspector_epochs, cfg.n_pos_syn, cfg.n_neg_syn = 5, 400, 300
        cfg.n_unknown_test = 100
        X, y, subj = make_synthetic(cfg, n_per_class=80, seed=cfg.seed)
    else:
        X, y, subj = load_cache(cfg, mmap=True)
    subjects = sorted(set(subj.tolist()))
    folds = subjects[:args.max_folds] if args.max_folds > 0 else subjects
    print(f"== P3 self-personalization loop ==  case={args.case} "
          f"m_adopt={args.m_adopt} blend={args.blend} lam={args.lam} "
          f"alpha={args.alpha} "
          f"folds={len(folds)}/{len(subjects)}  device={device}")

    rows = []
    for ts in folds:
        cs = next(s for s in subjects if s != ts)   # Run-A pairing (cache hits)
        r = fold_loop(cfg, X, y, subj, args.case, ts, cs, device, args)
        rows.append(r)
        print(f"  subj {ts} [{r['cache']}]: adopted {r['adopted']}/3 "
              f"(collected {r['collected']}, contam {100*r['contam']:.0f}%, "
              f"latency {r['latency']:.0f}/{r['stream']})  "
              f"acc_new {100*r['acc_cold']:.1f} -> {100*r['acc_warm']:.1f} "
              f"(oracle {100*r['acc_oracle']:.1f})  "
              f"rejU {100*r['rej_cold']:.0f} -> {100*r['rej_warm']:.0f}",
              flush=True)
        torch.cuda.empty_cache() if device == "cuda" else None

    def m(k):
        v = [r[k] for r in rows if not np.isnan(r[k])]
        return 100 * float(np.mean(v)) if v else float("nan")
    print(f"\n===== LOOP SUMMARY (mean over {len(rows)} folds, %) =====")
    print(f"acc_new  cold {m('acc_cold'):.1f} -> warm {m('acc_warm'):.1f} "
          f"(oracle-personal {m('acc_oracle'):.1f})")
    print(f"rejU     cold {m('rej_cold'):.1f} -> warm {m('rej_warm'):.1f}")
    print(f"adopted  {np.mean([r['adopted'] for r in rows]):.2f}/3 classes, "
          f"contamination {m('contam'):.1f}%, "
          f"latency {np.nanmean([r['latency'] for r in rows]):.0f} of "
          f"~{int(np.mean([r['stream'] for r in rows]))} stream samples")


if __name__ == "__main__":
    main()
