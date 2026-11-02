"""Gate 1 - re-test the three early-decision findings with a REAL deep encoder.

The pilot (`early_pilot.py`) used prefix statistics of the Viterbi range tracks plus
logistic regression.  Its headline claim - that waiting inflates confidence on gestures
outside the vocabulary - could be an artefact of a weak classifier.  This script
re-runs the same three questions with a causal deep model on the raw CIR.

Protocol notes that matter:

* CAUSAL NORMALISATION.  `uwb_osr/data.py:_norm` standardises with statistics over the
  WHOLE sequence (axis=(1,2)).  In a streaming setting that leaks the future into every
  prefix.  Here the per-view scale is a running estimate over frames 1..t only.
* ONE PASS, ALL PREFIXES.  A causal GRU emits a prediction at every frame, so a single
  forward pass yields the whole prefix curve; no need to train one model per k.
* DEEP SUPERVISION.  Cross-entropy is applied at every timestep, so the model is
  trained to be honest at short prefixes rather than only at t=90.
* OPEN SET.  Training uses only the `known` classes of a vocabulary.  The `unknown`
  classes of that vocabulary are never seen in training and appear at test only.
* LOSO.  Test subject is held out entirely.

Usage:
  python proposal_v2/pilot/early_deep.py --case 1 --epochs 25
  python proposal_v2/pilot/early_deep.py --case 1 --folds 1 --epochs 3 --smoke
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config as CFG

CACHE = r"D:\uwb_gestures\cache"
V, T, M = 3, 90, 189


# ---------------------------------------------------------------------------
def causal_norm(x):
    """x: (B,V,T,M) signed -> |x| standardised by a RUNNING per-view estimate.

    At frame t only frames 0..t are used, so the transform is deployable in a stream.
    """
    a = x.abs()
    s = a.mean(dim=3)                                   # (B,V,T) per-frame mean level
    c = torch.cumsum(s, dim=2)
    n = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, 1, T)
    run_mu = c / n                                      # running mean of the level
    c2 = torch.cumsum(s * s, dim=2)
    run_sd = torch.sqrt(torch.clamp(c2 / n - run_mu ** 2, min=0.0)) + 1e-3
    # early frames have a degenerate running sd; floor it with the running mean
    scale = torch.maximum(run_sd, 0.25 * run_mu) + 1e-6
    return (a - run_mu.unsqueeze(-1)) / scale.unsqueeze(-1)


class FrameEnc(nn.Module):
    """Shared per-view encoder of one range profile (189 bins) -> d dims."""

    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.out = nn.Linear(64 * 4, d)

    def forward(self, x):                               # x: (N, M)
        h = self.net(x.unsqueeze(1))
        return self.out(h.flatten(1))


class CausalEarlyNet(nn.Module):
    """Causal classifier with an optional generative head.

    The generative head is the cheapest possible version of research point 2.2: from
    the state summarising frames 0..t, predict the echo at frame t+1.  A gesture the
    model has never seen should be harder to extrapolate, so the running prediction
    error becomes a causal open-set score that owes nothing to the classifier's
    confidence.  Whether it actually beats max-softmax is the experiment.
    """

    def __init__(self, n_cls, d=64, h=192, pred_head=False):
        super().__init__()
        self.fe = FrameEnc(d)
        self.proj = nn.Sequential(nn.Linear(V * d, 256), nn.GELU(), nn.LayerNorm(256))
        self.rnn = nn.GRU(256, h, num_layers=1, batch_first=True)
        self.head = nn.Linear(h, n_cls)
        self.pred = nn.Sequential(nn.Linear(h, 512), nn.GELU(),
                                  nn.Linear(512, V * M)) if pred_head else None

    def forward(self, x):                               # x: (B,V,T,M) normalised
        B = x.shape[0]
        f = self.fe(x.permute(0, 2, 1, 3).reshape(B * T * V, M))
        f = f.view(B, T, V * f.shape[-1])
        z, _ = self.rnn(self.proj(f))
        logits = self.head(z)                           # (B,T,C)
        nxt = self.pred(z).view(B, T, V, M) if self.pred is not None else None
        return logits, nxt


# ---------------------------------------------------------------------------
def run_fold(Xm, y, subj, known, unknown, test_subj, args, device):
    cls = sorted(known)
    remap = {c: i for i, c in enumerate(cls)}
    is_known = np.isin(y, cls)
    tr = np.where(is_known & (subj != test_subj))[0]
    te = np.where((is_known | np.isin(y, unknown)) & (subj == test_subj))[0]
    if args.smoke:
        tr, te = tr[:256], te[:128]

    model = CausalEarlyNet(len(cls), pred_head=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, total_steps=args.epochs * max(1, len(tr) // args.bs + 1))
    ytr = torch.tensor([remap[c] for c in y[tr]], device=device)

    # Resident copy of just this fold's frames: removes memmap I/O from the loop.
    # fp16 storage, upcast per batch; ~110 MB per 1000 samples.
    Xtr = np.asarray(Xm[tr], dtype=np.float16)
    Xte = np.asarray(Xm[te], dtype=np.float16)

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    rng = np.random.default_rng(args.seed + test_subj)
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(tr))
        tot, nb = 0.0, 0
        for b0 in range(0, len(perm), args.bs):
            bi = perm[b0:b0 + args.bs]
            xb = torch.from_numpy(Xtr[bi]).to(device).float()
            xb = causal_norm(xb)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits, nxt = model(xb)                 # (B,T,C), (B,T,V,M)
                loss = F.cross_entropy(
                    logits.reshape(-1, len(cls)),
                    ytr[torch.from_numpy(bi).to(device)].repeat_interleave(T))
                # generative term: state at t must extrapolate the echo at t+1
                tgt = xb[:, :, 1:, :].permute(0, 2, 1, 3)          # (B,T-1,V,M)
                loss = loss + args.lam * F.mse_loss(nxt[:, :-1], tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss); nb += 1
        if args.verbose:
            print(f"    ep{ep+1:02d} loss {tot/max(nb,1):.4f}", flush=True)

    model.eval()
    P = np.zeros((len(te), T, len(cls)), dtype=np.float32)
    E = np.zeros((len(te), T), dtype=np.float32)        # model one-step error
    B0 = np.zeros((len(te), T), dtype=np.float32)       # persistence-baseline error
    with torch.no_grad():
        for b0 in range(0, len(te), args.bs):
            bi = np.arange(b0, min(b0 + args.bs, len(te)))
            xb = causal_norm(torch.from_numpy(Xte[bi]).to(device).float())
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits, nxt = model(xb)
            P[bi] = F.softmax(logits.float(), dim=-1).cpu().numpy()
            x = xb.permute(0, 2, 1, 3)                              # (B,T,V,M)
            e = ((nxt.float()[:, :-1] - x[:, 1:]) ** 2).mean(dim=(2, 3))
            # "assume the next frame equals this one" - the free predictor to beat
            b = ((x[:, :-1] - x[:, 1:]) ** 2).mean(dim=(2, 3))
            E[bi, :-1] = e.cpu().numpy(); E[bi, -1] = E[bi, -2]
            B0[bi, :-1] = b.cpu().numpy(); B0[bi, -1] = B0[bi, -2]
    del Xtr, Xte
    return dict(idx=te, prob=P, y=y[te], subj=subj[te], cls=np.array(cls),
                perr=E, pbase=B0)


# ---------------------------------------------------------------------------
def report(res, cls, unknown, tag):
    prob = np.concatenate([r["prob"] for r in res])          # (N,T,C)
    yy = np.concatenate([r["y"] for r in res])
    known_m = np.isin(yy, cls)
    pred = np.array(cls)[prob.argmax(-1)]                    # (N,T)
    msp = prob.max(-1)                                       # (N,T)
    corr = (pred == yy[:, None]) & known_m[:, None]

    KS = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
    print(f"\n{'='*74}\n{tag}\n{'='*74}")
    print("E1 accuracy of known classes vs prefix length")
    print(f"{'k':>4s} {'ms':>6s} {'acc %':>8s}   {'E3 conf known':>14s} "
          f"{'conf UNKNOWN':>13s} {'gap':>7s} {'AUROC':>7s}")
    curve = {}
    for k in KS:
        j = k - 1
        acc = corr[known_m, j].mean() * 100
        ck = msp[known_m, j].mean()
        cu = msp[~known_m, j].mean()
        a, b = msp[known_m, j], msp[~known_m, j]
        r = np.argsort(np.argsort(np.concatenate([a, b])))
        auroc = (r[:len(a)].mean() - (len(a) - 1) / 2) / len(b)
        curve[k] = (acc, ck, cu, auroc)
        print(f"{k:4d} {k/20*1000:6.0f} {acc:8.1f}   {ck:14.3f} {cu:13.3f} "
              f"{ck-cu:7.3f} {auroc:7.3f}")

    k0, k1 = KS[1], KS[-1]
    dk = curve[k1][1] - curve[k0][1]
    du = curve[k1][2] - curve[k0][2]
    best_k = max(KS, key=lambda k: curve[k][0])
    print(f"\n  peak accuracy {curve[best_k][0]:.1f}% at k={best_k} "
          f"({best_k/20*1000:.0f} ms); at k=90 it is {curve[90][0]:.1f}%"
          f"  -> evidence {'EXPIRES' if curve[90][0] < curve[best_k][0]-1 else 'does not expire'}")
    print(f"  confidence drift k={k0}->{k1}:  known {dk:+.3f}   UNKNOWN {du:+.3f}"
          f"   ratio {du/max(dk,1e-6):.2f}")
    print(f"  => {'CONFIRMED: waiting inflates confidence on unseen gestures'if du>0.05 else 'NOT confirmed'}")
    aur = [curve[k][3] for k in KS]
    print(f"  known-vs-unknown AUROC over k: min {min(aur):.3f} at k={KS[int(np.argmin(aur))]}, "
          f"max {max(aur):.3f} at k={KS[int(np.argmax(aur))]}, at k=90 {curve[90][3]:.3f}"
          f"  -> {'NON-MONOTONE' if max(aur)-curve[90][3] > 0.02 else 'monotone-ish'}")

    # ---- research point 2.2: does the generative head beat max-softmax? ----
    if "perr" in res[0]:
        perr = np.concatenate([r["perr"] for r in res])
        pbase = np.concatenate([r["pbase"] for r in res])
        cum = np.cumsum(perr, 1) / np.arange(1, prob.shape[1] + 1)
        cumb = np.cumsum(pbase, 1) / np.arange(1, prob.shape[1] + 1)
        skill = 1.0 - cum / (cumb + 1e-9)               # >0 = beats persistence
        print(f"\n  2.2  causal open-set score from the GENERATIVE head")
        print(f"{'k':>4s} {'ms':>6s} {'AUROC msp':>11s} {'AUROC -err':>12s} "
              f"{'AUROC -skill':>13s} {'AUROC msp+err':>14s}")
        for k in KS:
            j = k - 1
            def au(s):
                a, b = s[known_m], s[~known_m]
                r = np.argsort(np.argsort(np.concatenate([a, b])))
                return (r[:len(a)].mean() - (len(a) - 1) / 2) / len(b)
            z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
            print(f"{k:4d} {k/20*1000:6.0f} {curve[k][3]:11.3f} {au(-cum[:, j]):12.3f} "
                  f"{au(-skill[:, j]):13.3f} "
                  f"{au(z(msp[:, j]) - z(cum[:, j])):14.3f}")
        print(f"  (persistence skill of the generative head on KNOWN classes at k=90: "
              f"{skill[known_m, -1].mean():+.3f}; on unseen: {skill[~known_m, -1].mean():+.3f})")
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the generative term")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    X = np.load(os.path.join(CACHE, "X_clutterremoved.npy"), mmap_mode="r")
    y = np.load(os.path.join(CACHE, "y_clutterremoved.npy"))
    subj = np.load(os.path.join(CACHE, "subj_clutterremoved.npy"))

    spec = CFG.REG_TEST_CASES[args.case]
    unknown = list(spec["unknown"])
    known = [c for c in range(12) if c not in unknown]
    print(f"[case {args.case}] known {known}  UNKNOWN {unknown}  device={device}")

    res, t0 = [], time.time()
    for s in sorted(np.unique(subj))[:args.folds]:
        r = run_fold(X, y, subj, known, unknown, s, args, device)
        res.append(r)
        acc90 = ((np.array(known)[r["prob"][:, -1].argmax(-1)] == r["y"])
                 & np.isin(r["y"], known)).sum() / max(np.isin(r["y"], known).sum(), 1)
        print(f"  fold subj={s}  n_test={len(r['idx'])}  acc@90={acc90*100:.1f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)

    curve = report(res, known, unknown, f"GATE 1 - deep causal model, case {args.case}")

    out = args.out or f"logs/early_pred_c{args.case}.npz"
    np.savez(out,
             prob=np.concatenate([r["prob"] for r in res]),
             y=np.concatenate([r["y"] for r in res]),
             subj=np.concatenate([r["subj"] for r in res]),
             idx=np.concatenate([r["idx"] for r in res]),
             perr=np.concatenate([r["perr"] for r in res]),
             pbase=np.concatenate([r["pbase"] for r in res]),
             cls=np.array(known), unknown=np.array(unknown),
             curve=json.dumps({str(k): list(map(float, v)) for k, v in curve.items()}))
    print(f"\nsaved -> {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
