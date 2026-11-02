"""Gate 3 - the clean protocol.  Early decision WITHOUT being told when the gesture starts.

Why this script has to exist.  UWB-Gestures ships 90-frame windows that are already cut
to the gesture.  Frame 0 is therefore an oracle: a plain logistic regression on that
SINGLE frame - 50 ms, before any motion - already reaches 52.9% leave-one-subject-out
accuracy on nine classes (chance 11.1%), because each gesture has a distinctive starting
posture.  Any "early classification" number measured on the shipped windows is largely
posture recognition, and its latency axis is meaningless.  That contamination applies to
every early-decision result on this dataset, ours included.

The clean protocol builds a stream instead:

    [ idle frames ][ the 90 gesture frames ][ idle frames ]
      random onset o                          rest of the window

Idle frames come from the dataset's own empty-gesture class (G12) recorded on the SAME
subject, so the filler is real radar, not zeros.  The model is trained and tested on
streams, gets an explicit idle class, and is never told where the gesture begins.
Latency is then measured from the true onset, which is a number that means something.

Halting rule: commit at the first frame whose argmax is not idle and whose max softmax
reaches tau.  On a gesture outside the vocabulary any commit is a false commit; a commit
BEFORE the onset is a false alarm.

Usage:  python proposal_v2/pilot/early_stream.py --case 1 --epochs 15
        python proposal_v2/pilot/early_stream.py --case 1 --folds 1 --epochs 2 --smoke
"""
import argparse
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
V, TG, M = 3, 90, 189
IDLE_CLASS_ID = 11                    # G12, the dataset's empty gesture
W = 135                               # stream length
ONSET_LO, ONSET_HI = 5, 45


def causal_norm(x):
    """|x| standardised by a RUNNING per-view estimate: deployable in a stream."""
    a = x.abs()
    s = a.mean(dim=3)
    n = torch.arange(1, x.shape[2] + 1, device=x.device, dtype=x.dtype).view(1, 1, -1)
    mu = torch.cumsum(s, dim=2) / n
    sd = torch.sqrt(torch.clamp(torch.cumsum(s * s, dim=2) / n - mu ** 2, min=0.0)) + 1e-3
    scale = torch.maximum(sd, 0.25 * mu) + 1e-6
    return (a - mu.unsqueeze(-1)) / scale.unsqueeze(-1)


class FrameEnc(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(4))
        self.out = nn.Linear(64 * 4, d)

    def forward(self, x):
        return self.out(self.net(x.unsqueeze(1)).flatten(1))


class StreamNet(nn.Module):
    """Causal classifier over a stream, plus a one-step generative head (point 2.2)."""

    def __init__(self, n_cls, d=64, h=192, pred_head=True):
        super().__init__()
        self.fe = FrameEnc(d)
        self.proj = nn.Sequential(nn.Linear(V * d, 256), nn.GELU(), nn.LayerNorm(256))
        self.rnn = nn.GRU(256, h, batch_first=True)
        self.head = nn.Linear(h, n_cls)
        self.pred = nn.Sequential(nn.Linear(h, 512), nn.GELU(),
                                  nn.Linear(512, V * M)) if pred_head else None

    def forward(self, x):                                  # (B,V,W,M)
        B, _, Wn, _ = x.shape
        f = self.fe(x.permute(0, 2, 1, 3).reshape(B * Wn * V, M)).view(B, Wn, -1)
        z, _ = self.rnn(self.proj(f))
        return self.head(z), (self.pred(z).view(B, Wn, V, M) if self.pred else None)


# ---------------------------------------------------------------------------
def build_streams(Xg, Xi, onsets, rng):
    """Xg: (B,V,90,M) gesture frames; Xi: (P,V,90,M) pool of idle frames."""
    B = Xg.shape[0]
    out = np.empty((B, V, W, M), dtype=np.float16)
    for b in range(B):
        o = onsets[b]
        p1, p2 = rng.integers(0, len(Xi), size=2)
        s1 = rng.integers(0, TG - o) if TG - o > 0 else 0
        out[b, :, :o] = Xi[p1][:, s1:s1 + o]
        out[b, :, o:o + TG] = Xg[b]
        tail = W - o - TG
        if tail > 0:
            s2 = rng.integers(0, TG - tail) if TG - tail > 0 else 0
            out[b, :, o + TG:] = Xi[p2][:, s2:s2 + tail]
    return out


def run_fold(Xm, y, subj, gcls, unknown, ts, args, device):
    n_cls = len(gcls) + 1                       # + idle
    idle = len(gcls)
    remap = {c: i for i, c in enumerate(gcls)}

    g_tr = np.where(np.isin(y, gcls) & (subj != ts))[0]
    g_te = np.where((np.isin(y, gcls) | np.isin(y, unknown)) & (subj == ts))[0]
    i_tr = np.where((y == IDLE_CLASS_ID) & (subj != ts))[0]
    i_te = np.where((y == IDLE_CLASS_ID) & (subj == ts))[0]
    if args.smoke:
        g_tr, g_te, i_tr, i_te = g_tr[:192], g_te[:96], i_tr[:16], i_te[:8]

    Xg_tr = np.asarray(Xm[g_tr], dtype=np.float16)
    Xg_te = np.asarray(Xm[g_te], dtype=np.float16)
    Xi_tr = np.asarray(Xm[i_tr], dtype=np.float16)
    Xi_te = np.asarray(Xm[i_te], dtype=np.float16)

    model = StreamNet(n_cls).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, total_steps=args.epochs * max(1, len(g_tr) // args.bs + 1))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    rng = np.random.default_rng(args.seed + ts)

    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(g_tr))
        for b0 in range(0, len(perm), args.bs):
            bi = perm[b0:b0 + args.bs]
            o = rng.integers(ONSET_LO, ONSET_HI + 1, size=len(bi))
            xb = torch.from_numpy(build_streams(Xg_tr[bi], Xi_tr, o, rng)).to(device).float()
            xb = causal_norm(xb)
            lab = np.full((len(bi), W), idle, dtype=np.int64)
            for j, oo in enumerate(o):
                lab[j, oo:oo + TG] = remap[y[g_tr[bi[j]]]]
            lab = torch.from_numpy(lab).to(device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits, nxt = model(xb)
                loss = F.cross_entropy(logits.reshape(-1, n_cls), lab.reshape(-1))
                tgt = xb[:, :, 1:, :].permute(0, 2, 1, 3)
                loss = loss + args.lam * F.mse_loss(nxt[:, :-1], tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
        if args.verbose:
            print(f"    ep{ep+1:02d} loss {float(loss):.4f}", flush=True)

    # deterministic test streams: fixed onsets from a fixed seed
    model.eval()
    rng_te = np.random.default_rng(1000 + ts)
    onsets = rng_te.integers(ONSET_LO, ONSET_HI + 1, size=len(g_te))
    P = np.zeros((len(g_te), W, n_cls), dtype=np.float32)
    E = np.zeros((len(g_te), W), dtype=np.float32)
    Bl = np.zeros((len(g_te), W), dtype=np.float32)
    with torch.no_grad():
        for b0 in range(0, len(g_te), args.bs):
            bi = np.arange(b0, min(b0 + args.bs, len(g_te)))
            xb = torch.from_numpy(
                build_streams(Xg_te[bi], Xi_te, onsets[bi], np.random.default_rng(2000 + b0))
            ).to(device).float()
            xb = causal_norm(xb)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits, nxt = model(xb)
            P[bi] = F.softmax(logits.float(), -1).cpu().numpy()
            x = xb.permute(0, 2, 1, 3)
            E[bi, :-1] = ((nxt.float()[:, :-1] - x[:, 1:]) ** 2).mean((2, 3)).cpu().numpy()
            Bl[bi, :-1] = ((x[:, :-1] - x[:, 1:]) ** 2).mean((2, 3)).cpu().numpy()
    E[:, -1] = E[:, -2]; Bl[:, -1] = Bl[:, -2]
    del Xg_tr, Xg_te, Xi_tr, Xi_te
    return dict(prob=P, perr=E, pbase=Bl, onset=onsets, y=y[g_te], subj=subj[g_te])


# ---------------------------------------------------------------------------
def auroc(a, b):
    r = np.argsort(np.argsort(np.concatenate([a, b])))
    return float((r[:len(a)].mean() - (len(a) - 1) / 2) / len(b))


def report(res, gcls, unknown):
    prob = np.concatenate([r["prob"] for r in res])
    perr = np.concatenate([r["perr"] for r in res])
    pbase = np.concatenate([r["pbase"] for r in res])
    onset = np.concatenate([r["onset"] for r in res])
    y = np.concatenate([r["y"] for r in res])
    gcls = np.asarray(gcls)
    idle = len(gcls)
    known_m = np.isin(y, gcls)
    N = len(y)
    am = prob.argmax(-1)
    msp = prob.max(-1)
    fire = (am != idle) & (msp >= 0.0)                     # placeholder, tau applied below

    print("\n" + "=" * 78)
    print("GATE 3  clean streaming protocol - the model is NOT told where the gesture starts")
    print("=" * 78)
    print(f"  {N} test streams   known {known_m.sum()}  unseen {(~known_m).sum()}"
          f"   window {W} frames   onset uniform on [{ONSET_LO},{ONSET_HI}]")

    print(f"\n{'tau':>5s} | {'KNOWN gestures':^42s} | {'UNSEEN':^20s} | {'idle':>10s}")
    print(f"{'':>5s} | {'detected%':>10s} {'lat ms':>8s} {'p90 ms':>8s} {'acc%':>10s}"
          f" | {'FALSE COMMIT%':>14s} | {'false alarm%':>10s}")
    for tau in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        f = (am != idle) & (msp >= tau)
        first = np.where(f.any(1), f.argmax(1), -1)
        # a commit before the true onset is a false alarm on idle radar
        fa = (first >= 0) & (first < onset)
        det = (first >= 0) & (first >= onset) & known_m
        fc = (first >= 0) & (first >= onset) & ~known_m
        lat = (first[det] - onset[det]) / 20 * 1000
        acc = (gcls[am[det, first[det]]] == y[det]).mean() * 100 if det.sum() else np.nan
        print(f"{tau:5.2f} | {det.sum()/known_m.sum()*100:10.1f} "
              f"{np.median(lat) if det.sum() else np.nan:8.0f} "
              f"{np.percentile(lat,90) if det.sum() else np.nan:8.0f} {acc:10.1f}"
              f" | {fc.sum()/(~known_m).sum()*100:14.1f} | {fa.mean()*100:10.1f}")

    # ---- E3 on the clean protocol: confidence vs time SINCE ONSET ----------
    print(f"\n  E3 re-test, aligned to the true onset "
          f"(confidence in the top non-idle class):")
    nonidle = prob[:, :, :idle]
    conf = nonidle.max(-1) / (nonidle.sum(-1) + 1e-9)      # confidence among gestures
    gate = 1.0 - prob[:, :, idle]                          # how sure "something happened"
    sc = conf * gate
    print(f"{'ms after onset':>15s} {'known':>9s} {'UNSEEN':>9s} {'gap':>7s} "
          f"{'AUROC msp':>10s} {'AUROC -err':>11s} {'AUROC comb':>11s}")
    cum = np.cumsum(perr, 1) / np.arange(1, W + 1)
    cumb = np.cumsum(pbase, 1) / np.arange(1, W + 1)
    rows = []
    for dt in [0, 10, 20, 30, 40, 60, 80]:
        j = np.clip(onset + dt, 0, W - 1)
        r = np.arange(N)
        s_msp, s_err = sc[r, j], -cum[r, j]
        z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
        a1 = auroc(s_msp[known_m], s_msp[~known_m])
        a2 = auroc(s_err[known_m], s_err[~known_m])
        a3 = auroc((z(s_msp) + z(s_err))[known_m], (z(s_msp) + z(s_err))[~known_m])
        rows.append((dt, a1, a2, a3))
        print(f"{dt/20*1000:15.0f} {s_msp[known_m].mean():9.3f} "
              f"{s_msp[~known_m].mean():9.3f} "
              f"{s_msp[known_m].mean()-s_msp[~known_m].mean():7.3f} "
              f"{a1:10.3f} {a2:11.3f} {a3:11.3f}")
    d0, d1 = rows[0], rows[-1]
    print(f"\n  generative one-step skill vs persistence at onset+80: "
          f"known {1-np.mean(cum[known_m, np.clip(onset[known_m]+80,0,W-1)]/(cumb[known_m, np.clip(onset[known_m]+80,0,W-1)]+1e-9)):+.3f}"
          f"   unseen {1-np.mean(cum[~known_m, np.clip(onset[~known_m]+80,0,W-1)]/(cumb[~known_m, np.clip(onset[~known_m]+80,0,W-1)]+1e-9)):+.3f}")
    best = max(rows, key=lambda r: max(r[1], r[2], r[3]))
    print(f"  best open-set AUROC anywhere: {max(max(r[1:]) for r in rows):.3f}"
          f"   (max-softmax alone tops out at {max(r[1] for r in rows):.3f})")
    return prob, perr, pbase, onset, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    X = np.load(os.path.join(CACHE, "X_clutterremoved.npy"), mmap_mode="r")
    y = np.load(os.path.join(CACHE, "y_clutterremoved.npy"))
    subj = np.load(os.path.join(CACHE, "subj_clutterremoved.npy"))

    spec = CFG.REG_TEST_CASES[args.case]
    unknown = [c for c in spec["unknown"] if c != IDLE_CLASS_ID]
    if IDLE_CLASS_ID in spec["unknown"]:
        sys.exit(f"case {args.case} puts G12 in the unseen set; it is the idle filler here")
    gcls = [c for c in range(12) if c not in unknown and c != IDLE_CLASS_ID]
    print(f"[case {args.case}] gesture classes {gcls} + idle(G12)   UNSEEN {unknown}"
          f"   device={device}")

    res, t0 = [], time.time()
    for s in sorted(np.unique(subj))[:args.folds]:
        res.append(run_fold(X, y, subj, gcls, unknown, s, args, device))
        print(f"  fold subj={s} done ({time.time()-t0:.0f}s)", flush=True)

    prob, perr, pbase, onset, yy = report(res, gcls, unknown)
    out = args.out or f"logs/early_stream_c{args.case}.npz"
    np.savez(out, prob=prob, perr=perr, pbase=pbase, onset=onset, y=yy,
             gcls=np.array(gcls), unknown=np.array(unknown))
    print(f"\nsaved -> {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
