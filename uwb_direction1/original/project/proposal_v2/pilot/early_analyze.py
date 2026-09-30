"""Gate 1b/2 - what a standard halting rule actually does in open set, and whether
any causal signal spots an unseen gesture earlier than raw confidence does.

Input: the (N, T, C) per-frame softmax dumped by `early_deep.py`.  Everything here is
post-processing, so new signals cost seconds, not GPU hours.

Part A  Deployment simulation of the textbook ECTS rule
        "halt at the first frame where max softmax >= tau".
        For known gestures that yields a latency and an accuracy.  For gestures outside
        the vocabulary EVERY halt is a false commit - the system answers with a label
        that cannot be right.  Sweeping tau gives the latency / false-commit frontier
        that the ECTS literature never has to draw, because it assumes a closed set.

Part B  Signal race.  At each prefix length, how well does each CAUSAL score separate
        known from unknown?  Confidence-family scores (max softmax, margin, negative
        entropy) versus stability-family scores computed from the prediction TRAJECTORY
        so far (how often the argmax has flipped, how much the posterior still moves).
        Hypothesis: a model fed an unseen gesture is not just less confident, it is
        less SETTLED, and being unsettled shows up earlier.

Usage:  python proposal_v2/pilot/early_analyze.py logs/early_deep_e15_c1.npz
        python proposal_v2/pilot/early_analyze.py "logs/early_deep_e15_c*.npz"
"""
import argparse
import glob
import sys

import numpy as np

KS = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
TAUS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def auroc(pos, neg):
    """P(score(pos) > score(neg)); pos = known, neg = unknown."""
    r = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return float((r[:len(pos)].mean() - (len(pos) - 1) / 2) / len(neg))


def signals(prob):
    """Causal scores at every prefix length.  prob: (N,T,C) -> dict of (N,T) arrays.

    All scores are oriented so that HIGHER = more like a known class.
    """
    N, T, C = prob.shape
    srt = np.sort(prob, axis=-1)
    out = {
        "msp": prob.max(-1),
        "margin": srt[..., -1] - srt[..., -2],
        "neg_entropy": (prob * np.log(prob + 1e-12)).sum(-1),
    }
    am = prob.argmax(-1)                                     # (N,T)
    flip = np.zeros_like(prob[..., 0])
    flip[:, 1:] = (am[:, 1:] != am[:, :-1]).astype(np.float32)
    cum_flip = np.cumsum(flip, axis=1)
    idx = np.arange(1, T + 1, dtype=np.float32)
    out["neg_fliprate"] = -cum_flip / idx                    # churn so far

    # how much the posterior is still moving: running mean of ||p_t - p_{t-1}||_1
    step = np.zeros_like(prob[..., 0])
    step[:, 1:] = np.abs(prob[:, 1:] - prob[:, :-1]).sum(-1)
    out["neg_drift"] = -np.cumsum(step, axis=1) / idx

    # dwell: share of frames so far whose argmax equals the CURRENT argmax
    dwell = np.zeros_like(prob[..., 0])
    for t in range(T):
        dwell[:, t] = (am[:, :t + 1] == am[:, t:t + 1]).mean(axis=1)
    out["dwell"] = dwell

    # settled confidence: confidence discounted by how unsettled the run has been
    out["msp_x_dwell"] = out["msp"] * dwell
    return out


def part_a(prob, known_m, cls, y):
    am = prob.argmax(-1)
    pred = np.asarray(cls)[am]
    msp = prob.max(-1)
    T = prob.shape[1]
    print("\n" + "=" * 78)
    print("A  the textbook rule: halt at the first frame with max-softmax >= tau")
    print("=" * 78)
    print(f"{'tau':>6s} | {'KNOWN gestures':^34s} | {'UNSEEN gestures':^30s}")
    print(f"{'':>6s} | {'halted%':>8s} {'latency ms':>11s} {'acc@halt%':>12s} | "
          f"{'FALSE COMMIT%':>14s} {'its latency ms':>14s}")
    rows = []
    for tau in TAUS:
        fire = msp >= tau
        first = np.where(fire.any(1), fire.argmax(1), -1)
        hk = (first >= 0) & known_m
        hu = (first >= 0) & ~known_m
        lat_k = (first[hk] + 1) / 20 * 1000
        lat_u = (first[hu] + 1) / 20 * 1000
        acc = (pred[hk, first[hk]] == y[hk]).mean() * 100 if hk.sum() else np.nan
        rows.append((tau, hk.sum() / known_m.sum() * 100, np.median(lat_k) if hk.sum() else np.nan,
                     acc, hu.sum() / (~known_m).sum() * 100,
                     np.median(lat_u) if hu.sum() else np.nan))
        print(f"{tau:6.2f} | {rows[-1][1]:8.1f} {rows[-1][2]:11.0f} {acc:12.1f} | "
              f"{rows[-1][4]:14.1f} {rows[-1][5]:14.0f}")
    print(f"\n  reference: always wait the full {T} frames "
          f"({T/20*1000:.0f} ms) -> false commit 100.0% by construction")
    print("  every row above is a false commit on an unseen gesture: the rule has no")
    print("  way to say 'this will never become decidable'.")
    return rows


def part_b(prob, known_m):
    sig = signals(prob)
    names = ["msp", "margin", "neg_entropy", "neg_fliprate", "neg_drift",
             "dwell", "msp_x_dwell"]
    print("\n" + "=" * 78)
    print("B  signal race: AUROC(known vs unseen) from a causal prefix of length k")
    print("=" * 78)
    hdr = "".join(f"{n:>13s}" for n in names)
    print(f"{'k':>4s}{'ms':>7s}" + hdr)
    tab = {}
    for k in KS:
        j = k - 1
        row = []
        for n in names:
            s = sig[n][:, j]
            row.append(auroc(s[known_m], s[~known_m]))
        tab[k] = row
        print(f"{k:4d}{k/20*1000:7.0f}" + "".join(f"{v:13.3f}" for v in row))

    print("\n  best signal at each k:")
    for k in KS:
        b = int(np.argmax(tab[k]))
        d = tab[k][b] - tab[k][0]
        print(f"    k={k:2d} ({k/20*1000:4.0f} ms): {names[b]:>12s} "
              f"{tab[k][b]:.3f}   (max-softmax {tab[k][0]:.3f}, {d:+.3f})")

    # how early does each signal reach the level max-softmax needs the full run for?
    target = tab[90][0]
    print(f"\n  earliest k at which each signal reaches max-softmax's FULL-RUN "
          f"AUROC ({target:.3f}):")
    for i, n in enumerate(names):
        hit = [k for k in KS if tab[k][i] >= target]
        if hit:
            print(f"    {n:>12s}: k={hit[0]:2d} ({hit[0]/20*1000:4.0f} ms)"
                  f"   -> {(90-hit[0])/20*1000:4.0f} ms earlier")
        else:
            print(f"    {n:>12s}: never")
    return tab, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    args = ap.parse_args()
    files = sorted(glob.glob(args.pattern))
    if not files:
        sys.exit(f"no files match {args.pattern}")

    agg_b = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        prob, y, cls = d["prob"], d["y"], d["cls"]
        known_m = np.isin(y, cls)
        print("\n" + "#" * 78)
        print(f"# {f}   n={len(y)}  known={known_m.sum()}  unseen={(~known_m).sum()}"
              f"  |vocab|={len(cls)}  unseen classes={list(d['unknown'])}")
        print("#" * 78)
        part_a(prob, known_m, cls, y)
        tab, names = part_b(prob, known_m)
        agg_b[f] = tab

    if len(files) > 1:
        print("\n" + "=" * 78)
        print(f"ACROSS {len(files)} VOCABULARIES - mean AUROC(known vs unseen)")
        print("=" * 78)
        print(f"{'k':>4s}" + "".join(f"{n:>13s}" for n in names))
        for k in KS:
            m = np.mean([agg_b[f][k] for f in files], axis=0)
            print(f"{k:4d}" + "".join(f"{v:13.3f}" for v in m))
        print("\n  best signal per k (mean over vocabularies):")
        for k in KS:
            m = np.mean([agg_b[f][k] for f in files], axis=0)
            b = int(np.argmax(m))
            print(f"    k={k:2d}: {names[b]:>12s} {m[b]:.3f}  "
                  f"(max-softmax {m[0]:.3f}, {m[b]-m[0]:+.3f})")


if __name__ == "__main__":
    main()
