"""Pilot E1-E3 - is "when to decide" a real problem on THIS dataset, and does the
open-set case break the standard halting assumption?

Every early-classification (ECTS) method assumes a closed vocabulary: keep waiting
until the classifier is confident.  If the incoming gesture is NOT in the vocabulary,
that assumption has no answer - waiting cannot produce legitimate confidence.  The
question this pilot asks is what actually happens instead.

E1  Does evidence accumulate?   accuracy of a prefix-only classifier vs prefix length k.
E2  Is the decidable moment heterogeneous across samples?  If every sample becomes
    decidable at the same k, an adaptive halting rule has no headroom (this is the
    same headroom test as pilot X2b).
E3  THE KEY ONE.  For samples from classes the classifier has never seen, how does
    confidence evolve with k?  If unknown-class confidence rises with k just like
    known-class confidence, then "wait until confident" is pathological in open set:
    the system grows more sure while being wrong by construction.

Zero GPU.  Features are prefix statistics of the three Viterbi range tracks, classifier
is multinomial logistic regression, protocol is leave-one-subject-out.

Usage:  python proposal_v2/pilot/early_pilot.py [--per-cell 20]
"""
import argparse
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from uwb_osr.trajectory import viterbi_track

CACHE = r"D:\uwb_gestures\cache"
# config.py REG_TEST_CASES case 0: base = the other nine, unknown = these three
UNKNOWN = [0, 2, 3]
KS = [10, 20, 30, 40, 50, 60, 70, 80, 90]


def prefix_feats(r, k):
    """Features computable from the first k frames only, per view, causally."""
    f = []
    for v in range(r.shape[0]):
        x = r[v, :k] - r[v, 0]
        d = np.diff(x) if k > 1 else np.zeros(1)
        f += [x[-1], x.mean(), x.std(), np.abs(x).max(),
              d.mean(), d.std(), np.abs(d).sum(), x.min(), x.max()]
    # cross-view contrasts: the geometry that single-view features cannot see
    a = r[:, :k] - r[:, :1]
    f += [(a[0] - a[2])[-1], (a[0] + a[2])[-1] - 2 * a[1][-1],
          (a[0] - a[2]).std(), (a[1] - 0.5 * (a[0] + a[2])).std()]
    return np.array(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=20, help="samples per (gesture, subject)")
    ap.add_argument("--out", default="logs/early_pilot.npz")
    args = ap.parse_args()

    X = np.load(os.path.join(CACHE, "X_clutterremoved.npy"), mmap_mode="r")
    y = np.load(os.path.join(CACHE, "y_clutterremoved.npy"))
    subj = np.load(os.path.join(CACHE, "subj_clutterremoved.npy"))

    rng = np.random.default_rng(0)
    picks = []
    for g in np.unique(y):
        for s in np.unique(subj):
            idx = np.where((y == g) & (subj == s))[0]
            if len(idx):
                picks.extend(rng.choice(idx, min(args.per_cell, len(idx)),
                                        replace=False).tolist())
    picks = np.array(sorted(picks))
    print(f"[pick] {len(picks)} samples  (unknown classes = "
          f"{['G'+str(u+1) for u in UNKNOWN]})")

    # ---- range tracks once, features for every prefix length ---------------
    t0 = time.time()
    R = np.empty((len(picks), 3, 90))
    for n, i in enumerate(picks):
        Xi = np.asarray(X[i], dtype=np.float64)
        for v in range(3):
            R[n, v], _ = viterbi_track(Xi[v])
        if (n + 1) % 400 == 0:
            print(f"  tracked {n+1}/{len(picks)}  ({time.time()-t0:.0f}s)")
    yy, ss = y[picks], subj[picks]
    known_mask = ~np.isin(yy, UNKNOWN)
    print(f"[track] done ({time.time()-t0:.0f}s)   known {known_mask.sum()}  "
          f"unknown {(~known_mask).sum()}\n")

    F = {k: np.array([prefix_feats(R[n], k) for n in range(len(picks))]) for k in KS}

    # ---- leave-one-subject-out, train on KNOWN classes only ----------------
    acc = {k: [] for k in KS}
    conf_known, conf_unk = {k: [] for k in KS}, {k: [] for k in KS}
    correct_curve = np.zeros((len(picks), len(KS)), dtype=bool)   # per-sample
    conf_curve = np.zeros((len(picks), len(KS)))

    for s in np.unique(ss):
        tr = known_mask & (ss != s)
        te = ss == s
        for ki, k in enumerate(KS):
            sc = StandardScaler().fit(F[k][tr])
            clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
            clf.fit(sc.transform(F[k][tr]), yy[tr])
            P = clf.predict_proba(sc.transform(F[k][te]))
            pred = clf.classes_[P.argmax(1)]
            mx = P.max(1)
            te_known = te & known_mask
            te_unk = te & ~known_mask
            acc[k].append(float((pred[known_mask[te]] == yy[te_known]).mean()))
            conf_known[k].append(float(mx[known_mask[te]].mean()))
            conf_unk[k].append(float(mx[~known_mask[te]].mean()))
            correct_curve[np.where(te)[0], ki] = (pred == yy[te])
            conf_curve[np.where(te)[0], ki] = mx

    # ---------------- E1 ---------------------------------------------------
    print("=" * 72)
    print("E1  does evidence accumulate?  (LOSO, train on the 9 known classes)")
    print("=" * 72)
    print(f"{'prefix k':>9s} {'frames->ms':>12s} {'acc known (%)':>15s}")
    for k in KS:
        print(f"{k:9d} {k/20*1000:11.0f}  {np.mean(acc[k])*100:14.1f}")

    # ---------------- E2 ---------------------------------------------------
    print("\n" + "=" * 72)
    print("E2  is the decidable moment heterogeneous?")
    print("=" * 72)
    KA = np.array(KS)
    first_ok = np.full(len(picks), np.nan)
    for n in range(len(picks)):
        c = correct_curve[n]
        # first k after which the prediction stays correct through k=90
        ok = np.where(np.cumprod(c[::-1])[::-1] == 1)[0]
        if len(ok):
            first_ok[n] = KA[ok[0]]
    km = known_mask & ~np.isnan(first_ok)
    fo = first_ok[km]
    print(f"  samples that are ever stably correct: {km.sum()}/{known_mask.sum()}"
          f" ({km.sum()/known_mask.sum()*100:.0f}%)")
    print(f"  their stabilisation prefix k: median {np.median(fo):.0f}"
          f"  IQR {np.percentile(fo,25):.0f}-{np.percentile(fo,75):.0f}"
          f"  10-90% {np.percentile(fo,10):.0f}-{np.percentile(fo,90):.0f}")
    acc_full = np.mean(acc[90])
    for k in KS:
        share = float(np.mean(fo <= k))
        print(f"    by k={k:2d} ({k/20*1000:4.0f} ms): {share*100:5.1f}% of samples "
              f"have already stabilised")
    print(f"  -> a fixed rule must wait for the slowest; an adaptive rule could stop "
          f"half of them at k={np.median(fo):.0f} ({np.median(fo)/20*1000:.0f} ms)")

    # ---------------- E3 THE KEY ONE ---------------------------------------
    print("\n" + "=" * 72)
    print("E3  open set: what does WAITING do to a gesture outside the vocabulary?")
    print("=" * 72)
    print(f"{'prefix k':>9s} {'conf KNOWN':>12s} {'conf UNKNOWN':>14s} {'gap':>8s}")
    for k in KS:
        ck, cu = np.mean(conf_known[k]), np.mean(conf_unk[k])
        print(f"{k:9d} {ck:12.3f} {cu:14.3f} {ck-cu:8.3f}")
    ck0, ck1 = np.mean(conf_known[KS[0]]), np.mean(conf_known[KS[-1]])
    cu0, cu1 = np.mean(conf_unk[KS[0]]), np.mean(conf_unk[KS[-1]])
    print(f"\n  known-class confidence   {ck0:.3f} -> {ck1:.3f}   ({ck1-ck0:+.3f})")
    print(f"  UNKNOWN-class confidence {cu0:.3f} -> {cu1:.3f}   ({cu1-cu0:+.3f})")
    if cu1 - cu0 > 0:
        print(f"  ==> waiting makes the system MORE confident about gestures it has "
              f"never seen.\n      'wait until confident' is pathological in open set.")
    else:
        print(f"  ==> unknown-class confidence does NOT grow; the standard halting "
              f"rule survives open set here.")

    # separability of known vs unknown by confidence, per k (AUROC)
    print(f"\n  can confidence alone tell known from unknown at each k?  (AUROC)")
    for k in KS:
        ki = KS.index(k)
        a, b = conf_curve[known_mask, ki], conf_curve[~known_mask, ki]
        ranks = np.argsort(np.argsort(np.concatenate([a, b])))
        auroc = (ranks[:len(a)].mean() - (len(a) - 1) / 2) / len(b)
        print(f"    k={k:2d}: AUROC {auroc:.3f}")

    np.savez(args.out, KS=np.array(KS), acc=np.array([acc[k] for k in KS]),
             conf_known=np.array([conf_known[k] for k in KS]),
             conf_unk=np.array([conf_unk[k] for k in KS]),
             correct_curve=correct_curve, conf_curve=conf_curve,
             first_ok=first_ok, y=yy, subj=ss, known_mask=known_mask)
    print(f"\nsaved -> {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
