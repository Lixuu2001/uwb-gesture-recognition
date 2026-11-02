# -*- coding: utf-8 -*-
"""Training-subject scaling curve for streaming gesture event detection.

The 2x2 isolation (proposal_v2/28 sec 5bis) showed the subject axis dominates:
within-subject 0.736 vs cross-subject 0.357 event F1, roughly 6x the effect of
the normalization fix.  The obvious follow-up question is whether that gap is a
data-quantity problem (too few training subjects) or a representation problem
(the model never learns a subject-invariant cue).

This script answers it the only way that separates the two: hold the TEST
subject fixed and sweep the number of TRAINING subjects.  If the curve is still
climbing at the largest N, more subjects would help and the gap is partly a
sample-size artefact; if it saturates well below the within-subject reference,
more data will not fix it and the problem is representational.

Usage:
  python tools/fmcw_scaling.py --test-users 3 4 5 --pool 6 7 8 9 10 11 12 \
      --n-train 1 2 4 7 --epochs 15
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmcw_stream as fs                                   # noqa: E402


class Args:
    """Minimal stand-in for the argparse namespace fmcw_stream.run_fold wants."""

    def __init__(self, **kw):
        self.epochs = 15
        self.batch = 32
        self.taus = [0.5, 0.7, 0.9]
        self.held_out_class = None
        self.stream_recs = 200
        self.within_user = False
        self.geom = False
        self.geom_only = False
        self.gt_shift = 0
        self.norm = "window"
        self.verbose = False
        self.__dict__.update(kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-users", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--pool", type=int, nargs="+",
                    default=[6, 7, 8, 9, 10, 11, 12])
    ap.add_argument("--n-train", type=int, nargs="+", default=[1, 2, 4, 7])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--stream-recs", type=int, default=200)
    ap.add_argument("--norm", default="window")
    ap.add_argument("--geom", action="store_true")
    ap.add_argument("--geom-only", action="store_true")
    ap.add_argument("--gt-shift", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/fmcw_scaling.npz")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    print("scaling curve: test %s, pool %s, n_train %s, norm %s, geom %s"
          % (a.test_users, a.pool, a.n_train, a.norm, a.geom))

    results = {}
    t0 = time.time()
    for n in a.n_train:
        rows = []
        for tu in a.test_users:
            pool = [u for u in a.pool if u != tu]
            train = list(rng.choice(pool, size=min(n, len(pool)), replace=False))
            args = Args(epochs=a.epochs, stream_recs=a.stream_recs,
                        norm=a.norm, geom=a.geom or a.geom_only,
                        geom_only=a.geom_only, gt_shift=a.gt_shift)
            # run_fold trains on `users` minus `test_user`, so pass train+[tu]
            print("  n_train=%d test=user%d train=%s (%.0fs)"
                  % (n, tu, train, time.time() - t0), flush=True)
            rows += fs.run_fold(train + [tu], tu, args, rng)
        for tau in args.taus:
            sel = [r for r in rows if r["tau"] == tau]
            results[(n, tau)] = {k: float(np.nanmean([r[k] for r in sel]))
                                 for k in ("event_f1", "recall", "precision",
                                           "onset_mae", "false_alarm_per_min")}

    print("\n%8s %6s %10s %8s %8s %10s" %
          ("n_train", "tau", "event_f1", "recall", "prec", "fa/min"))
    for n in a.n_train:
        for tau in [0.5, 0.7, 0.9]:
            m = results[(n, tau)]
            print("%8d %6.2f %10.3f %8.3f %8.3f %10.2f"
                  % (n, tau, m["event_f1"], m["recall"], m["precision"],
                     m["false_alarm_per_min"]))
    print("\nreference points (same metric, tau=0.9): within-subject 0.736, "
          "6-fold LOSO with 5 training subjects 0.357")
    os.makedirs("logs", exist_ok=True)
    np.savez(a.out, results={str(k): v for k, v in results.items()},
             test_users=a.test_users, pool=a.pool, n_train=a.n_train)
    print("saved: " + a.out)


if __name__ == "__main__":
    main()
