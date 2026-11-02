"""G2 formal verdict from the adaptive-stream dumps (design doc SS8 G2).

Pure post-processing of tools/p4_g2_stream.py's per-(condition x budget x
fold) npz dumps. At the main budget B=5, per (vocab, condition):

  P  pool contamination < 5% (uncapped, the deployed gate's actual pool);
  C  full-stream offline known-class coverage, 8-fold macro mean in
     [87, 93] (SS5.6 level-1; per-fold +-3 gates are pre-registered OUT --
     binomial SE ~2.4 on ~150 new-class samples would false-kill);
  H  H-mean at least +5 pts over the frozen static reference (B=0, same
     bank, same stream). NOTE: SS8 words this against "paper-3's static
     closed loop"; the singleton-personalisation baseline (SS6.1 #3) is a
     separate ablation, so this script judges against B=0 and says so;
  F  base forgetting (B=0 acc_b minus B=5 acc_b) <= 1 pt.

Also reported: per-fold |cov-90| median / p90 / worst vs the static
reference (the scale-lottery diagnostic; cluster bootstrap deferred to
the paper run), and the B in {0,2,5,10} budget curve on H.

Usage:
  python tools/p4_g2_verdict.py                 # cases 0,1, all conditions
  python tools/p4_g2_verdict.py --cases 0 --budget 5
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from p4_stream import G0_FAMILIES, _cond_sfx     # noqa: E402

VIEW_TOKENS = (0, 1, 2, "all")


def _load(ci, key, B, ts):
    f = f"logs/p4_g2_c{ci}_{key}_B{B}_s{ts}.npz"
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    return {k: (d[k].item() if d[k].ndim == 0 else d[k])
            for k in d.files if k != "pool"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="0,1")
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--budgets", default="0,2,5,10")
    args = ap.parse_args()
    B = args.budget
    budgets = [int(b) for b in args.budgets.split(",")]

    for ci in [int(c) for c in args.cases.split(",")]:
        folds = sorted({int(f.rsplit("_s", 1)[1].split(".")[0]) for f in
                        glob.glob(f"logs/p4_g2_c{ci}_clean_B{B}_s*.npz")})
        if not folds:
            print(f"case {ci}: no dumps, skipped")
            continue
        conds = ["clean"] + [f + _cond_sfx(vt) for vt in VIEW_TOKENS
                             for f in G0_FAMILIES]
        print(f"\n== G2 case {ci} (B={B} vs B=0, {len(folds)} folds) ==")
        print(f"  {'condition':13s} {'H0':>6s} {'H5':>6s} {'dH':>6s} "
              f"{'cov':>6s} {'covErr med/p90/max':>19s} {'contam':>7s} "
              f"{'forget':>7s} {'rejU0/5':>10s} {'gate':>6s}")
        n_pass = n_all = 0
        for key in conds:
            pairs = [(ts, _load(ci, key, 0, ts), _load(ci, key, B, ts))
                     for ts in folds]
            pairs = [(ts, a, b) for ts, a, b in pairs if a and b]
            if not pairs:
                continue
            fts = [ts for ts, _, _ in pairs]
            r0 = [a for _, a, _ in pairs]
            r5 = [b for _, _, b in pairs]
            # pooled known-class coverage: covered[] from the g2 dump,
            # kind[] from the G0 stream dump (same fold, same event order)
            kinds = {ts: np.load(f"logs/p4_stream_c{ci}_clean_s{ts}.npz")
                     ["kind"] for ts in fts}

            def pooled(rs):
                out = []
                for ts, r in zip(fts, rs):
                    known = kinds[ts] <= 1
                    out.append(float(np.asarray(r["covered"])[known].mean()))
                return out

            cov5, cov0 = pooled(r5), pooled(r0)
            H0 = 100 * float(np.mean([r["H"] for r in r0]))
            H5 = 100 * float(np.mean([r["H"] for r in r5]))
            cov = 100 * float(np.mean(cov5))
            ce5 = [abs(100 * c - 90) for c in cov5]
            ce0 = [abs(100 * c - 90) for c in cov0]
            contam = 100 * float(np.nanmean([r["contam"] for r in r5]))
            forget = 100 * (float(np.mean([r["acc_b"] for r in r0]))
                            - float(np.mean([r["acc_b"] for r in r5])))
            rej0 = 100 * float(np.mean([r["rej_u"] for r in r0]))
            rej5 = 100 * float(np.mean([r["rej_u"] for r in r5]))
            gate = {"P": (contam < 5) or np.isnan(contam),
                    "C": 87 <= cov <= 93,
                    "H": H5 - H0 >= 5,
                    "F": forget <= 1}
            ok = "/".join(k for k, v in gate.items() if v)
            n_pass += all(gate.values())
            n_all += 1
            print(f"  {key:13s} {H0:6.1f} {H5:6.1f} {H5-H0:+6.1f} "
                  f"{cov:6.1f} {np.median(ce5):6.1f}/{np.percentile(ce5, 90):4.1f}"
                  f"/{max(ce5):4.1f} {contam:7.1f} {forget:+7.1f} "
                  f"{rej0:4.1f}/{rej5:4.1f} {ok:>6s}")
            # scale-lottery diagnostic vs static
            if key == "clean":
                print(f"    per-fold |cov-90|: static med {np.median(ce0):.1f} "
                      f"p90 {np.percentile(ce0, 90):.1f} max {max(ce0):.1f} "
                      f"-> adaptive med {np.median(ce5):.1f} "
                      f"p90 {np.percentile(ce5, 90):.1f} max {max(ce5):.1f}")
        print(f"  gate P/C/H/F all-pass: {n_pass}/{n_all} conditions")

        # budget curve on the headline conditions
        print(f"  budget curve (H, mean over folds):")
        for key in ("clean", "drop", "domain"):
            row = f"    {key:8s}"
            for b in budgets:
                rr = [_load(ci, key, b, ts) for ts in folds]
                rr = [r for r in rr if r]
                row += (f"  B{b}={100 * float(np.mean([r['H'] for r in rr])):5.1f}"
                        if rr else f"  B{b}=  -  ")
            print(row)


if __name__ == "__main__":
    main()
