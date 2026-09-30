"""G0 formal verdict from the controlled-stream dumps (design doc SS8 G0).

Pure numpy post-processing of tools/p4_stream.py's per-fold/family npz
dumps -- no torch, no forward passes. Per (case, family) it aggregates the
fold metrics (via p4_stream._fold_metrics, the single evaluator), prints
the tripwire table against the clean reference, and applies the SS8 G0
gate: the problem "compound drift breaks the frozen recipe" stands when at
least TWO degradation families trip any of

  H : H-mean drop >= 10 pts vs clean (full stream or degraded subset);
  C : max |coverage - 90| grows >= 10 pts vs clean;
  A : SS39 adoption-pool (first m=5/class singleton hits) contamination
      grows >= 10 pts vs clean. The doc's literal absolute ">10%" bar is
      degenerate: clean-stream contam5 already exceeds 10% on folds whose
      per-class new q collapses (the SS38 calib scale lottery), so even
      atten -- bit-identical scores to clean -- would "trip". All three
      wires therefore read drift-attributable DAMAGE relative to clean.

Conditions are (family x degraded-view) cells; the gate counts distinct
FAMILIES, and a family trips when ANY of its cells trips in ANY vocab (the
G0 claim is existential: some realisation of that drift type breaks the
frozen recipe; every cell is still reported, per SS4.4).

Usage:
  python tools/p4_g0_verdict.py                # cases 0,1 / all families
  python tools/p4_g0_verdict.py --cases 0
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from p4_stream import G0_FAMILIES, _cond_sfx, _fold_metrics    # noqa: E402

VIEW_TOKENS = (0, 1, 2, "all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="0,1")
    ap.add_argument("--prefix", default="logs/p4_stream")
    args = ap.parse_args()
    cases = [int(c) for c in args.cases.split(",")]

    trip_union = {}                               # fam -> set of trip letters
    save = {}
    for ci in cases:
        rows = {}                                 # fam -> [fold metric dicts]
        folds = sorted(int(f.rsplit("_s", 1)[1].split(".")[0]) for f in
                       glob.glob(f"{args.prefix}_c{ci}_clean_s*.npz"))
        if not folds:
            print(f"case {ci}: no dumps found, skipped")
            continue
        conds = ["clean"] + [fam + _cond_sfx(vt) for vt in VIEW_TOKENS
                             for fam in G0_FAMILIES]
        for key in conds:
            for ts in folds:
                f = f"{args.prefix}_c{ci}_{key}_s{ts}.npz"
                if not os.path.exists(f):
                    continue
                d = np.load(f)
                rows.setdefault(key, []).append(_fold_metrics(d["R"], d["E"], d))

        def m(fam, grp, k):
            return 100 * float(np.nanmean([r[grp][k] for r in rows[fam]]))

        def cerr(fam):
            return [abs(m(fam, "full", c) - 90) for c in ("cov_b", "cov_n")]

        cl_full, cl_hot = m("clean", "full", "H"), m("clean", "hot", "H")
        cl_ce = max(cerr("clean"))
        cl_c5 = 100 * float(np.nanmean([r["contam5"] for r in rows["clean"]]))
        print(f"\n== G0 case {ci} ({len(rows['clean'])} folds; clean "
              f"H={cl_full:.1f} maxCovErr={cl_ce:.1f} contam5={cl_c5:.1f}) ==")
        print(f"  {'condition':13s} {'dH':>6s} {'dH_hot':>7s} {'dCovE':>6s} "
              f"{'contam5':>8s} {'contam':>7s} {'nAd5':>5s} {'trip':>5s}")
        for vt in VIEW_TOKENS:
            for fam in G0_FAMILIES:
                key = fam + _cond_sfx(vt)
                if key not in rows:
                    continue
                dh = cl_full - m(key, "full", "H")
                dhh = cl_hot - m(key, "hot", "H")
                dce = max(cerr(key)) - cl_ce
                c5 = 100 * float(np.nanmean([r["contam5"] for r in rows[key]]))
                cu = 100 * float(np.nanmean([r["contam"] for r in rows[key]]))
                n5 = float(np.mean([r["n_adopt5"] for r in rows[key]]))
                trips = set()
                if dh >= 10 or dhh >= 10:
                    trips.add("H")
                if dce >= 10:
                    trips.add("C")
                if c5 - cl_c5 >= 10:
                    trips.add("A")
                trip_union.setdefault(fam, set()).update(trips)
                print(f"  {key:13s} {dh:6.1f} {dhh:7.1f} {dce:6.1f} {c5:8.1f} "
                      f"{cu:7.1f} {n5:5.1f} "
                      f"{'/'.join(sorted(trips)) or '-':>5s}")
                save[f"c{ci}_{key}"] = np.array([dh, dhh, dce, c5, cu])

    tripped = sorted(f for f, t in trip_union.items() if t)
    print(f"\nG0 gate: families tripping in >=1 vocab: "
          f"{', '.join(tripped) or 'none'} "
          f"({len(tripped)}/{len(G0_FAMILIES)}; need >=2)")
    print(f"G0 verdict: {'PASS -- the problem stands' if len(tripped) >= 2 else 'FAIL -- stop the direction (SS8 G0)'}")
    if save:
        np.savez("logs/p4_g0_verdict.npz",
                 **{k: v for k, v in save.items()})
        print("saved logs/p4_g0_verdict.npz")


if __name__ == "__main__":
    main()
