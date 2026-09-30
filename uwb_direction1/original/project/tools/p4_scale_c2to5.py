"""Data analysis for the c2-c5 final-phase dumps (paper-4, RESULTS SS43).

Three questions, all pure numpy post-processing on logs/n3_scores_bal_b2e_c{2..5}:

  1. Does calib-select (G1 main method, SS42) hold on the 4 new vocabularies?
     Same conventions as tools/p4_g1_verdict.py: s grid 1.0..3.0 step .1,
     base-drop<=2pts constraint vs same-bank s=1, b2e gate readout.
  2. Which register classes are twin-eaten?  Per-register-class routing acc
     at s=1 and where the misrouted mass goes (argmin base column).
  3. How big is the per-fold lottery?  Per-fold H spread at calib-s.

Usage:
  python tools/p4_scale_c2to5.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REG_TEST_CASES  # noqa: E402

S_GRID = np.round(np.arange(1.0, 3.0 + 1e-9, 0.1), 2)
BASE_DROP = 0.02
CASES = [2, 3, 4, 5]


def _acc(R, yt, nb, s):
    Rs = R.copy()
    Rs[:, :nb] *= s
    return float((Rs.argmin(1) == yt).mean())


def _select(Rb, yb, Rr, yr, nb):
    cv = []
    for s in S_GRID:
        ab, an = _acc(Rb, yb, nb, s), _acc(Rr, yr, nb, s)
        cv.append((s, ab, an, 2 * ab * an / (ab + an + 1e-9)))
    feas = [t for t in cv if t[1] >= cv[0][1] - BASE_DROP]
    return (max(feas, key=lambda t: t[3])[0] if feas else 1.0), cv


def main():
    for ci in CASES:
        rc = REG_TEST_CASES[ci]
        base_g, reg_g = rc["base"], rc["register"]
        rows, twin = [], {}          # twin[reg_col] = Counter over pred cols
        for ts in range(1, 9):
            f = f"logs/n3_scores_bal_b2e_c{ci}_s{ts}.npz"
            if not os.path.exists(f):
                continue
            d = np.load(f)
            nb = int(d["n_base"])
            s_cal, _ = _select(d["R_cal"], d["y_cal"],
                               d["R_cal_reg"], d["y_cal_reg"], nb)
            s_orc, _ = _select(d["R_base"], d["y_base"],
                               d["R_reg"], d["y_reg"], nb)
            row = {"ts": ts, "s_cal": s_cal, "s_orc": s_orc}
            for tag, s in [("1", 1.0), ("15", 1.5), ("cal", s_cal),
                           ("orc", s_orc)]:
                ab = _acc(d["R_base"], d["y_base"], nb, s)
                an = _acc(d["R_reg"], d["y_reg"], nb, s)
                row[f"b{tag}"], row[f"n{tag}"] = ab, an
                row[f"H{tag}"] = 2 * ab * an / (ab + an + 1e-9)
            rows.append(row)
            # twin audit at s=1 (raw geometry, before any correction)
            pred = d["R_reg"].argmin(1)
            for k in range(nb, nb + len(reg_g)):
                m = d["y_reg"] == k
                for p in pred[m]:
                    twin.setdefault(k, {}).setdefault(int(p), 0)
                    twin[k][int(p)] += 1

        def m(k):
            return 100 * float(np.mean([r[k] for r in rows]))

        print(f"\n== b2e c{ci}  base={base_g} register={reg_g} "
              f"({len(rows)} folds) ==")
        print(f"  s_cal mean {np.mean([r['s_cal'] for r in rows]):.2f}  "
              f"oracle mean {np.mean([r['s_orc'] for r in rows]):.2f}  "
              f"|s_cal-oracle|<=0.3 in "
              f"{sum(abs(r['s_cal']-r['s_orc']) <= 0.3 for r in rows)}/{len(rows)}")
        for name, tag in [("s=1", "1"), ("s=1.5", "15"),
                          ("calib-sel", "cal"), ("oracle", "orc")]:
            print(f"  {name:10s} b {m('b'+tag):5.1f}  n {m('n'+tag):5.1f}  "
                  f"H {m('H'+tag):5.1f}")
        g_orc, g_15 = m("Horc") - m("Hcal"), m("H15") - m("Hcal")
        g_bd, g_nn = m("b1") - m("bcal"), m("ncal") - m("n1")
        print(f"  gate: H gap to oracle {g_orc:.1f} "
              f"({'PASS' if g_orc <= 2 else 'FAIL'}); "
              f"vs s=1.5 {-g_15:+.1f} ({'PASS' if g_15 <= 1 else 'FAIL'}); "
              f"base drop {g_bd:.1f} ({'PASS' if g_bd <= 2 else 'FAIL'}); "
              f"new gain {g_nn:+.1f} ({'PASS' if g_nn >= 5 else 'FAIL'})")
        # fold lottery at calib-s
        Hs = [100 * r["Hcal"] for r in rows]
        print(f"  fold H@cal: min {min(Hs):.1f}  max {max(Hs):.1f}  "
              f"spread {max(Hs)-min(Hs):.1f}")
        # per-register-class routing at s=1 (gesture ids via config)
        nb = 6
        print("  twin audit (s=1): reg gesture -> routed-to top sinks")
        for k in sorted(twin):
            g_id = reg_g[k - nb]
            tot = sum(twin[k].values())
            acc_k = 100 * twin[k].get(k, 0) / tot
            sinks = sorted(((c, n) for c, n in twin[k].items() if c != k),
                           key=lambda t: -t[1])[:2]
            lab = ", ".join(
                f"{'base g'+str(base_g[c]) if c < nb else 'reg g'+str(reg_g[c-nb])}"
                f" {100*n/tot:.0f}%" for c, n in sinks)
            print(f"    reg g{g_id}: own {acc_k:.0f}%   leaks -> {lab}")


if __name__ == "__main__":
    main()
