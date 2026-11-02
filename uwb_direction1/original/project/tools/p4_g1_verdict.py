"""G1 formal verdict: calib-select vs the re-anchored gate (paper-4 SS8 G1b).

calib-select picks the routing scale on the deployed bank using ONLY the
calibration users' base (R_cal) and register (R_cal_reg) rows -- data the
frozen recipe's conformal side already requires -- then the pick transfers
to the held-out test user. Evaluated separately on the two dump families:

  b2e   fixed pairing, 6 training subjects, s=1 uncorrected (SS36/SS40);
  rot2  rotated pairing, 5 training subjects + 2 pooled calib, s=1 already
        half-corrected by the loose-base side effect (SS40) -- the "+5 new
        acc vs s=1" clause is only demanded of b2e for that reason.

Pure numpy post-processing on the dumps; no training. Prints the per-family
gate table and saves logs/p4_g1_verdict.npz for the paper.

Usage:
  python tools/p4_g1_verdict.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

S_GRID = np.round(np.arange(1.0, 3.0 + 1e-9, 0.1), 2)
BASE_DROP = 0.02

FAMILIES = [                        # (family, case, dump prefix)
    ("b2e", 0, "logs/n3_scores_bal_b2e_c0"),
    ("rot2", 0, "logs/n3_scores_bal_b2e_rot2_c0"),
    ("rot2", 1, "logs/n3_scores_bal_b2e_rot2_c1"),
]


def _acc(R, yt, nb, s):
    Rs = R.copy()
    Rs[:, :nb] *= s
    return float((Rs.argmin(1) == yt).mean())


def _select(Rb, yb, Rr, yr, nb):
    """H-max on the grid subject to base drop <= 2pts vs the same bank s=1."""
    cv = []
    for s in S_GRID:
        ab, an = _acc(Rb, yb, nb, s), _acc(Rr, yr, nb, s)
        cv.append((s, ab, an, 2 * ab * an / (ab + an + 1e-9)))
    feas = [t for t in cv if t[1] >= cv[0][1] - BASE_DROP]
    return (max(feas, key=lambda t: t[3])[0] if feas else 1.0), cv


def main():
    all_rows = {}
    for fam, ci, prefix in FAMILIES:
        rows = []
        for ts in range(1, 9):
            f = f"{prefix}_s{ts}.npz"
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
        if not rows:
            continue
        all_rows[(fam, ci)] = rows

        def m(k):
            return 100 * float(np.mean([r[k] for r in rows]))
        print(f"\n== {fam} case {ci} ({len(rows)} folds) ==")
        print(f"  s_cal mean {np.mean([r['s_cal'] for r in rows]):.2f}  "
              f"oracle mean {np.mean([r['s_orc'] for r in rows]):.2f}  "
              f"|s_cal-oracle|<=0.3 in "
              f"{sum(abs(r['s_cal']-r['s_orc']) <= 0.3 for r in rows)}/{len(rows)}")
        print(f"  {'variant':10s} {'acc_b':>7s} {'acc_n':>7s} {'H':>7s}")
        for name, tag in [("s=1", "1"), ("s=1.5", "15"),
                          ("calib-sel", "cal"), ("oracle", "orc")]:
            print(f"  {name:10s} {m('b'+tag):7.1f} {m('n'+tag):7.1f} "
                  f"{m('H'+tag):7.1f}")
        # gate readout (twin-heavy case 1 reported, not judged on oracle gap)
        g_orc = m("Horc") - m("Hcal")
        g_15 = m("H15") - m("Hcal")
        g_bd = m("b1") - m("bcal")
        g_nn = m("ncal") - m("n1")
        checks = [f"H gap to oracle {g_orc:.1f} ({'PASS' if g_orc <= 2 else 'FAIL'})",
                  f"vs s=1.5 {-g_15:+.1f} ({'PASS' if g_15 <= 1 else 'FAIL'})",
                  f"base drop {g_bd:.1f} ({'PASS' if g_bd <= 2 else 'FAIL'})"]
        if fam == "b2e":
            checks.append(f"new gain {g_nn:+.1f} "
                          f"({'PASS' if g_nn >= 5 else 'FAIL'})")
        print("  gate: " + "; ".join(checks))

    np.savez("logs/p4_g1_verdict.npz",
             **{f"{fam}_c{ci}_{k}": np.array([r[k] for r in rows])
                for (fam, ci), rows in all_rows.items()
                for k in rows[0]})
    print("\nsaved logs/p4_g1_verdict.npz")


if __name__ == "__main__":
    main()
