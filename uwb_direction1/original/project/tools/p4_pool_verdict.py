"""Pooled-calibration fix verdict (RESULTS SS43.4(b) follow-up, -> SS43.5).

Compares calib-select on the same two vocabularies under the two calibration
regimes, side by side:

  b2e   c4/c5: leave-2-out, 1 calib subject (SS43.4 failure modes:
        c4 constraint-bound 7/8 folds, c5 interior peak misplaced 6/8);
  rot2  c4/c5: leave-3-out, 2 pooled calib subjects, 5 training subjects
        (encoders retrained; p4_rot2_c4c5.bat, 2026-07-15..18).

Reports per (family, case): s_cal/oracle means, gate metrics, constraint-
bound vs interior fold counts, the s_cal/s_orc ratio (relative underestimate
-- the mechanism-level question), and the ceiling decomposition (how much of
any oracle-gap closure comes from the oracle H itself moving vs H@cal).

Same conventions as tools/p4_g1_verdict.py: grid 1.0..3.0 step .1, base-drop
<=2pts vs same-bank s=1.  Pure numpy post-processing; saves
logs/p4_pool_verdict.npz.

Usage:
  python tools/p4_pool_verdict.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

S_GRID = np.round(np.arange(1.0, 3.0 + 1e-9, 0.1), 2)
BASE_DROP = 0.02
RUNS = [("b2e", "logs/n3_scores_bal_b2e_c{c}_s{t}.npz"),
        ("rot2", "logs/n3_scores_bal_b2e_rot2_c{c}_s{t}.npz")]
CASES = [4, 5]


def _acc(R, yt, nb, s):
    Rs = R.copy()
    Rs[:, :nb] *= s
    return float((Rs.argmin(1) == yt).mean())


def _select(Rb, yb, Rr, yr, nb):
    """Returns (s_cal, s_max_feasible)."""
    cv = [(s, _acc(Rb, yb, nb, s), _acc(Rr, yr, nb, s)) for s in S_GRID]
    feas = [t for t in cv if t[1] >= cv[0][1] - BASE_DROP]
    best = max(feas, key=lambda t: 2 * t[1] * t[2] / (t[1] + t[2] + 1e-9))
    return best[0], max(t[0] for t in feas)


def main():
    saved = {}
    for c in CASES:
        stats = {}
        for fam, fmt in RUNS:
            rows = []
            for ts in range(1, 9):
                f = fmt.format(c=c, t=ts)
                if not os.path.exists(f):
                    continue
                d = np.load(f)
                nb = int(d["n_base"])
                s_cal, s_max = _select(d["R_cal"], d["y_cal"],
                                       d["R_cal_reg"], d["y_cal_reg"], nb)
                s_orc, _ = _select(d["R_base"], d["y_base"],
                                   d["R_reg"], d["y_reg"], nb)
                row = {"ts": ts, "s_cal": s_cal, "s_orc": s_orc,
                       "bound": float(s_cal >= s_max - 1e-9)}
                for tag, s in [("1", 1.0), ("15", 1.5), ("cal", s_cal),
                               ("orc", s_orc)]:
                    ab = _acc(d["R_base"], d["y_base"], nb, s)
                    an = _acc(d["R_reg"], d["y_reg"], nb, s)
                    row[f"b{tag}"], row[f"n{tag}"] = ab, an
                    row[f"H{tag}"] = 2 * ab * an / (ab + an + 1e-9)
                rows.append(row)
            if not rows:
                continue
            stats[fam] = rows
            saved.update({f"{fam}_c{c}_{k}": np.array([r[k] for r in rows])
                          for k in rows[0]})

            def m(k):
                return 100 * float(np.mean([r[k] for r in rows]))
            sc = np.mean([r["s_cal"] for r in rows])
            so = np.mean([r["s_orc"] for r in rows])
            print(f"\n== {fam} c{c} ({len(rows)} folds) ==")
            print(f"  s_cal {sc:.2f}  oracle {so:.2f}  ratio {sc/so:.2f}  "
                  f"|d|<=0.3: "
                  f"{sum(abs(r['s_cal']-r['s_orc']) <= 0.3 for r in rows)}"
                  f"/{len(rows)}  constraint-bound "
                  f"{int(sum(r['bound'] for r in rows))}/{len(rows)}")
            for name, tag in [("s=1", "1"), ("s=1.5", "15"),
                              ("calib-sel", "cal"), ("oracle", "orc")]:
                print(f"  {name:10s} b {m('b'+tag):5.1f}  n {m('n'+tag):5.1f}"
                      f"  H {m('H'+tag):5.1f}")
            print(f"  gates: H gap to oracle {m('Horc')-m('Hcal'):.1f} "
                  f"({'PASS' if m('Horc')-m('Hcal') <= 2 else 'FAIL'}); "
                  f"vs s=1.5 {m('Hcal')-m('H15'):+.1f} "
                  f"({'PASS' if m('H15')-m('Hcal') <= 1 else 'FAIL'}); "
                  f"base drop {m('b1')-m('bcal'):.1f} "
                  f"({'PASS' if m('b1')-m('bcal') <= 2 else 'FAIL'})")
            Hs = [100 * r["Hcal"] for r in rows]
            print(f"  fold H@cal min {min(Hs):.1f} max {max(Hs):.1f} "
                  f"spread {max(Hs)-min(Hs):.1f}")

        if len(stats) == 2:                     # ceiling decomposition
            def mm(fam, k):
                return 100 * float(np.mean([r[k] for r in stats[fam]]))
            d_gap = ((mm("b2e", "Horc") - mm("b2e", "Hcal"))
                     - (mm("rot2", "Horc") - mm("rot2", "Hcal")))
            d_orc = mm("b2e", "Horc") - mm("rot2", "Horc")
            d_cal = mm("rot2", "Hcal") - mm("b2e", "Hcal")
            print(f"\n  c{c} gap closure {d_gap:.1f} = oracle ceiling down "
                  f"{d_orc:.1f} + H@cal up {d_cal:.1f}")

    np.savez("logs/p4_pool_verdict.npz", **saved)
    print("\nsaved logs/p4_pool_verdict.npz")


if __name__ == "__main__":
    main()
