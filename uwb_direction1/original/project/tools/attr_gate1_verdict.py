"""Gate 1 verdict for the attribute-prior preregistration (2026-07-26).

Pure post-processing of the five arms' dumps -- no GPU, no retraining.

Locked pass conditions (quoted from the preregistration, not re-derived here):
  1. acc_new(G7) at K=1 is at least +5 points over the baseline
  2. base accuracy drops <= 2 points and rejU drops <= 3 points
  3. direction consistency: >= 6/8 folds positive
  negative control: G10/G11 acc_new moves within +-2 points; if they rise too, the
  gain is "more positives" rather than "the attribute prior" -> CONFOUNDED, not a pass

Class ids in REG case 1: base [1,4,5,7,8,11] -> 0..5, register [6,9,10] -> 6,7,8,
so bank id 6 = G7 (the prior target), 7 = G10, 8 = G11 (controls).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from uwb_osr.registration import conformal_report                    # noqa: E402

TARGET, CTRL = 6, (7, 8)
NAMES = {6: "G7 RL+UD (target)", 7: "G10 anti-clock (ctrl)", 8: "G11 push (ctrl)"}
ARMS = [("g1_aoff", "A / baseline"), ("g1_apa", "A / P-A synth positives"),
        ("g1_boff", "B / baseline"), ("g1_bpb", "B / P-B blend"),
        ("g1_bpc", "B / P-C prior only (K=0)")]


def fold_metrics(path, s_route=1.5, alpha=0.1):
    d = np.load(path)
    nb = int(d["n_base"])
    def route(R):
        r = np.array(R, dtype=np.float64)
        r[:, :nb] *= s_route                    # frozen paper-3 routing prior
        return r.argmin(1)
    pred_b, pred_r = route(d["R_base"]), route(d["R_reg"])
    yb, yr = d["y_base"], d["y_reg"]
    out = {"acc_base": 100 * np.mean(pred_b == yb)}
    for c in sorted(set(int(v) for v in yr)):
        m = yr == c
        out[f"acc_{c}"] = 100 * np.mean(pred_r[m] == c)
    out["acc_new"] = float(np.mean([out[f"acc_{c}"] for c in sorted(set(int(v) for v in yr))]))
    rep = conformal_report({k: d[k] for k in d.files}, alpha, "hybrid")
    out["rejU"] = 100 * rep["reject_unknown"]
    out["cov_new"] = 100 * rep["cov_new"]
    a, b = out["acc_base"], out["acc_new"]
    out["H"] = 0.0 if a + b == 0 else 2 * a * b / (a + b)
    return out


def collect(tag, folds, s_route):
    rows = []
    for ts in folds:
        p = f"logs/n3_scores_bal_{tag}_c1_s{ts}.npz"
        if os.path.exists(p):
            rows.append(fold_metrics(p, s_route))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s-route", type=float, default=1.5)
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(1, 9)))
    args = ap.parse_args()

    data = {}
    print(f"routing scale s={args.s_route} (frozen paper-3 recipe)\n")
    print(f"{'arm':28s}{'folds':>6s}{'base':>8s}{'acc_new':>9s}"
          f"{'G7':>8s}{'G10':>8s}{'G11':>8s}{'rejU':>8s}{'H':>8s}")
    for tag, name in ARMS:
        rows = collect(tag, args.folds, args.s_route)
        if not rows:
            print(f"{name:28s}{'-- missing --':>6s}")
            continue
        data[tag] = rows
        m = lambda k: np.mean([r.get(k, np.nan) for r in rows])       # noqa: E731
        print(f"{name:28s}{len(rows):6d}{m('acc_base'):8.2f}{m('acc_new'):9.2f}"
              f"{m('acc_6'):8.2f}{m('acc_7'):8.2f}{m('acc_8'):8.2f}"
              f"{m('rejU'):8.2f}{m('H'):8.2f}")

    print("\n=== Gate 1 conditions (per variant, against its own method's baseline) ===")
    for var, base, label in (("g1_apa", "g1_aoff", "P-A"),
                             ("g1_bpb", "g1_boff", "P-B"),
                             ("g1_bpc", "g1_boff", "P-C (K=0, reported only)")):
        if var not in data or base not in data:
            print(f"\n{label}: missing dumps")
            continue
        v, b = data[var], data[base]
        n = min(len(v), len(b))
        d7 = np.array([v[i]["acc_6"] - b[i]["acc_6"] for i in range(n)])
        dbase = np.mean([v[i]["acc_base"] - b[i]["acc_base"] for i in range(n)])
        drej = np.mean([v[i]["rejU"] - b[i]["rejU"] for i in range(n)])
        dctrl = {c: np.mean([v[i][f"acc_{c}"] - b[i][f"acc_{c}"] for i in range(n)])
                 for c in CTRL}
        c1 = d7.mean() >= 5
        c2 = (dbase >= -2) and (drej >= -3)
        c3 = int((d7 > 0).sum()) >= 6
        ctrl_ok = all(abs(x) <= 2 for x in dctrl.values())
        print(f"\n{label}   (n={n} folds)")
        print(f"  1. acc_new(G7) delta = {d7.mean():+6.2f}  (need >= +5)      "
              f"{'PASS' if c1 else 'FAIL'}")
        print(f"     per fold: {' '.join(f'{x:+.0f}' for x in d7)}")
        print(f"  2. base delta {dbase:+.2f} (need >= -2), rejU delta {drej:+.2f} "
              f"(need >= -3)   {'PASS' if c2 else 'FAIL'}")
        print(f"  3. direction consistency {int((d7 > 0).sum())}/{n} positive "
              f"(need >= 6)     {'PASS' if c3 else 'FAIL'}")
        print(f"  negative control: G10 {dctrl[7]:+.2f}, G11 {dctrl[8]:+.2f} "
              f"(need |d| <= 2)   {'OK' if ctrl_ok else 'CONFOUNDED'}")
        verdict = ("PASS" if (c1 and c2 and c3 and ctrl_ok)
                   else "CONFOUNDED" if not ctrl_ok else "FAIL")
        print(f"  --> {label}: {verdict}")


if __name__ == "__main__":
    main()
