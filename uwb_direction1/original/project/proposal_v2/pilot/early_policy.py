"""Research point 2.1, simulated - two triggers instead of one.

The textbook halting rule has a single job: decide WHEN I know the answer.  In an open
vocabulary a second, independent job exists: decide when to stop hoping.  This script
simulates a policy with both triggers and measures what the second one buys.

Policy, evaluated frame by frame on a causal score s_t (default: max softmax):

    commit(argmax)   if  s_t >= tau_hi
    abstain          if  t >= t_min  and  s_t <= tau_lo     ("this is not converging")
    abstain          if  t == T                             (timeout)
    wait             otherwise

Three outcomes per sample, and they mean different things depending on the truth:

              known gesture                unseen gesture
  commit      right or wrong answer        FALSE COMMIT   (the harmful case)
  abstain     missed interaction           correct refusal

The comparison of record is at MATCHED false-commit rate: given that a deployment can
tolerate at most X% false commits on unseen gestures, which policy answers known
gestures faster, and how many of them does it still answer at all?

Usage:  python proposal_v2/pilot/early_policy.py logs/early_deep_e15_c1.npz
        python proposal_v2/pilot/early_policy.py "logs/early_deep_e15_c*.npz"
"""
import argparse
import glob
import sys

import numpy as np

FPS = 20.0


def run_policy(score, pred, y, known_m, tau_hi, tau_lo, t_min, T):
    """Vectorised simulation.  Returns per-sample (outcome, frame, prediction).

    outcome: 1 = commit, 0 = abstain.
    """
    hi = score >= tau_hi
    lo = np.zeros_like(hi)
    if tau_lo is not None:
        lo[:, t_min:] = score[:, t_min:] <= tau_lo

    first_hi = np.where(hi.any(1), hi.argmax(1), T + 1)
    first_lo = np.where(lo.any(1), lo.argmax(1), T + 1)

    commit = first_hi <= np.minimum(first_lo, T - 1)
    frame = np.where(commit, first_hi, np.minimum(first_lo, T - 1))
    p = pred[np.arange(len(y)), np.clip(frame, 0, T - 1)]
    return commit, frame, p


def summarise(commit, frame, p, y, known_m):
    n_k, n_u = known_m.sum(), (~known_m).sum()
    ck = commit & known_m
    cu = commit & ~known_m
    lat_k = (frame[ck] + 1) / FPS * 1000
    return dict(
        answered=ck.sum() / n_k * 100,
        acc_answered=(p[ck] == y[ck]).mean() * 100 if ck.sum() else np.nan,
        # accuracy over ALL known samples: abstaining on a known gesture is a miss
        acc_overall=(p[ck] == y[ck]).sum() / n_k * 100,
        lat_med=np.median(lat_k) if ck.sum() else np.nan,
        lat_p90=np.percentile(lat_k, 90) if ck.sum() else np.nan,
        false_commit=cu.sum() / n_u * 100,
        refuse=(~commit & ~known_m).sum() / n_u * 100,
    )


def frontier(rows, key_x="false_commit", key_y="lat_med", lower_better=True):
    """Pareto-optimal subset: minimise x and y (and maximise answered)."""
    out = []
    for r in rows:
        dominated = any(
            (o[key_x] <= r[key_x] and o[key_y] <= r[key_y]
             and o["acc_overall"] >= r["acc_overall"]
             and (o[key_x] < r[key_x] or o[key_y] < r[key_y]
                  or o["acc_overall"] > r["acc_overall"]))
            for o in rows)
        if not dominated:
            out.append(r)
    return sorted(out, key=lambda r: r[key_x])


def analyse(f):
    d = np.load(f, allow_pickle=True)
    prob, y, cls = d["prob"], d["y"], np.asarray(d["cls"])
    known_m = np.isin(y, cls)
    T = prob.shape[1]
    score = prob.max(-1)
    pred = cls[prob.argmax(-1)]

    print("\n" + "#" * 78)
    print(f"# {f}   known {known_m.sum()} / unseen {(~known_m).sum()}   "
          f"vocab {len(cls)}   unseen classes {list(d['unknown'])}")
    print("#" * 78)

    # ---- arm 1: single trigger (the textbook rule) -----------------------
    rows1 = []
    for th in np.arange(0.50, 0.999, 0.01):
        c, fr, p = run_policy(score, pred, y, known_m, th, None, 0, T)
        r = summarise(c, fr, p, y, known_m); r["tau_hi"], r["tau_lo"] = th, None
        rows1.append(r)

    # ---- arm 2: two triggers --------------------------------------------
    rows2 = []
    for th in np.arange(0.50, 0.999, 0.02):
        for tl in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
            if tl >= th:
                continue
            for tmin in [10, 20, 30, 40]:
                c, fr, p = run_policy(score, pred, y, known_m, th, tl, tmin, T)
                r = summarise(c, fr, p, y, known_m)
                r["tau_hi"], r["tau_lo"], r["t_min"] = th, tl, tmin
                rows2.append(r)

    print("\nMATCHED-BUDGET COMPARISON: at most B% false commits on unseen gestures")
    print(f"{'budget':>7s} | {'single trigger':^38s} | {'two triggers':^38s}")
    print(f"{'B%':>7s} | {'answered%':>10s} {'acc all%':>9s} {'lat ms':>8s} {'p90':>7s}"
          f" | {'answered%':>10s} {'acc all%':>9s} {'lat ms':>8s} {'p90':>7s}")
    for B in [10, 20, 30, 40, 50, 60]:
        a = [r for r in rows1 if r["false_commit"] <= B]
        b = [r for r in rows2 if r["false_commit"] <= B]
        # among feasible, pick the one that answers the most known gestures correctly,
        # breaking ties by latency
        ba = max(a, key=lambda r: (round(r["acc_overall"], 1), -r["lat_med"])) if a else None
        bb = max(b, key=lambda r: (round(r["acc_overall"], 1), -r["lat_med"])) if b else None
        def fmt(r):
            if r is None:
                return f"{'infeasible':>10s} {'-':>9s} {'-':>8s} {'-':>7s}"
            return (f"{r['answered']:10.1f} {r['acc_overall']:9.1f} "
                    f"{r['lat_med']:8.0f} {r['lat_p90']:7.0f}")
        print(f"{B:7d} | {fmt(ba)} | {fmt(bb)}")

    print("\n  the same table, read as 'what does the second trigger buy':")
    for B in [10, 20, 30, 40, 50, 60]:
        a = [r for r in rows1 if r["false_commit"] <= B]
        b = [r for r in rows2 if r["false_commit"] <= B]
        if not a or not b:
            print(f"    budget {B:2d}%: single trigger INFEASIBLE at this budget")
            continue
        ba = max(a, key=lambda r: (round(r["acc_overall"], 1), -r["lat_med"]))
        bb = max(b, key=lambda r: (round(r["acc_overall"], 1), -r["lat_med"]))
        print(f"    budget {B:2d}%:  correct answers on known {ba['acc_overall']:.1f}"
              f" -> {bb['acc_overall']:.1f} ({bb['acc_overall']-ba['acc_overall']:+.1f})"
              f"   median latency {ba['lat_med']:.0f} -> {bb['lat_med']:.0f} ms"
              f" ({bb['lat_med']-ba['lat_med']:+.0f})")

    # ---- how low can each arm push the false-commit rate at all? ---------
    print(f"\n  lowest reachable false-commit rate:"
          f"  single {min(r['false_commit'] for r in rows1):.1f}%"
          f"   two-trigger {min(r['false_commit'] for r in rows2):.1f}%")
    best1 = min(rows1, key=lambda r: r["false_commit"])
    best2 = min(rows2, key=lambda r: r["false_commit"])
    print(f"    at that point, correct answers on known gestures:"
          f"  single {best1['acc_overall']:.1f}%   two-trigger {best2['acc_overall']:.1f}%")
    return rows1, rows2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    args = ap.parse_args()
    files = sorted(glob.glob(args.pattern))
    if not files:
        sys.exit(f"no files match {args.pattern}")
    for f in files:
        analyse(f)


if __name__ == "__main__":
    main()
