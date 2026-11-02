"""Latency / risk frontier on the CLEAN streaming protocol.

Consumes the dumps from `early_stream.py`.  Unlike `early_policy.py`, latency here is
measured from the true gesture onset, a commit before the onset counts as a false alarm
on idle radar, and the model has a legitimate idle outlet.

Two arms:
  single   commit at the first frame with argmax != idle and max softmax >= tau_hi
  double   same, plus an explicit give-up trigger: if by `t_wait` frames after the
           first non-idle frame the score has not reached tau_hi, abstain and stop
           looking, rather than continuing to accumulate towards a wrong label

Reported at matched false-commit budget, which is the only comparison that means
anything when the two arms trade risk for latency differently.
"""
import argparse
import glob
import sys

import numpy as np

FPS = 20.0


def analyse(f):
    d = np.load(f, allow_pickle=True)
    prob, onset, y = d["prob"], d["onset"], d["y"]
    gcls = np.asarray(d["gcls"])
    idle = len(gcls)
    N, W, _ = prob.shape
    known_m = np.isin(y, gcls)
    am = prob.argmax(-1)
    msp = prob.max(-1)

    print("\n" + "#" * 78)
    print(f"# {f}   {N} streams   known {known_m.sum()}  unseen {(~known_m).sum()}"
          f"   |gestures| {len(gcls)}   unseen {list(d['unknown'])}")
    print("#" * 78)

    # where do unseen gestures end up when the model is left alone?
    at_end = am[:, -1]
    late = am[np.arange(N), np.clip(onset + 60, 0, W - 1)]
    print(f"\n  routing of UNSEEN gestures 3 s after onset: "
          f"idle {np.mean(late[~known_m] == idle)*100:.1f}%  "
          f"a real gesture {np.mean(late[~known_m] != idle)*100:.1f}%")
    print(f"  same for KNOWN gestures:                    "
          f"idle {np.mean(late[known_m] == idle)*100:.1f}%  "
          f"a real gesture {np.mean(late[known_m] != idle)*100:.1f}%")

    rows = []
    for tau in np.arange(0.50, 0.999, 0.01):
        for twait in [None, 20, 30, 40, 60]:
            fire = (am != idle) & (msp >= tau)
            first = np.where(fire.any(1), fire.argmax(1), W + 1)
            if twait is not None:
                # give up if nothing has fired by twait frames after the first non-idle
                ni = am != idle
                first_ni = np.where(ni.any(1), ni.argmax(1), W + 1)
                deadline = first_ni + twait
                first = np.where(first <= deadline, first, W + 1)
            committed = first < W
            fa = committed & (first < onset)
            det = committed & (first >= onset) & known_m
            fc = committed & (first >= onset) & ~known_m
            lat = (first[det] - onset[det]) / FPS * 1000
            acc = ((gcls[am[det, first[det]]] == y[det]).mean() * 100
                   if det.sum() else np.nan)
            rows.append(dict(tau=tau, twait=twait,
                             det=det.sum() / known_m.sum() * 100,
                             acc=acc,
                             correct_all=(gcls[am[det, first[det]]] == y[det]).sum()
                             / known_m.sum() * 100,
                             lat=np.median(lat) if det.sum() else np.nan,
                             lat90=np.percentile(lat, 90) if det.sum() else np.nan,
                             fc=fc.sum() / (~known_m).sum() * 100,
                             fa=fa.mean() * 100))

    print(f"\n  MATCHED-BUDGET FRONTIER (best = most known gestures answered correctly,"
          f" ties broken by latency)")
    print(f"{'budget':>7s} | {'single trigger':^40s} | {'+ give-up trigger':^40s}")
    print(f"{'fc%':>7s} | {'answered%':>10s} {'correct all%':>13s} {'lat ms':>7s} {'p90':>6s}"
          f" | {'answered%':>10s} {'correct all%':>13s} {'lat ms':>7s} {'p90':>6s}")
    for B in [1, 2, 5, 10, 20, 40]:
        a = [r for r in rows if r["twait"] is None and r["fc"] <= B]
        b = [r for r in rows if r["twait"] is not None and r["fc"] <= B]
        key = lambda r: (round(r["correct_all"], 1), -r["lat"] if r["lat"] == r["lat"] else 0)
        ba = max(a, key=key) if a else None
        bb = max(b, key=key) if b else None
        fmt = lambda r: (f"{'infeasible':>10s} {'-':>13s} {'-':>7s} {'-':>6s}" if r is None
                         else f"{r['det']:10.1f} {r['correct_all']:13.1f} "
                              f"{r['lat']:7.0f} {r['lat90']:6.0f}")
        print(f"{B:7d} | {fmt(ba)} | {fmt(bb)}")

    fmin = min(r["fc"] for r in rows if r["twait"] is None)
    fmin2 = min(r["fc"] for r in rows)
    print(f"\n  lowest reachable false-commit: single {fmin:.1f}%   with give-up {fmin2:.1f}%")
    # a deployment-shaped operating point
    ok = [r for r in rows if r["fc"] <= 5 and r["det"] >= 80]
    if ok:
        best = min(ok, key=lambda r: r["lat"])
        print(f"  example operating point (<=5% false commit, >=80% of known gestures "
              f"answered):\n    tau={best['tau']:.2f} give-up={best['twait']}  "
              f"answered {best['det']:.1f}%  accuracy@commit {best['acc']:.1f}%  "
              f"median latency {best['lat']:.0f} ms (p90 {best['lat90']:.0f})  "
              f"false alarm on idle {best['fa']:.2f}%")
    return rows


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
