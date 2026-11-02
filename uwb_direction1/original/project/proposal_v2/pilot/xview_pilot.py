"""Pilot X1/X2 - feasibility of the "imagine an unobserved viewpoint" direction.

Two questions, both zero-training, pure numpy/scipy on the cached real frames.

X1  Can the echo at a held-out radar be PREDICTED from the other two?
    Ladder of predictors for the held-out view v, evaluated in the SIGNAL domain
    (the whole 90x189 range-profile movie), not in label space:

      B0    zero / class-mean frame           - the "no information" floor
      Bcopy copy an observed view's movie     - "just reuse the nearest sensor"
      Bphys fit the trajectory from the 2 observed views, render v (physics only)
      Bphys+w  same, with the width-matched template (logs/w3_width_gap.npz)
      Boracle fit from ALL THREE views, render v  - upper bound of the physics route

    Metrics: envelope correlation, per-frame peak-bin error in mm, scale-invariant
    NMSE.  Bphys > Bcopy is the falsifiable claim: geometry, not sensor proximity,
    is what carries a viewpoint across.

X2  Does the MOST INFORMATIVE view depend on the gesture?
    Per gesture, the cost (in mm of held-out range error) of dropping each view.
    If the ranking moves across gestures, choosing where to look has headroom;
    if one view always wins, active sensing is pointless here.

Usage:  python proposal_v2/pilot/xview_pilot.py [--reps 2] [--out logs/xview_pilot.npz]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from uwb_osr.forward import clutter_filter, load_template, render
from uwb_osr.trajectory import (BIN_M, M_BINS, RADAR_POS, bspline_basis,
                                fit_trajectory, reproject, viterbi_track)

CACHE = r"D:\uwb_gestures\cache"
VIEWS = ["Left", "Top", "Right"]
GAINS = (0.876, 1.043, 1.081)          # measured, proposal_v2/13 section 2
NBINS = M_BINS


# ---------------------------------------------------------------------------
def analytic_env(x):
    """|analytic signal| along the last axis."""
    n = x.shape[-1]
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(np.fft.fft(x, axis=-1) * h, axis=-1))


def compare(pred, real):
    """Signal-domain agreement between two (T, M) range-profile movies."""
    ep, er = analytic_env(pred.astype(np.float64)), analytic_env(real.astype(np.float64))
    a, b = ep.ravel() - ep.mean(), er.ravel() - er.mean()
    corr = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    # scale-invariant NMSE on the envelope (one global gain fitted by LS)
    g = float(ep.ravel() @ er.ravel() / (ep.ravel() @ ep.ravel() + 1e-30))
    nmse = float(np.sum((g * ep - er) ** 2) / (np.sum(er ** 2) + 1e-30))
    # per-frame peak location error, in mm
    pk = np.abs(ep.argmax(axis=1) - er.argmax(axis=1)) * BIN_M * 1000.0
    return corr, nmse, float(np.median(pk))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2, help="samples per (gesture, subject)")
    ap.add_argument("--out", default="logs/xview_pilot.npz")
    args = ap.parse_args()

    X = np.load(os.path.join(CACHE, "X_clutterremoved.npy"), mmap_mode="r")
    y = np.load(os.path.join(CACHE, "y_clutterremoved.npy"))
    subj = np.load(os.path.join(CACHE, "subj_clutterremoved.npy"))
    print(f"[data] X{X.shape}  {len(np.unique(y))} gestures  {len(np.unique(subj))} subjects")

    tmpl_narrow = load_template()                       # (full, centre)
    d = np.load("logs/w3_width_gap.npz")
    tmpl_wide = (d["t_wide"], NBINS // 2)
    print(f"[tmpl] narrow (as calibrated) and width-matched loaded")

    rng = np.random.default_rng(0)
    picks = []
    for g in np.unique(y):
        for s in np.unique(subj):
            idx = np.where((y == g) & (subj == s))[0]
            if len(idx) == 0:
                continue
            picks.extend(rng.choice(idx, min(args.reps, len(idx)), replace=False).tolist())
    picks = np.array(sorted(picks))
    print(f"[pick] {len(picks)} samples\n")

    B = bspline_basis(90)
    ARMS = ["B0zero", "Bcopy", "Bphys", "Bphys_w", "Boracle"]
    rec = {k: {a: [] for a in ARMS} for k in ("corr", "nmse", "peak")}
    rng_err = {"loo": [], "full": []}          # range-track error, mm
    meta = []

    t0 = time.time()
    for n, i in enumerate(picks):
        Xi = np.asarray(X[i], dtype=np.float64)         # (3, 90, 189)
        r = np.empty((3, 90)); conf = np.empty((3, 90))
        for v in range(3):
            r[v], conf[v] = viterbi_track(Xi[v])

        fit_all = fit_trajectory(r, conf, views=(0, 1, 2), B=B)
        ren_all = clutter_filter(render(fit_all["p"], template=tmpl_narrow, gains=GAINS))

        for v in range(3):
            others = [u for u in range(3) if u != v]
            fit = fit_trajectory(r, conf, views=others, B=B)

            # --- range level -------------------------------------------------
            e_loo = np.abs(reproject(fit["C"], B, v) - r[v]) * 1000.0
            e_full = np.abs(reproject(fit_all["C"], B, v) - r[v]) * 1000.0
            rng_err["loo"].append(np.median(e_loo))
            rng_err["full"].append(np.median(e_full))

            # --- signal level ------------------------------------------------
            real = Xi[v]
            preds = {
                "B0zero": np.zeros_like(real),
                "Bcopy": Xi[others[0]],
                "Bphys": clutter_filter(render(fit["p"], template=tmpl_narrow,
                                               gains=GAINS))[v],
                "Bphys_w": clutter_filter(render(fit["p"], template=tmpl_wide,
                                                gains=GAINS))[v],
                "Boracle": ren_all[v],
            }
            for a, p in preds.items():
                c, nm, pk = compare(p, real)
                rec["corr"][a].append(c)
                rec["nmse"][a].append(nm)
                rec["peak"][a].append(pk)
            meta.append((int(y[i]), int(subj[i]), v))

        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(picks)}  ({time.time()-t0:.0f}s)")

    meta = np.array(meta)
    for k in rec:
        for a in rec[k]:
            rec[k][a] = np.array(rec[k][a])
    for k in rng_err:
        rng_err[k] = np.array(rng_err[k])

    # ---------------- X1 report ------------------------------------------
    print("\n" + "=" * 74)
    print("X1  signal-domain predictability of a HELD-OUT view  "
          f"(n={len(meta)} view-instances)")
    print("=" * 74)
    print(f"{'arm':>10s} {'env corr':>18s} {'NMSE(scale-inv)':>18s} {'peak err mm':>18s}")
    for a in ARMS:
        c, nm, pk = rec["corr"][a], rec["nmse"][a], rec["peak"][a]
        print(f"{a:>10s} {c.mean():8.3f} +-{c.std():5.3f} "
              f"{np.median(nm):11.3f}       {np.median(pk):11.1f}")

    print(f"\n  range-track error of the HELD-OUT view (median over frames, mm):")
    print(f"    leave-one-view-out fit : median {np.median(rng_err['loo']):6.1f}  "
          f"mean {rng_err['loo'].mean():6.1f}")
    print(f"    all-three fit (oracle) : median {np.median(rng_err['full']):6.1f}  "
          f"mean {rng_err['full'].mean():6.1f}")

    # paired: does physics beat "copy the neighbouring sensor"?
    dc = rec["corr"]["Bphys_w"] - rec["corr"]["Bcopy"]
    print(f"\n  paired Bphys_w - Bcopy (env corr): mean {dc.mean():+.3f}, "
          f"win rate {np.mean(dc > 0)*100:.1f}%")
    dw = rec["corr"]["Bphys_w"] - rec["corr"]["Bphys"]
    print(f"  paired Bphys_w - Bphys  (width fix): mean {dw.mean():+.3f}, "
          f"win rate {np.mean(dw > 0)*100:.1f}%")

    # ---------------- X2 report ------------------------------------------
    print("\n" + "=" * 74)
    print("X2  is the most informative view gesture-dependent?")
    print("    per gesture: median held-out range error (mm) when each view is dropped")
    print("=" * 74)
    print(f"{'gesture':>8s} {'drop Left':>11s} {'drop Top':>11s} {'drop Right':>11s} "
          f"{'worst':>8s} {'best':>8s} {'spread':>8s}")
    per_g = {}
    for g in np.unique(meta[:, 0]):
        row = []
        for v in range(3):
            m = (meta[:, 0] == g) & (meta[:, 2] == v)
            row.append(np.median(rng_err["loo"][m]))
        per_g[int(g)] = row
        order = np.argsort(row)
        print(f"{'G'+str(int(g)+1):>8s} {row[0]:11.1f} {row[1]:11.1f} {row[2]:11.1f} "
              f"{VIEWS[order[-1]]:>8s} {VIEWS[order[0]]:>8s} "
              f"{row[order[-1]]-row[order[0]]:8.1f}")

    worst = [int(np.argmax(per_g[g])) for g in sorted(per_g)]
    print(f"\n  'which view hurts most to lose' across the 12 gestures: "
          f"{[VIEWS[w] for w in worst]}")
    print(f"  distinct answers: {len(set(worst))} of 3  -> "
          f"{'view choice IS gesture-dependent' if len(set(worst))>1 else 'one view always wins'}")

    # ---------------- X2b: adaptive vs fixed 2-view subset -----------------
    # Each sample contributes three held-out errors (one per dropped view).
    # A fixed policy always drops the same view; an oracle adaptive policy drops
    # the cheapest one for THAT sample.  The gap is the headroom for choosing
    # where to look, in millimetres - no classification accuracy involved.
    n_s = len(meta) // 3
    E = rng_err["loo"].reshape(n_s, 3)              # (sample, dropped view)
    print("\n" + "=" * 74)
    print("X2b headroom for CHOOSING which view to spend: median held-out error, mm")
    print("=" * 74)
    for v in range(3):
        print(f"  fixed policy 'always drop {VIEWS[v]:>5s}' : {np.median(E[:, v]):7.1f}")
    best_fixed = int(np.argmin(np.median(E, axis=0)))
    print(f"  best fixed policy               : {np.median(E[:, best_fixed]):7.1f}"
          f"  (drop {VIEWS[best_fixed]})")
    print(f"  oracle adaptive (per sample)    : {np.median(E.min(axis=1)):7.1f}")
    print(f"  worst-case adaptive             : {np.median(E.max(axis=1)):7.1f}")
    red = 1 - np.median(E.min(axis=1)) / np.median(E[:, best_fixed])
    print(f"  -> adaptive beats the best fixed policy by {red*100:.1f}% of the error")
    choice = E.argmin(axis=1)
    frac = [float(np.mean(choice == v)) for v in range(3)]
    print(f"  per-sample best choice spread   : "
          f"Left {frac[0]*100:.0f}% / Top {frac[1]*100:.0f}% / Right {frac[2]*100:.0f}%")

    # ---------------- per-view difficulty of imagining ---------------------
    print("\n  how hard is each view to IMAGINE (env corr, Bphys_w):")
    for v in range(3):
        m = meta[:, 2] == v
        print(f"    held-out {VIEWS[v]:>5s}: corr {rec['corr']['Bphys_w'][m].mean():.3f}"
              f"   oracle {rec['corr']['Boracle'][m].mean():.3f}"
              f"   copy {rec['corr']['Bcopy'][m].mean():.3f}")

    os.makedirs("logs", exist_ok=True)
    np.savez(args.out, meta=meta,
             **{f"corr_{a}": rec["corr"][a] for a in ARMS},
             **{f"nmse_{a}": rec["nmse"][a] for a in ARMS},
             **{f"peak_{a}": rec["peak"][a] for a in ARMS},
             rng_loo=rng_err["loo"], rng_full=rng_err["full"])
    print(f"\nsaved -> {args.out}   ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
