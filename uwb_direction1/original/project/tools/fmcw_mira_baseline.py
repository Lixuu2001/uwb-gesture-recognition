# -*- coding: utf-8 -*-
"""MIRA-style rule-based onset detector as the mandatory baseline.

Reimplements the onset stage described in Seifi et al. 2025 (IJMWT, same
dataset): per-frame total energy over range/Doppler/antennas -> normalize ->
Savitzky-Golay smoothing -> first local peak above threshold = gesture onset.
The published method then crops a FIXED window after the onset and classifies
it with a rule system; here we only score the ONSET stage, which is the part
our streaming detector must beat.

Reported on the same streaming protocol as tools/fmcw_stream.py so the numbers
are comparable: onset MAE (frames), detection rate within +-4 frames, and false
alarms per minute on the concatenated stream.

Usage: python tools/fmcw_mira_baseline.py --users 3 4 5 6 7 8
"""
import argparse
import os

import numpy as np
from scipy.signal import savgol_filter, find_peaks

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "fmcw_frame_labeled")
CACHE = os.path.join(BASE, "cache")
FPS = 33.0


def load_user(u):
    z = np.load(os.path.join(CACHE, "user%d_e1_rd.npz" % u))
    return z["feats"].astype(np.float32), z["targets"].astype(np.int64)


def energy_curve(F):
    """F: (T,32,32) log-magnitude RD -> (T,) normalized smoothed energy.
    Doppler bins near zero velocity are dropped so static clutter does not
    dominate (the MTI role in the original pipeline)."""
    e = F.copy()
    e[:, 14:18, :] = 0.0                       # suppress near-zero Doppler
    v = e.reshape(len(e), -1).sum(1)
    v = v - np.median(v)
    s = v.std()
    if s > 0:
        v = v / s
    win = 9 if len(v) >= 9 else (len(v) // 2) * 2 + 1
    return savgol_filter(v, win, 2) if win >= 5 else v


def radial_distance_curve(F, amp_q=0.90):
    """MIRA's actual anchor rule: the gesture anchor is the frame of MINIMUM
    radial distance to the radar, with an amplitude threshold applied first
    (Seifi 2025).  Per frame: zero out near-zero Doppler (clutter), take the
    moving-target energy per range bin, apply an amplitude gate, and report the
    nearest range bin that passes.  Frames with no bin above the gate get the
    max range (i.e. no target).
    Returns (T,) distance in range-bin units."""
    e = F.copy()
    e[:, 14:18, :] = 0.0
    per_range = e.sum(axis=1)                  # (T, 32) energy per range bin
    gate = np.quantile(per_range, amp_q)
    dist = np.full(len(F), per_range.shape[1] - 1, dtype=np.float64)
    for t in range(len(F)):
        hits = np.nonzero(per_range[t] > gate)[0]
        if len(hits):
            dist[t] = hits[0]                  # nearest passing range bin
    win = 9 if len(dist) >= 9 else (len(dist) // 2) * 2 + 1
    return savgol_filter(dist, win, 2) if win >= 5 else dist


def detect_onsets(F, thr, refractory=25):
    v = energy_curve(F)
    pk, _ = find_peaks(v, height=thr, distance=refractory)
    return list(pk), v


def events_from_labels(seq):
    ev, s = [], 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[s]:
            if seq[s] != 0:
                ev.append((int(seq[s]), s, i - 1))
            s = i
    return ev


def score(onsets, gt, n_frames, tol=4):
    """Class-agnostic onset scoring (the MIRA onset stage predicts no class)."""
    used, errs = set(), []
    for s in sorted(onsets):
        best, bi = None, None
        for j, (_, s2, _) in enumerate(gt):
            if j in used:
                continue
            d = abs(s - s2)
            if best is None or d < best:
                best, bi = d, j
        if bi is not None and best <= tol:
            used.add(bi)
            errs.append(best)
    tp = len(errs)
    prec = tp / max(1, len(onsets))
    rec = tp / max(1, len(gt))
    minutes = n_frames / FPS / 60
    return dict(detect_rate=rec, precision=prec,
                f1=2 * prec * rec / max(1e-9, prec + rec),
                onset_mae=float(np.mean(errs)) if errs else float("nan"),
                fa_per_min=(len(onsets) - tp) / max(1e-9, minutes),
                n_pred=len(onsets), n_gt=len(gt))



def per_recording_mode(users, seed=0):
    """Calibration check: run the SAME detector under the original paper's
    assumption -- exactly one gesture per 100-frame recording, so take the single
    strongest peak.  If our reimplementation is faithful this should land near
    the published 90.13 pct onset accuracy (+-4 frame tolerance); if it does not,
    the streaming degradation reported above cannot be attributed to the loss of
    the one-gesture prior."""
    print("\n--- calibration: per-recording mode (one peak per 100 frames) ---")
    accs = []
    for u in users:
        f, y = load_user(u)
        ok = 0
        n = 0
        for r in range(len(f)):
            gt = events_from_labels(y[r])
            if not gt:
                continue
            d = radial_distance_curve(f[r])
            pred = int(np.argmin(d))           # MIRA anchor = min radial distance
            ok += abs(pred - gt[0][1]) <= 4
            n += 1
        accs.append(ok / max(1, n))
        print("  user%d: %.3f (%d recs)" % (u, accs[-1], n))
    print("  mean onset accuracy: %.4f  (published MIRA: 0.9013)" % np.mean(accs))
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8])
    ap.add_argument("--thrs", type=float, nargs="+",
                    default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--stream-recs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("MIRA-style onset baseline, users %s" % args.users)
    per_thr = {t: [] for t in args.thrs}
    for u in args.users:
        f, y = load_user(u)
        idx = rng.permutation(len(f))[:args.stream_recs]
        F = f[idx].reshape(-1, 32, 32)
        Y = y[idx].reshape(-1)
        gt = events_from_labels(Y)
        for t in args.thrs:
            on, _ = detect_onsets(F, t)
            per_thr[t].append(score(on, gt, len(F)))

    keys = ["detect_rate", "precision", "f1", "onset_mae", "fa_per_min"]
    print("\n%6s " % "thr" + " ".join("%12s" % k for k in keys))
    for t in args.thrs:
        m = {k: float(np.nanmean([r[k] for r in per_thr[t]])) for k in keys}
        print("%6.2f " % t + " ".join("%12.3f" % m[k] for k in keys))
    per_recording_mode(args.users, args.seed)
    print("\nNote: class-agnostic onsets (the rule stage predicts no class), so "
          "detect_rate/f1 are an UPPER bound relative to the class-aware "
          "event F1 reported by tools/fmcw_stream.py")


if __name__ == "__main__":
    main()
