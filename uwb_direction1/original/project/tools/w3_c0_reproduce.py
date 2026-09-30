"""C0 provenance check: bit-faithful restoration of the RESULTS 49.5 readout.

`scratchpad/rotation_probe.py` was deleted with the rest of `scratchpad/`.  The
original source was recovered verbatim from the Claude session transcript
`6cf85f86-b148-4c5f-b067-d1483c7ac17d.jsonl` (line 125 = the script, line 128 =
its output).  Everything between the RESTORED markers below is that code,
unchanged except for import style and the added verdict check.

This is a CODE-PROVENANCE CHECK, not an experiment.  It must reproduce

    G9 CW vs G10 CCW        d'=0.10  acc=51.62  (per-subject 40-65)
    G11 push vs G12 empty   d'=0.01  acc=49.38  (per-subject 34-65)
    G1 vs G2 (horizontal)   d'=0.69  acc=63.50  (per-subject 30-75)

exactly.  Only after it does is the C0 formula locked (proposal_v2/17 3.4b).

Two properties of the original that must NOT be carried into the new analysis:
  * it uses `[::2]`, i.e. 800 of the 1600 G9/G10 samples;
  * the polarity `pol` is fitted on ALL samples, so "training-free" refers to
    the absence of a learned classifier, not to held-out evaluation.
Both are fine for reproducing a recorded number and unusable for L1.

`--extend` drops only the FIRST of those two: it runs all 1600 samples but
KEEPS the whole-set polarity, purely to show what the sample count changes.  It
is therefore still not an L1 result.  The held-out analysis - polarity from the
training subjects of each LOSO fold - lives in `w3_readout_predict.py` and is
not implemented here.  Report `--extend` numbers as "full-data extension",
never as the historical reproduction and never as L1.

Zero GPU.  Usage: python tools/w3_c0_reproduce.py [--extend]
"""
import argparse
import sys

import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, ".")

CACHE = r"D:\uwb_gestures\cache"
T, M, W, HOP = 90, 189, 24, 6

# recorded values from the session transcript, line 128
RECORDED = {
    "G9 CW vs G10 CCW":      (0.10, 51.62, 40, 65),
    "G11 push vs G12 empty": (0.01, 49.38, 34, 65),
    "G1 vs G2 (horizontal)": (0.69, 63.50, 30, 75),
}
PAIRS = [("G9 CW vs G10 CCW", 8, 9),
         ("G11 push vs G12 empty", 10, 11),
         ("G1 vs G2 (horizontal)", 0, 1)]


# ------------------------- RESTORED from rotation_probe.py -------------------
def st_doppler(cube):                       # cube (n,V,T,M) -> (n,V,nwin) signed centroid
    z = hilbert(cube.astype(np.float64), axis=-1)
    z = z - z.mean(axis=2, keepdims=True)
    out = []
    f = np.fft.fftshift(np.fft.fftfreq(W))
    for st in range(0, T - W + 1, HOP):
        Z = np.fft.fftshift(np.fft.fft(z[:, :, st:st + W], axis=2), axes=2)
        pd = (np.abs(Z) ** 2).sum(-1)
        pd = pd / (pd.sum(-1, keepdims=True) + 1e-30)
        out.append((pd * f).sum(-1))
    return np.stack(out, -1)


def loop_area(v):                            # v (n,V,nw) -> signed area in (v_h, v_v)
    vh = 0.5 * (v[:, 2] - v[:, 0])           # Right minus Left = horizontal velocity
    vv = v[:, 1]                             # Top = vertical velocity
    dh, dv = np.diff(vh, axis=-1), np.diff(vv, axis=-1)
    return (vh[:, :-1] * dv - vv[:, :-1] * dh).sum(-1)
# --------------------------- end RESTORED ------------------------------------


def score(X, idx):
    return np.concatenate([st_doppler(np.asarray(X[idx[i:i + 300]]))
                           for i in range(0, len(idx), 300)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extend", action="store_true",
                    help="also run the full-data extension (all samples, no [::2])")
    args = ap.parse_args()

    X = np.load(f"{CACHE}/X_raw.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_raw.npy")
    s = np.load(f"{CACHE}/subj_raw.npy")

    print("=" * 78)
    print("C0 provenance check - must reproduce RESULTS 49.5 exactly")
    print(f"  input X_raw.npy   W={W} frames  HOP={HOP}  slow-time FFT centroid")
    print("  v_h = 0.5*(Right - Left)   v_v = Top   area = sum[v_h*dv_v - v_v*dv_h]")
    print("=" * 78)

    ok_all = True
    store = {}
    for name, ga, gb in PAIRS:
        idx = np.where(np.isin(y, [ga, gb]))[0][::2]
        V = score(X, idx)
        A = loop_area(V)
        yi, si = y[idx], s[idx]
        a, b = A[yi == ga], A[yi == gb]
        d = abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-30)
        pol = 1 if a.mean() > b.mean() else -1
        acc = ((pol * A > 0) == (yi == ga)).mean()
        per = [(((pol * A > 0) == (yi == ga))[si == u]).mean() for u in np.unique(si)]
        lo, hi = 100 * np.min(per), 100 * np.max(per)

        # the transcript preserves 2 decimals for d'/acc and 0 for the per-subject
        # range, so the check must be made at exactly that precision - a tighter
        # absolute tolerance would compare against digits the record never had
        rd, ra, rlo, rhi = RECORDED[name]
        ok = (round(d, 2) == rd and round(100 * acc, 2) == ra
              and round(lo) == rlo and round(hi) == rhi)
        ok_all &= ok
        print(f"  {name:24s} n={len(idx):4d}  d'={d:5.2f}  acc={100 * acc:5.2f}"
              f"  (per-subject {lo:.0f}-{hi:.0f})   recorded {rd:.2f}/{ra:.2f}/"
              f"{rlo}-{rhi}   {'MATCH' if ok else 'MISMATCH'}")
        store[name] = dict(idx=idx, A=A, y=yi, s=si)

    print("\n  => PROVENANCE " + ("CONFIRMED - C0 formula is locked"
                                  if ok_all else "FAILED - do not lock C0"))

    if args.extend:
        print("\n" + "=" * 78)
        print("FULL-DATA EXTENSION (NOT the historical reproduction)")
        print("=" * 78)
        print("  all samples (no [::2]); polarity still whole-set here, ONLY to show")
        print("  the effect of sample count.  The L1 analysis in 17 must take the")
        print("  polarity from training subjects only - not done in this script.\n")
        for name, ga, gb in PAIRS:
            idx = np.where(np.isin(y, [ga, gb]))[0]
            A = loop_area(score(X, idx))
            yi, si = y[idx], s[idx]
            a, b = A[yi == ga], A[yi == gb]
            d = abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-30)
            pol = 1 if a.mean() > b.mean() else -1
            acc = ((pol * A > 0) == (yi == ga)).mean()
            per = [(((pol * A > 0) == (yi == ga))[si == u]).mean() for u in np.unique(si)]
            print(f"  {name:24s} n={len(idx):4d}  d'={d:5.2f}  acc={100 * acc:5.2f}"
                  f"  (per-subject {100 * np.min(per):.0f}-{100 * np.max(per):.0f})")
            np.savez(f"logs/w3_c0_{ga}_{gb}.npz", idx=idx, area=A, y=yi, subj=si)
        print("\n[saved] logs/w3_c0_<ga>_<gb>.npz  (area per sample, for the L1 analysis)")

    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
