"""W1 pre-flight diagnostics, round 2.

Round 1 problems:
  Q1 used slow-time ENERGY as the alignment feature.  Energy is modulated very
     differently per view (a left-right swipe barely modulates the Top radar), so the
     cross-correlation was ill-conditioned (sd = 30 frames).  Use instead the SPEED
     profile |dr_v/dt| of the range track: the hand is fast/slow at the same instants
     in every view regardless of projection sign.
  Q2b solved 3 ranges for 3 unknowns -> residual is identically zero, no information.
     The informative version is WHERE the solution lands: if the absolute ranges are
     valid, the solved point must sit in front of the radar plane at a sane standoff.
"""
import numpy as np
from scipy.optimize import least_squares

CACHE = r"D:\uwb_gestures\cache"
M, T = 189, 90
BIN_M = 1.2 / M
VIEWS = ["Left", "Top", "Right"]
POS = np.array([[-0.55, 0.0, 0.0],
                [0.0, 0.55, 0.0],
                [0.55, 0.0, 0.0]])


def analytic(x):
    n = x.shape[-1]
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(np.fft.fft(x, axis=-1) * h, axis=-1)


def track(frames, win=8):
    w = np.abs(analytic(frames.astype(np.float64))) ** 2
    pk = w.argmax(axis=1)
    bins = np.arange(M)
    idx = np.empty(frames.shape[0])
    for t in range(frames.shape[0]):
        lo, hi = max(0, pk[t] - win), min(M, pk[t] + win + 1)
        ww = w[t, lo:hi]
        idx[t] = (bins[lo:hi] * ww).sum() / (ww.sum() + 1e-12)
    return idx * BIN_M


def smooth(x, k=5):
    return np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def main():
    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    subj = np.load(f"{CACHE}/subj_clutterremoved.npy")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), 300, replace=False)

    lags = {(0, 1): [], (0, 2): [], (1, 2): []}
    sols, keep_g, keep_s = [], [], []
    for i in idx:
        unit = np.asarray(X[i])
        r = np.stack([smooth(track(unit[v]), 5) for v in range(3)])
        sp = np.abs(np.gradient(r, axis=1))                  # speed profile per view
        sp = (sp - sp.mean(axis=1, keepdims=True)) / (sp.std(axis=1, keepdims=True) + 1e-9)
        for a, b in lags:
            c = np.correlate(sp[a], sp[b], mode="full")
            lags[(a, b)].append(c.argmax() - (T - 1))
        # absolute-range solve at the frame with the strongest echo
        r0 = r[:, T // 2]
        out = least_squares(lambda p: np.linalg.norm(p - POS, axis=1) - r0,
                            np.array([0.0, 0.0, 0.6]))
        sols.append(out.x)
        keep_g.append(int(y[i])); keep_s.append(int(subj[i]))
    sols = np.array(sols)

    print("=== Q1 (redone)  frame alignment via the SPEED profile |dr/dt| ===")
    for k, v in lags.items():
        v = np.array(v)
        print(f"  {VIEWS[k[0]]:>5s}-{VIEWS[k[1]]:<5s} median={np.median(v):5.1f}  "
              f"|lag|<=1: {100*np.mean(np.abs(v)<=1):5.1f}%   "
              f"|lag|<=3: {100*np.mean(np.abs(v)<=3):5.1f}%   "
              f"|lag|<=5: {100*np.mean(np.abs(v)<=5):5.1f}%   sd={v.std():5.2f}")

    print("\n=== Q2 (redone)  where does the absolute-range solution land? ===")
    print("  a valid absolute range scale => z (standoff in front of the radar plane)")
    print("  should be positive and sane (~0.3-1.0 m), x/y within arm's reach\n")
    for j, nm in enumerate("xyz"):
        s = sols[:, j]
        print(f"  {nm}: p10={np.percentile(s,10):6.3f}  median={np.median(s):6.3f}  "
              f"p90={np.percentile(s,90):6.3f}  m")
    print(f"\n  solved standoff z > 0 in {100*np.mean(sols[:,2]>0):.1f}% of samples;"
          f"  0.2<z<1.2 m in {100*np.mean((sols[:,2]>0.2)&(sols[:,2]<1.2)):.1f}%")
    print("  per-subject median (x, y, z) in metres:")
    for s in range(1, 9):
        m = sols[np.array(keep_s) == s]
        if len(m):
            print(f"    HV{s}: ({np.median(m[:,0]):6.3f}, {np.median(m[:,1]):6.3f},"
                  f" {np.median(m[:,2]):6.3f})   n={len(m)}")


if __name__ == "__main__":
    main()
