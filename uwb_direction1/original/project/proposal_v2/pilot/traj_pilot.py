"""Pilot: can we recover a 2-D hand trajectory from the three range tracks?

Zero training. Pure signal processing on the cached ClutterRemoved frames.

Steps
  1. per view, per frame: envelope over fast time -> range centroid r_v(t)
  2. differential ranges dr_v(t) = r_v(t) - r_v(0)   (kills the unknown window offset)
  3. linearised trilateration: dr_v = u_v . dp   with u_v the unit vector radar->hand
     at a nominal standoff z0.  3 equations, 3 unknowns (dx, dy, dz) per frame.
  4. checks
     A. translation: sign of net dx separates G1 (L->R) from G2 (R->L)?  likewise dy for G3/G4
     B. chirality:  signed loop area of the reconstructed (x,y) path separates G9 (CW) from G10 (CCW)?
        -> compare against the d'=0.10 that the Doppler-centroid route got (RESULTS 49.5)
     C. leave-one-radar-out reprojection: fit a smooth path from 2 views, predict the 3rd
        view's range track, report the residual in mm.
"""
import sys
import numpy as np

CACHE = r"D:\uwb_gestures\cache"
RANGE_WINDOW_M = 1.2          # dataset paper: radar range adjusted to 1.2 m -> 189 bins
M = 189
BIN_M = RANGE_WINDOW_M / M    # 6.35 mm
FPS = 20.0

# radar positions (metres) in the sensor plane, from the dataset description:
# left-right separation 1.1 m, top radar 0.55 m from their midpoint
P = {"L": np.array([-0.55, 0.0, 0.0]),
     "T": np.array([0.0, 0.55, 0.0]),
     "R": np.array([0.55, 0.0, 0.0])}
VIEW_ORDER = ["L", "T", "R"]


def analytic(x):
    """Analytic signal along the last (fast-time) axis."""
    n = x.shape[-1]
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(np.fft.fft(x, axis=-1) * h, axis=-1)


def range_track(frames, mode="centroid", win=8):
    """frames: (T, M) -> r(t) in metres (relative to the window start) + per-frame SNR proxy."""
    env = np.abs(analytic(frames.astype(np.float64)))      # (T, M)
    w = env ** 2
    peak = w.max(axis=1)
    if mode == "argmax":
        idx = w.argmax(axis=1).astype(np.float64)
    else:
        # centroid inside a window around the per-frame peak: robust to a second
        # scatterer (body/arm) sitting elsewhere in the range profile
        idx = np.empty(frames.shape[0])
        pk = w.argmax(axis=1)
        bins = np.arange(M)
        for t in range(frames.shape[0]):
            lo, hi = max(0, pk[t] - win), min(M, pk[t] + win + 1)
            ww = w[t, lo:hi]
            idx[t] = (bins[lo:hi] * ww).sum() / (ww.sum() + 1e-12)
    return idx * BIN_M, peak


def solve_dp(dr, z0):
    """dr: (3, T) differential ranges in view order L,T,R -> dp: (T, 3) displacement."""
    p0 = np.array([0.0, 0.0, z0])
    U = np.stack([(p0 - P[v]) / np.linalg.norm(p0 - P[v]) for v in VIEW_ORDER])  # (3,3)
    return np.linalg.solve(U, dr).T           # exactly determined: 3 eq, 3 unknowns


def smooth(x, k=5):
    ker = np.ones(k) / k
    return np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), ker, mode="valid")


def signed_area(x, y):
    """shoelace: >0 counter-clockwise in a right-handed (x right, y up) frame."""
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def dprime(a, b):
    return (a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-12)


def main():
    z0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    mode = sys.argv[2] if len(sys.argv) > 2 else "centroid"
    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    subj = np.load(f"{CACHE}/subj_clutterremoved.npy")
    print(f"X{X.shape}  z0={z0} m  bin={BIN_M*1000:.2f} mm  mode={mode}")

    want = [0, 1, 2, 3, 8, 9]          # L->R, R->L, U->D, D->U, CW, CCW
    per_g = 40                          # samples per gesture per subject cap
    feats = {g: {"dx": [], "dy": [], "area": [], "resid": [], "subj": []} for g in want}

    for g in want:
        for s in range(1, 9):
            idx = np.where((y == g) & (subj == s))[0][:per_g]
            for i in idx:
                unit = np.asarray(X[i])                    # (3, 90, 189)
                rr, snr = [], []
                for v in range(3):
                    r, pk = range_track(unit[v], mode=mode)
                    rr.append(smooth(r, 5))
                    snr.append(pk)
                rr = np.stack(rr)                          # (3, T)
                dr = rr - rr[:, :1]
                dp = solve_dp(dr, z0)                      # (T, 3)
                x, yy = dp[:, 0], dp[:, 1]
                feats[g]["dx"].append(x[-1] - x[0])
                feats[g]["dy"].append(yy[-1] - yy[0])
                feats[g]["area"].append(signed_area(x - x.mean(), yy - yy.mean()))
                # leave-one-out reprojection: rebuild dp from views {T,R}+smoothness,
                # predict L. 2 eq / 3 unknowns per frame -> constrain dz to the
                # 3-view solution's low-pass (crude but offset-free)
                U = np.stack([(np.array([0, 0, z0]) - P[v]) / np.linalg.norm(np.array([0, 0, z0]) - P[v])
                              for v in VIEW_ORDER])
                dz = smooth(dp[:, 2], 15)
                A = U[1:, :2]                              # T,R rows, xy columns
                rhs = dr[1:] - np.outer(U[1:, 2], dz)
                pxy = np.linalg.lstsq(A, rhs, rcond=None)[0]     # (2, T)
                pred_L = U[0, 0] * pxy[0] + U[0, 1] * pxy[1] + U[0, 2] * dz
                feats[g]["resid"].append(np.sqrt(np.mean((pred_L - dr[0]) ** 2)) * 1000)
                feats[g]["subj"].append(s)

    for g in want:
        for k in ("dx", "dy", "area", "resid"):
            feats[g][k] = np.asarray(feats[g][k])
        feats[g]["subj"] = np.asarray(feats[g]["subj"])

    print("\n--- A. net displacement (mm, mean +- sd over samples) ---")
    for g in want:
        print(f"  G{g+1:<3} dx {feats[g]['dx'].mean()*1000:8.1f} +- {feats[g]['dx'].std()*1000:6.1f}"
              f"   dy {feats[g]['dy'].mean()*1000:8.1f} +- {feats[g]['dy'].std()*1000:6.1f}"
              f"   n={len(feats[g]['dx'])}")

    def acc_sign(ga, gb, key):
        a, b = feats[ga][key], feats[gb][key]
        # decision: a should be > b; accuracy of the zero-threshold sign rule
        return 100 * (np.mean(a > 0) + np.mean(b < 0)) / 2, dprime(a, b)

    print("\n--- sign-rule accuracy / d' (zero threshold, no training) ---")
    for (ga, gb, key, name) in [(0, 1, "dx", "G1 vs G2 horizontal"),
                                (3, 2, "dy", "G4 vs G3 vertical"),
                                (9, 8, "area", "G10(CCW) vs G9(CW) chirality")]:
        acc, d = acc_sign(ga, gb, key)
        print(f"  {name:32s} acc={acc:6.2f}%   d'={d:6.3f}")

    print("\n--- per-subject chirality d' (the RESULTS 49.5 baseline was d'=0.10) ---")
    for s in range(1, 9):
        a = feats[9]["area"][feats[9]["subj"] == s]
        b = feats[8]["area"][feats[8]["subj"] == s]
        acc = 100 * (np.mean(a > 0) + np.mean(b < 0)) / 2
        print(f"  HV{s}: d'={dprime(a, b):6.3f}  acc={acc:6.2f}%")

    print("\n--- C. leave-one-radar-out reprojection residual (mm RMS on dr_L) ---")
    for g in want:
        r = feats[g]["resid"]
        print(f"  G{g+1:<3} {np.median(r):7.2f} (median)   {r.mean():7.2f} +- {r.std():6.2f}")


if __name__ == "__main__":
    main()
