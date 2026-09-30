"""W1 pre-flight diagnostics.

Two questions that decide how the formal pipeline is modelled:

  Q1  Are the three radars frame-aligned?  (dataset only says "saved independently")
      -> cross-correlate the per-view slow-time energy envelopes, look at the peak lag.

  Q2  Is the ABSOLUTE range usable, i.e. does bin index * 6.35 mm equal the distance
      from that radar to the hand?  If yes we can solve the true nonlinear range
      equations; if not we need a per-view offset b_v as an extra unknown.
      -> look at where the echo actually sits in the range profile, and check whether
         the three radii are geometrically consistent (triangle inequality against the
         known 1.1 m / 0.55 m baselines).
"""
import numpy as np

CACHE = r"D:\uwb_gestures\cache"
M, T = 189, 90
BIN_M = 1.2 / M
VIEWS = ["Left", "Top", "Right"]
P = {"Left": np.array([-0.55, 0.0, 0.0]),
     "Top": np.array([0.0, 0.55, 0.0]),
     "Right": np.array([0.55, 0.0, 0.0])}


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


def envelope(frames):
    return np.abs(analytic(frames.astype(np.float64)))


def main():
    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    subj = np.load(f"{CACHE}/subj_clutterremoved.npy")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), 400, replace=False)

    lags = {("Left", "Top"): [], ("Left", "Right"): [], ("Top", "Right"): []}
    peak_bins = {v: [] for v in VIEWS}
    peak_bins_by_subj = {(v, s): [] for v in VIEWS for s in range(1, 9)}
    energy_frac = {v: [] for v in VIEWS}

    for i in idx:
        unit = np.asarray(X[i])
        envs, pows = [], []
        for v in range(3):
            E = envelope(unit[v])
            envs.append(E)
            pows.append((E ** 2).sum(axis=1))            # slow-time energy
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            pa = pows[a] - pows[a].mean()
            pb = pows[b] - pows[b].mean()
            c = np.correlate(pa, pb, mode="full")
            lag = c.argmax() - (T - 1)
            lags[(VIEWS[a], VIEWS[b])].append(lag)
        for v in range(3):
            E = envs[v]
            w = (E ** 2)
            # bin holding the peak of the strongest frame
            t_star = w.sum(axis=1).argmax()
            peak_bins[VIEWS[v]].append(w[t_star].argmax())
            peak_bins_by_subj[(VIEWS[v], int(subj[i]))].append(w[t_star].argmax())
            # how much of the profile energy sits in the first 20 bins (near field)
            energy_frac[VIEWS[v]].append(w[:, :20].sum() / w.sum())

    print("=== Q1  frame alignment: peak lag of the slow-time energy cross-correlation ===")
    for k, v in lags.items():
        v = np.array(v)
        print(f"  {k[0]:>5s}-{k[1]:<5s}  median={np.median(v):5.1f}  "
              f"|lag|<=1 in {100*np.mean(np.abs(v) <= 1):5.1f}% of samples   "
              f"|lag|<=3 in {100*np.mean(np.abs(v) <= 3):5.1f}%   sd={v.std():.2f}")

    print("\n=== Q2a  where does the echo sit in the range profile? ===")
    for v in VIEWS:
        b = np.array(peak_bins[v])
        print(f"  {v:>5s}: peak bin p10/p50/p90 = {np.percentile(b,10):5.1f} /"
              f" {np.median(b):5.1f} / {np.percentile(b,90):5.1f}"
              f"   -> range {np.median(b)*BIN_M*100:5.1f} cm"
              f"   near-field(<20 bin) energy frac {np.mean(energy_frac[v]):.3f}")

    print("\n  per-subject median peak bin (a stable per-view offset would show up here)")
    print("        " + "".join(f"  HV{s}" for s in range(1, 9)))
    for v in VIEWS:
        row = "".join(f"{np.median(peak_bins_by_subj[(v,s)]):5.0f}" for s in range(1, 9))
        print(f"  {v:>5s} {row}")

    print("\n=== Q2b  geometric consistency of the three absolute radii ===")
    # if r_v = |p - P_v| exactly, the three ranges must be realisable by SOME point p.
    # Solve for p in least squares and report the residual; a large residual means the
    # absolute scale/offset is wrong.
    res_abs, res_off = [], []
    for i in idx[:200]:
        unit = np.asarray(X[i])
        r = []
        for v in range(3):
            w = envelope(unit[v]) ** 2
            t_star = w.sum(axis=1).argmax()
            r.append(w[t_star].argmax() * BIN_M)
        r = np.array(r)
        for tag, store in (("abs", res_abs), ("off", res_off)):
            def resid(p, rr):
                return np.array([np.linalg.norm(p - P[v]) for v in VIEWS]) - rr
            from scipy.optimize import least_squares
            if tag == "abs":
                out = least_squares(resid, np.array([0.0, 0.0, 0.6]), args=(r,))
                store.append(np.abs(out.fun).max())
            else:
                # allow one shared unknown offset b: r_meas = |p-P_v| + b
                def resid2(q):
                    return resid(q[:3], r - q[3])
                out = least_squares(resid2, np.array([0.0, 0.0, 0.6, 0.0]))
                store.append(np.abs(out.fun).max())
    print(f"  no offset      : max|residual| median = {np.median(res_abs)*1000:7.1f} mm")
    print(f"  one shared b   : max|residual| median = {np.median(res_off)*1000:7.1f} mm"
          f"   (b is a free scalar per sample here, so this is a lower bound)")


if __name__ == "__main__":
    main()
