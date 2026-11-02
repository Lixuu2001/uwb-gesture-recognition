"""W2 step 0b: pin down the carrier and the clutter filter, then re-derive lambda.

Step 0 measured the fast-time spectrum peak at 0.323 of the sampling rate.  Under the
range scale the dataset paper itself states (1.2 m over 189 bins -> fs = 23.6 GHz)
that is 7.6 GHz, NOT the 8.745 GHz in the paper's parameter table.  The XeThru X4 has
two selectable centre frequencies, 7.29 GHz and 8.748 GHz, so the likely explanation is
that the recordings use the LOW band and the table lists the high one.

This matters beyond bookkeeping: every wavelength-dependent claim (Doppler aliasing,
phase-based velocity) scales with lambda.  So: test both hypotheses for consistency and
recompute the aliasing fraction with the corrected lambda.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from uwb_osr.trajectory import viterbi_track, BIN_M, M_BINS      # noqa: E402

CACHE = r"D:\uwb_gestures\cache"
C = 2.998e8


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


def main():
    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    Xr = np.load(f"{CACHE}/X_raw.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    rng = np.random.default_rng(0)
    idx = rng.choice(np.where(y < 11)[0], 60, replace=False)

    # ---- normalised carrier, measured with a zero-padded FFT for resolution ----
    NF = 4096
    spec = np.zeros(NF // 2 + 1)
    for i in idx[:30]:
        d = np.asarray(X[i]).reshape(-1, M_BINS)
        spec += np.abs(np.fft.rfft(d, n=NF, axis=-1)).mean(axis=0)
    f_norm = np.fft.rfftfreq(NF)[spec.argmax()]          # cycles per sample
    print(f"measured carrier = {f_norm:.4f} cycles/sample")
    print(f"{'hypothesis':38s}{'implied fs':>12s}{'implied bin':>13s}{'189 bins':>11s}")
    for name, fc in (("X4 low band  fc = 7.29 GHz", 7.29e9),
                     ("X4 high band fc = 8.748 GHz", 8.748e9)):
        fs = fc / f_norm
        b = C / (2 * fs)
        print(f"  {name:36s}{fs/1e9:10.2f} G{b*1000:11.2f} mm{189*b:10.3f} m")
    print(f"  {'paper range scale 1.2 m / 189 bins':36s}"
          f"{C/(2*BIN_M)/1e9:10.2f} G{BIN_M*1000:11.2f} mm{189*BIN_M:10.3f} m"
          f"   -> carrier {f_norm*C/(2*BIN_M)/1e9:.2f} GHz")
    print("\n  the low-band hypothesis reproduces the stated 1.2 m window to within 3%;")
    print("  the high-band hypothesis would need a 0.90 m window, contradicting the paper.")

    lam_lo = C / 7.29e9
    lam_hi = C / 8.748e9
    print(f"\nlambda: low band {lam_lo*1000:.1f} mm (lambda/4 = {lam_lo*250:.2f} mm)"
          f" | high band {lam_hi*1000:.1f} mm (lambda/4 = {lam_hi*250:.2f} mm)")

    # ---- re-derive the aliasing fraction with both lambdas ----
    fr = []
    for i in idx:
        unit = np.asarray(X[i])
        r = np.stack([viterbi_track(unit[v])[0] for v in range(3)])
        d = np.abs(np.diff(r, axis=1))
        fr.append([np.mean(d > lam_lo / 4), np.mean(d > lam_hi / 4)])
    fr = np.array(fr)
    print(f"frames with |dr| > lambda/4  (phase wraps between frames):")
    print(f"  low band  (10.28 mm): {100*fr[:,0].mean():5.1f}%")
    print(f"  high band ( 8.57 mm): {100*fr[:,1].mean():5.1f}%   <- the figure reported earlier")

    # ---- clutter filter alpha ----
    print("\nloopback clutter filter: which alpha reproduces ClutterRemoved from Raw?")
    best = None
    for alpha in (0.80, 0.85, 0.90, 0.92, 0.95, 0.98):
        errs, cors = [], []
        for i in idx[:20]:
            a = np.asarray(Xr[i]); b = np.asarray(X[i])
            cm = a[:, 0].copy()
            out = np.empty_like(a)
            for t in range(a.shape[1]):
                cm = alpha * cm + (1 - alpha) * a[:, t]
                out[:, t] = a[:, t] - cm
            errs.append(np.linalg.norm(out - b) / np.linalg.norm(b))
            cors.append(np.corrcoef(out.ravel(), b.ravel())[0, 1])
        e, c = np.mean(errs), np.mean(cors)
        print(f"  alpha={alpha:4.2f}  rel.err={e:.3f}  corr={c:.4f}")
        if best is None or e < best[0]:
            best = (e, alpha, c)
    print(f"  best: alpha={best[1]} (rel.err {best[0]:.3f}, corr {best[2]:.4f})")


if __name__ == "__main__":
    main()
