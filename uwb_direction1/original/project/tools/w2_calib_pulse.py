"""W2 step 0: calibrate the forward model against the real echoes.

Before writing a renderer we measure, from the data itself:
  1. the fast-time carrier (should land near 8.745 GHz if bin = 1.2 m / 189)
  2. the pulse template g(tau): the average complex range profile around the tracked
     scatterer, aligned to sub-bin precision -> this becomes the renderer's kernel,
     so the forward model is data-driven rather than an assumed Gaussian
  3. how the echo amplitude falls off with range (the 1/R^2 assumption)
  4. how wide the target is in range (one scatterer or several?)
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from uwb_osr.trajectory import viterbi_track, BIN_M, M_BINS       # noqa: E402

CACHE = r"D:\uwb_gestures\cache"
C = 2.998e8
FS = C / (2 * BIN_M)            # fast-time sampling rate implied by the range window


def main():
    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    Xr = np.load(f"{CACHE}/X_raw.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    rng = np.random.default_rng(0)
    idx = rng.choice(np.where(y < 11)[0], 120, replace=False)

    print(f"range bin = {BIN_M*1000:.2f} mm  ->  implied fast-time fs = {FS/1e9:.2f} GHz "
          f"(Nyquist {FS/2e9:.2f} GHz)")

    # ---- 1. carrier ------------------------------------------------------
    spec = np.zeros(M_BINS // 2 + 1)
    for i in idx[:40]:
        f = np.abs(np.fft.rfft(np.asarray(X[i]).reshape(-1, M_BINS), axis=-1))
        spec += f.mean(axis=0)
    fr = np.fft.rfftfreq(M_BINS, d=1 / FS)
    k = spec.argmax()
    half = spec.max() * 10 ** (-10 / 20)
    band = fr[spec >= half]
    print(f"\n1. fast-time spectrum peak at {fr[k]/1e9:.3f} GHz "
          f"(dataset says centre 8.745 GHz)")
    print(f"   -10 dB band {band[0]/1e9:.2f}-{band[-1]/1e9:.2f} GHz "
          f"= {(band[-1]-band[0])/1e9:.2f} GHz (dataset says 1.5 GHz)")

    # ---- 2. pulse template ----------------------------------------------
    W = 24
    acc = np.zeros(2 * W + 1, dtype=complex)
    n = 0
    amps, rngs, widths = [], [], []
    for i in idx:
        unit = np.asarray(X[i])
        for v in range(3):
            r, conf = viterbi_track(unit[v])
            E = np.abs(_analytic(unit[v]))
            for t in range(0, 90, 5):
                m = int(round(r[t] / BIN_M))
                if m - W < 0 or m + W + 1 > M_BINS or conf[t] < 0.5:
                    continue
                seg = _analytic(unit[v][t])[m - W:m + W + 1]
                # align phase so templates add coherently
                acc += seg * np.exp(-1j * np.angle(seg[W]))
                n += 1
                e = E[t, m - W:m + W + 1]
                amps.append(e.max()); rngs.append(r[t])
                # -6 dB width of the envelope, in bins
                widths.append(np.sum(e >= 0.5 * e.max()))
    tmpl = acc / n
    env = np.abs(tmpl)
    env /= env.max()
    print(f"\n2. pulse template from {n} snapshots")
    print("   envelope (normalised), bins -8..+8 around the peak:")
    print("   " + " ".join(f"{env[W+j]:.2f}" for j in range(-8, 9)))
    supp = np.where(env > 0.1)[0]
    print(f"   support >10%: {len(supp)} bins = {len(supp)*BIN_M*100:.1f} cm;"
          f"  -6 dB width median {np.median(widths):.1f} bins"
          f" = {np.median(widths)*BIN_M*1000:.0f} mm")

    # ---- 3. amplitude vs range ------------------------------------------
    amps, rngs = np.array(amps), np.array(rngs)
    ok = (rngs > 0.25) & (rngs < 1.1) & (amps > 0)
    p = np.polyfit(np.log(rngs[ok]), np.log(amps[ok]), 1)
    print(f"\n3. amplitude ~ R^{p[0]:.2f}   (a point scatterer two-way gives R^-2)")

    # ---- 4. raw vs clutter-removed --------------------------------------
    i = idx[0]
    a = np.asarray(Xr[i])[0]
    b = np.asarray(X[i])[0]
    alpha = 0.9
    c = np.zeros_like(a)
    cm = a[0].copy()
    for t in range(a.shape[0]):
        cm = alpha * cm + (1 - alpha) * a[t]
        c[t] = a[t] - cm
    err = np.linalg.norm(c - b) / (np.linalg.norm(b) + 1e-12)
    corr = np.corrcoef(c.ravel(), b.ravel())[0, 1]
    print(f"\n4. reproducing ClutterRemoved from Raw with the loopback filter alpha=0.9:"
          f"\n   relative error {err:.3f}, correlation {corr:.4f}"
          f"   -> the renderer can emulate the same preprocessing")

    np.savez("logs/w2_pulse.npz", template=tmpl, env=env, W=W,
             carrier=fr[k], amp_exp=p[0], width_bins=np.median(widths))
    print("\nsaved -> logs/w2_pulse.npz")


def _analytic(x):
    n = x.shape[-1]
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(np.fft.fft(x, axis=-1) * h, axis=-1)


if __name__ == "__main__":
    main()
