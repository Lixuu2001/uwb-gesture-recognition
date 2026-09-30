# -*- coding: utf-8 -*-
"""Per-frame GEOMETRIC feature cache (radial distance, velocity, azimuth,
elevation, magnitude) from the raw FMCW ADC cube.

Motivation (proposal_v2/28 sec 5ter): the dataset's labels are defined by the
frame of MINIMUM RADIAL DISTANCE, computed by the authors from the raw ADC with
the full antenna chain.  Our 32x32 RD feature cache (antenna-averaged, log
magnitude) provably does not carry that quantity: four attempts failed to
recover even the deterministic anchor rule from it.  So the representation may
be missing exactly what the labels encode, which is a candidate mechanism for
the cross-user recall collapse.  This script extracts MIRA's own feature set so
that hypothesis can be tested by adding these channels.

BGT60TR13C: 1 Tx, 3 Rx in an L-shape.  With the antenna order used here, Rx0/Rx1
are separated along one axis and Rx1/Rx2 along the other, so the phase
difference of each pair gives one angular coordinate each.  The absolute
azimuth/elevation convention does not matter for our purpose: we only need a
per-frame quantity that varies with the hand's angular position.

Output per user: (n, 100, 6) float32
    0 radial distance (range bin of peak moving-target energy)
    1 range centroid  (energy-weighted, sub-bin)
    2 radial velocity (Doppler centroid, bins, signed)
    3 angle A         (phase diff Rx0-Rx1, radians)
    4 angle B         (phase diff Rx1-Rx2, radians)
    5 log magnitude   (total moving-target energy)

Usage: python tools/fmcw_prep_geom.py [user3_e1 ...]   (default: all *.npz)
"""
import os
import sys
import time

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "fmcw_frame_labeled")
CACHE = os.path.join(BASE, "cache")
NEAR_DC = slice(14, 18)          # near-zero Doppler bins after fftshift (32 chirps)


def geom_features(x):
    """x: (n,100,3,32,64) float32 -> (n,100,6) float32."""
    x = x - x.mean(axis=-1, keepdims=True)
    r = np.fft.rfft(x, axis=-1)[..., 1:33]          # (n,T,3,32,32) range bins
    d = np.fft.fftshift(np.fft.fft(r, axis=-2), axes=-2)   # Doppler
    d = d.copy()
    d[..., NEAR_DC, :] = 0                          # MTI
    mag = np.abs(d)                                 # (n,T,3,nD,nR)
    p = mag.mean(axis=2)                            # antenna-averaged (n,T,nD,nR)

    rng_prof = p.sum(axis=-2)                       # (n,T,nR) energy vs range
    dop_prof = p.sum(axis=-1)                       # (n,T,nD) energy vs Doppler
    nR, nD = rng_prof.shape[-1], dop_prof.shape[-1]

    peak_r = rng_prof.argmax(-1).astype(np.float32)
    w = rng_prof / (rng_prof.sum(-1, keepdims=True) + 1e-12)
    cen_r = (w * np.arange(nR)).sum(-1).astype(np.float32)
    wv = dop_prof / (dop_prof.sum(-1, keepdims=True) + 1e-12)
    cen_v = ((wv * (np.arange(nD) - nD // 2)).sum(-1)).astype(np.float32)

    # angle: phase of the dominant moving-target cell, per antenna pair
    n_, T_ = mag.shape[0], mag.shape[1]
    flat = p.reshape(n_, T_, -1).argmax(-1)
    di, ri = np.divmod(flat, nR)
    ii, tt = np.meshgrid(np.arange(n_), np.arange(T_), indexing="ij")
    c0 = d[ii, tt, 0, di, ri]
    c1 = d[ii, tt, 1, di, ri]
    c2 = d[ii, tt, 2, di, ri]
    angA = np.angle(c0 * np.conj(c1)).astype(np.float32)
    angB = np.angle(c1 * np.conj(c2)).astype(np.float32)
    logm = np.log1p(rng_prof.sum(-1)).astype(np.float32)

    return np.stack([peak_r, cen_r, cen_v, angA, angB, logm], axis=-1)


def prep(name, block=25):
    src = os.path.join(BASE, name + ".npz")
    dst = os.path.join(CACHE, name + "_geom.npz")
    if os.path.exists(dst):
        print("skip (cached):", name)
        return
    if not os.path.exists(src):
        print("missing:", src)
        return
    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()
    z = np.load(src, mmap_mode="r")
    X, T = z["inputs"], z["targets"]
    n = X.shape[0]
    G = np.empty((n, 100, 6), dtype=np.float32)
    for i in range(0, n, block):
        G[i:i + block] = geom_features(np.asarray(X[i:i + block]))
    np.savez_compressed(dst, geom=G, targets=np.asarray(T).astype(np.int16))
    print("%s: %d recs -> %s in %.0fs" % (name, n, dst, time.time() - t0))


if __name__ == "__main__":
    names = sys.argv[1:]
    if not names:
        names = sorted(f[:-4] for f in os.listdir(BASE) if f.endswith(".npz"))
    for nm in names:
        prep(nm)
