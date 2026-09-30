# -*- coding: utf-8 -*-
"""Precompute per-frame Range-Doppler features for the frame-labeled FMCW dataset.

Raw member npz is (N,100,3,32,64) float32 = 2.46 GB. This turns each into a
compact per-frame feature cache (N,100,32,32) float16 = ~205 MB, so all users
fit in memory for LOSO experiments.

Pipeline per frame (3 antennas x 32 chirps x 64 samples):
  remove per-chirp DC -> range FFT (64 -> 32 bins, drop DC) -> Doppler FFT over
  32 chirps -> fftshift -> |.| -> mean over antennas -> log1p
No normalization here: causal (running) normalization happens at train time,
per the streaming-leakage lesson in proposal_v2/27.

Usage: python tools/fmcw_prep.py [user3_e1 user4_e1 ...]   (default: all npz found)
"""
import os
import sys
import time

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "fmcw_frame_labeled")
CACHE = os.path.join(BASE, "cache")


def rd_features(x):
    """x: (n,100,3,32,64) float32 -> (n,100,32,32) float32 log-magnitude RD."""
    x = x - x.mean(axis=-1, keepdims=True)
    r = np.fft.rfft(x, axis=-1)[..., 1:33]      # range bins 1..32, drop DC
    d = np.fft.fft(r, axis=-2)                  # Doppler over 32 chirps
    d = np.fft.fftshift(d, axes=-2)
    return np.log1p(np.abs(d).mean(axis=2)).astype(np.float32)


def prep(name, block=40):
    src = os.path.join(BASE, name + ".npz")
    dst = os.path.join(CACHE, name + "_rd.npz")
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
    F = np.empty((n, 100, 32, 32), dtype=np.float16)
    for i in range(0, n, block):
        F[i:i + block] = rd_features(np.asarray(X[i:i + block])).astype(np.float16)
    np.savez_compressed(dst, feats=F, targets=np.asarray(T).astype(np.int16))
    print(f"{name}: {n} recs -> {dst} ({os.path.getsize(dst)/1e6:.0f} MB) "
          f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    names = sys.argv[1:]
    if not names:
        names = sorted(f[:-4] for f in os.listdir(BASE) if f.endswith(".npz"))
    for nm in names:
        prep(nm)
