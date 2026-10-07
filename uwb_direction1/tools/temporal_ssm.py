"""Established temporal Gram descriptor; no labels, fitting or novelty claim."""
import numpy as np


def grams(raw):
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 4 or x.shape[1:] != (3, 90, 189):
        raise ValueError('Expected N,3,90,189 real RF')
    h = np.zeros(189); h[0] = 1; h[1:95] = 2
    envelope = np.abs(np.fft.ifft(np.fft.fft(x, axis=-1)*h, axis=-1))
    envelope -= envelope.mean(axis=-2, keepdims=True)
    envelope /= np.maximum(np.linalg.norm(envelope, axis=-1, keepdims=True), 1e-12)
    z = envelope.reshape(len(x), 3, 18, 5, 189).mean(axis=-2)
    return z @ z.swapaxes(-1, -2)


def describe(g, ids, seed=20261007):
    i, j = np.triu_indices(18)
    lag = np.stack([np.diagonal(g, offset=k, axis1=-2, axis2=-1).mean(-1)
                    for k in range(18)], axis=-1)
    shuffled = np.empty_like(g)
    for n, sample in enumerate(ids):
        p = np.random.default_rng(np.random.SeedSequence([seed, int(sample)])).permutation(18)
        shuffled[n] = g[n][:, p][:, :, p]
    return dict(ordered=g[..., i, j].reshape(len(g), 513),
                lag_only=lag[..., j-i].reshape(len(g), 513),
                shuffled=shuffled[..., i, j].reshape(len(g), 513))


def features(raw, ids, seed=20261007):
    return describe(grams(raw), ids, seed)
