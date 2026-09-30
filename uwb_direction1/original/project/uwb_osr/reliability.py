"""P4 G3 per-view reliability (SS5.4; c-lite pre-registration v0.2,
docs/superpowers/specs/2026-07-19-g3-h4-clite-preregistration.md).

Locked definitions -- do not tune on test results:

    m_v            raw per-masked-view reconstruction MSE (recon_error
                   per_view=True)
    e_v = m_v / (median(m_cal,v) + 1e-8)     calib-median normalisation
                   (keeps e_v positive; never invert a z-score)
    u_v = 1 / (e_v + 1e-8)
    w_v = clip(u_v, p5(u_cal,v), p95(u_cal,v)), then normalised over views
    all-view alert:  e_v > p99(e_cal,v) for EVERY view simultaneously

Aggregation over the per-view distance tensor D (N, C, V) with per-class
per-view scales S (C, V):

    r_k = sum_v w_v * (d_kv / sigma_kv)

uniform-rv is the same formula at w_v = 1/V (the aggregation-change
control); oracle-mask zeroes the known-polluted view's weight and
renormalises (EVALUATOR-side ceiling only, never a method).
"""
import numpy as np

EPS = 1e-8


class ReliabilityCalib:
    """Frozen calibration state for the locked reliability definition."""

    def __init__(self, M_cal: np.ndarray):
        M_cal = np.asarray(M_cal, float)
        if M_cal.ndim != 2:
            raise ValueError("M_cal must be (N_cal, V) raw per-view MSEs")
        self.med = np.median(M_cal, axis=0) + EPS
        E = M_cal / self.med
        U = 1.0 / (E + EPS)
        self.u_lo = np.percentile(U, 5, axis=0)
        self.u_hi = np.percentile(U, 95, axis=0)
        self.e_p99 = np.percentile(E, 99, axis=0)

    def weights(self, M: np.ndarray):
        """(N, V) raw MSEs -> (W (N, V) normalised weights, alert (N,))."""
        E = np.asarray(M, float) / self.med
        U = 1.0 / (E + EPS)
        W = np.clip(U, self.u_lo[None, :], self.u_hi[None, :])
        W = W / W.sum(axis=1, keepdims=True)
        alert = (E > self.e_p99[None, :]).all(axis=1)
        return W, alert


def per_view_r(D: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Per-view normalised distances z (N, C, V) = d_kv / sigma_kv."""
    return np.asarray(D, float) / (np.asarray(S, float)[None] + 1e-6)


def aggregate(D: np.ndarray, S: np.ndarray, W: np.ndarray) -> np.ndarray:
    """R (N, C) = sum_v W[n, v] * (D[n, k, v] / S[k, v])."""
    return (W[:, None, :] * per_view_r(D, S)).sum(axis=2)


def uniform_weights(n: int, V: int) -> np.ndarray:
    return np.full((n, V), 1.0 / V)


def oracle_mask_weights(n: int, V: int, polluted_view: int) -> np.ndarray:
    """Evaluator-only ceiling: zero the known-polluted view, renormalise."""
    W = np.full((n, V), 1.0 / max(V - 1, 1))
    W[:, polluted_view] = 0.0
    return W
