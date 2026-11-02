"""Hand-trajectory reconstruction from the three UWB range tracks.

This is research point 2.1 of `proposal_v2/02_研究内容重构_2x3.md`, W1 scope:
robust range tracking (Viterbi) + a temporally regularised NONLINEAR solve on a
B-spline trajectory + leak-free leave-one-radar-out reprojection.

Why not the per-frame closed form: three ranges against three unknowns is exactly
determined, so the per-frame residual is identically zero and carries no information.
Parameterising the path with K << T control points makes the system over-determined,
which is what makes leave-one-radar-out reprojection a real test.

Geometry.  The dataset paper gives left-right separation 1.1 m and 0.55 m from their
midpoint to the top radar; the range window is 1.2 m over 189 fast-time bins.  The
pre-flight diagnostic (`proposal_v2/pilot/w1_diag2.py`) showed the ABSOLUTE range
scale is usable: solving the three radii for a 3-D point lands at z = 0.45 m in front
of the sensor plane for 93% of samples and is stable across all eight subjects, so no
per-view range offset is introduced here.  Radar positions are still nominal - joint
self-calibration is W2.
"""
import numpy as np
from scipy.optimize import least_squares

M_BINS = 189
RANGE_WINDOW_M = 1.2
BIN_M = RANGE_WINDOW_M / M_BINS          # 6.35 mm
FPS = 20.0
Z_REF = 0.45                             # population standoff, from w1_diag2

# nominal sensor positions, view order Left, Top, Right (matches config.VIEW_ORDER)
RADAR_POS = np.array([[-0.55, 0.0, 0.0],
                      [0.0, 0.55, 0.0],
                      [0.55, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# 1. range tracking
# ---------------------------------------------------------------------------
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


def viterbi_track(frames, max_jump=20, jump_penalty=0.15, return_path=False):
    """Most likely range-bin path through the envelope.

    frames : (T, M) clutter-removed RF frames.
    Returns (r, w): range in metres per frame and a per-frame confidence in [0,1].
    With return_path=True the integer bin path is appended: (r, w, path).

    A greedy per-frame argmax hops between the hand, the forearm and the body when
    their reflections cross; the path prior (bounded jump + linear penalty) keeps the
    track on one scatterer.  max_jump=20 bins = 12.7 cm/frame = 2.5 m/s at 20 fps.

    Why the switch rather than an unconditional third return value: `r` carries the
    sub-bin parabolic refinement, which is meaningful for RANGE but cannot be used to
    index the analytic signal for PHASE, so the readout channels of
    `proposal_v2/17` 3.1 need the raw integer bins.  Appending unconditionally would
    break `tools/w2_calib_pulse.py:55`, which unpacks exactly two values.  The default
    branch below is bit-for-bit what this function returned before the switch existed.
    """
    p = np.abs(_analytic(frames.astype(np.float64))) ** 2
    T, M = p.shape
    em = np.log(p / (p.max(axis=1, keepdims=True) + 1e-30) + 1e-6)   # (T,M)

    score = em[0].copy()
    back = np.zeros((T, M), dtype=np.int16)
    offs = np.arange(-max_jump, max_jump + 1)
    pen = jump_penalty * np.abs(offs)
    for t in range(1, T):
        # cand[j, m] = score[m - offs[j]] - pen[j]
        cand = np.full((len(offs), M), -np.inf)
        for j, o in enumerate(offs):
            if o >= 0:
                cand[j, o:] = score[:M - o] - pen[j] if o > 0 else score - pen[j]
            else:
                cand[j, :M + o] = score[-o:] - pen[j]
        best = cand.argmax(axis=0)
        back[t] = np.arange(M) - offs[best]
        score = cand[best, np.arange(M)] + em[t]

    path = np.zeros(T, dtype=np.int64)
    path[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]

    # sub-bin refinement by parabolic interpolation on the power profile
    r = np.empty(T)
    conf = np.empty(T)
    for t in range(T):
        m = path[t]
        if 0 < m < M - 1:
            a, b, c = p[t, m - 1], p[t, m], p[t, m + 1]
            denom = a - 2 * b + c
            delta = 0.5 * (a - c) / denom if abs(denom) > 1e-30 else 0.0
            delta = float(np.clip(delta, -1.0, 1.0))
        else:
            delta = 0.0
        r[t] = (m + delta) * BIN_M
        conf[t] = p[t, m]
    conf = np.sqrt(conf / (conf.max() + 1e-30))
    return (r, conf, path) if return_path else (r, conf)


# ---------------------------------------------------------------------------
# 2. B-spline trajectory model
# ---------------------------------------------------------------------------
def bspline_basis(T=90, K=15, degree=3):
    """(T, K) cubic B-spline design matrix on a uniform knot vector."""
    from scipy.interpolate import BSpline
    x = np.linspace(0.0, 1.0, T)
    n_inner = K - degree + 1
    knots = np.concatenate([np.zeros(degree),
                            np.linspace(0.0, 1.0, n_inner),
                            np.ones(degree)])
    B = np.empty((T, K))
    for k in range(K):
        c = np.zeros(K)
        c[k] = 1.0
        B[:, k] = BSpline(knots, c, degree, extrapolate=False)(x)
    return np.nan_to_num(B)


def _second_diff(K):
    D = np.zeros((K - 2, K))
    for i in range(K - 2):
        D[i, i:i + 3] = (1.0, -2.0, 1.0)
    return D


def fit_trajectory(r, conf, views=(0, 1, 2), B=None, lam=3.0, mu=0.05,
                   z_ref=Z_REF, pos=RADAR_POS, init=None):
    """Fit p(t) so that ||p(t) - P_v|| matches the measured range tracks.

    r    : (3, T) measured ranges (metres); rows for views not in `views` are ignored
    conf : (3, T) per-frame confidence, used as residual weights
    lam  : smoothness weight on the second difference of the control points
    mu   : weak prior pulling the standoff towards z_ref; it is what keeps the
           two-view (leave-one-out) problem well posed.  It never uses held-out data.

    Returns dict with control points C (K,3), path p (T,3) and the weighted rms.
    """
    T = r.shape[1]
    if B is None:
        B = bspline_basis(T)
    K = B.shape[1]
    D2 = _second_diff(K)
    views = list(views)
    W = conf[views]                                     # (V,T)

    if init is None:
        C0 = np.tile(np.array([0.0, 0.0, z_ref]), (K, 1))
    else:
        C0 = init.copy()

    n_geo = len(views) * T
    n_sm = 3 * (K - 2)
    n_pr = T

    def unpack(x):
        return x.reshape(K, 3)

    def residual(x):
        C = unpack(x)
        p = B @ C                                        # (T,3)
        d = p[None, :, :] - pos[views][:, None, :]       # (V,T,3)
        rng = np.linalg.norm(d, axis=2)                  # (V,T)
        res_geo = (W * (rng - r[views])).ravel()
        res_sm = np.sqrt(lam) * (D2 @ C).ravel()
        res_pr = np.sqrt(mu) * (p[:, 2] - z_ref)
        return np.concatenate([res_geo, res_sm, res_pr])

    def jac(x):
        C = unpack(x)
        p = B @ C
        d = p[None, :, :] - pos[views][:, None, :]
        rng = np.linalg.norm(d, axis=2) + 1e-12
        u = d / rng[:, :, None]                          # (V,T,3) unit vectors
        J = np.zeros((n_geo + n_sm + n_pr, K * 3))
        for vi in range(len(views)):
            block = np.einsum("td,tk->tkd", W[vi][:, None] * u[vi], B)   # (T,K,3)
            J[vi * T:(vi + 1) * T] = block.reshape(T, K * 3)
        for dim in range(3):
            rows = n_geo + dim * (K - 2)
            cols = np.arange(K) * 3 + dim
            J[rows:rows + K - 2][:, cols] = np.sqrt(lam) * D2
        cols_z = np.arange(K) * 3 + 2
        J[n_geo + n_sm:][:, cols_z] = np.sqrt(mu) * B
        return J

    out = least_squares(residual, C0.ravel(), jac=jac, method="lm", max_nfev=200)
    C = unpack(out.x)
    p = B @ C
    d = p[None, :, :] - pos[views][:, None, :]
    rng = np.linalg.norm(d, axis=2)
    rms = float(np.sqrt(np.mean((rng - r[views]) ** 2)))
    return {"C": C, "p": p, "rms": rms, "B": B}


def reproject(C, B, view, pos=RADAR_POS):
    """Predicted range track of `view` from a fitted trajectory."""
    p = B @ C
    return np.linalg.norm(p - pos[view], axis=1)


# ---------------------------------------------------------------------------
# 3. readouts on a reconstructed trajectory
# ---------------------------------------------------------------------------
def readouts(p, fps=FPS):
    x, y = p[:, 0], p[:, 1]
    vx, vy = np.gradient(x) * fps, np.gradient(y) * fps
    return {
        "sd_x": x.std(), "sd_y": y.std(),
        "asym_x": np.mean(vx ** 3), "asym_y": np.mean(vy ** 3),
        "area": 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]),
        "path_len": np.sum(np.hypot(np.diff(x), np.diff(y))),
        "vmax": np.max(np.hypot(vx, vy)),
    }


def dprime(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float((a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-12))
