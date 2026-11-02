"""Forward echo model: trajectory -> three-view RF frames.

Research point 2.2.  The renderer is DATA-DRIVEN: the pulse template, the amplitude
falloff exponent, the per-view gain, the noise level and the clutter-filter constant
are all measured from the dataset (`tools/w2_calib_pulse.py`, `tools/w2_calib2.py`)
rather than assumed, so synthetic frames can be compared against real ones on equal
terms.

Physics.  A scatterer at range R delays the RF pulse by tau = 2R/c.  Delaying the
measured RF template by a fractional number of samples via an FFT phase ramp moves the
envelope AND advances the carrier phase by -2*pi*f_c*tau, which is exactly the
range-dependent phase, so no separate carrier bookkeeping is needed.

Calibration constants measured on UWB-Gestures (ClutterRemoved / Raw caches):
  * fast-time carrier   0.3203 cycles/sample; with the paper's 1.2 m / 189-bin range
    scale that is 7.56 GHz, i.e. the XeThru X4 LOW band (7.29 GHz, lambda = 41.1 mm),
    not the 8.748 GHz printed in the dataset paper's parameter table
  * amplitude ~ R^-2.41 (a two-way point scatterer would give R^-2)
  * clutter removal = loopback filter with alpha ~ 0.80 (the paper states 0.9;
    0.80 reproduces the released ClutterRemoved files to 13.8% relative error)
"""
import numpy as np

from uwb_osr.trajectory import BIN_M, M_BINS, RADAR_POS

C_LIGHT = 2.998e8
ALPHA_CLUTTER = 0.80
AMP_EXP = -2.41


# ---------------------------------------------------------------------------
def load_template(path="logs/w2_pulse.npz", n_bins=M_BINS):
    """Measured RF pulse template, zero-padded to the full range profile length."""
    d = np.load(path)
    t = np.real(d["template"])
    t = t / np.abs(t).max()
    W = int(d["W"])
    full = np.zeros(n_bins)
    c = n_bins // 2
    full[c - W:c + W + 1] = t
    return full, c


def _delay(sig, shift):
    """Fractional-sample delay of a real signal by an FFT phase ramp."""
    n = sig.shape[-1]
    f = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(sig) * np.exp(-2j * np.pi * f * shift), n=n)


def render(traj, template=None, pos=RADAR_POS, gains=(1.0, 1.0, 1.0),
           amp_exp=AMP_EXP, noise=0.0, scatterers=None, rng=None):
    """Render three-view RAW frames for a trajectory.

    traj       : (T, 3) hand position in metres
    scatterers : optional (N, 3) rigid offsets around the hand, with `weights`
                 defaulting to equal; None = a single effective scatterer whose
                 extent is already baked into the measured template
    noise      : additive Gaussian sigma, in units of the peak template amplitude
    Returns (V, T, M) float32.
    """
    rng = rng or np.random.default_rng(0)
    if template is None:
        template, centre = load_template()
    else:
        template, centre = template
    T = traj.shape[0]
    V = pos.shape[0]
    offs = np.zeros((1, 3)) if scatterers is None else np.asarray(scatterers)
    w = np.ones(len(offs)) / len(offs)

    out = np.zeros((V, T, M_BINS))
    for v in range(V):
        for t in range(T):
            frame = np.zeros(M_BINS)
            for o, wi in zip(offs, w):
                R = np.linalg.norm(traj[t] + o - pos[v])
                n = R / BIN_M                       # range in samples
                amp = wi * gains[v] * R ** amp_exp
                frame += amp * _delay(template, n - centre)
            out[v, t] = frame
    if noise > 0:
        out += rng.normal(0.0, noise, out.shape)
    return out.astype(np.float32)


def clutter_filter(raw, alpha=ALPHA_CLUTTER):
    """The dataset's loopback background subtraction, applied along slow time."""
    raw = np.asarray(raw, dtype=np.float64)
    out = np.empty_like(raw)
    cm = raw[:, 0].copy()
    for t in range(raw.shape[1]):
        cm = alpha * cm + (1 - alpha) * raw[:, t]
        out[:, t] = raw[:, t] - cm
    return out.astype(np.float32)


def render_cr(traj, **kw):
    """Convenience: render and apply the same preprocessing the dataset shipped."""
    return clutter_filter(render(traj, **kw))


# ---------------------------------------------------------------------------
def calibrate_gains_noise(X, y, subj, n=40, seed=0):
    """Per-view amplitude gain and noise sigma, measured on real ClutterRemoved data."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.where(y < 11)[0], n, replace=False)
    peaks = [[] for _ in range(3)]
    noise = [[] for _ in range(3)]
    for i in idx:
        unit = np.asarray(X[i])
        for v in range(3):
            e = np.abs(unit[v])
            peaks[v].append(np.percentile(e.max(axis=1), 75))
            # bins beyond the range where a hand can be = noise floor
            noise[v].append(e[:, 170:].std())
    g = np.array([np.median(p) for p in peaks])
    s = np.array([np.median(q) for q in noise])
    return g, s
