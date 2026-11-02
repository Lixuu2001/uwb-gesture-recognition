"""L1/L2 of the readout-failure preregistration (`proposal_v2/17` 3-6).

    Gate 0 passed (logs/w3_gate0_sim.log), so this analysis is unlocked.  Every
    formula, threshold, sample size and decision rule below is transcribed from
    `17`, which is FROZEN.  Nothing here may be tuned after seeing a result: if
    something in `17` turns out to be wrong or ambiguous, the fix is a new `18`
    stating that it happened after the results were seen - not an edit here.

Two layers, deliberately kept apart (17 0), because their evidence differs:

  L1 "the failure is predictable"  - carried by the C0 arm, the recovered
     RESULTS 49.5 readout.  Its range-domain companion is NOT a matched
     functional, so L1 can say the recorded failure was foreseeable but CANNOT
     name wrapping as the cause.
  L2 "the mechanism is wrapping"   - carried by the C1 arm, where channels D and
     R differ ONLY in where radial velocity comes from.  That matched contrast
     is the whole falsification power of the design: if the range channel dies
     too, the risk score was measuring something else and the attribution dies
     with it.

Two channels, identical after the velocity step (17 3):
  D  phase / Doppler, read on the INTEGER tracked bin, limited by lambda/4
  R  range domain, from the RAW Viterbi track, no wrapping limit

Eight subjects are the only independent unit (17 5).  ~1600 repetitions per arm
are NOT 1600 observations; every test here reduces to eight per-subject slopes
and an exact 2^8 sign-flip enumeration over them.

IMPLEMENTATION CONVENTIONS (fixed 2026-07-31, BEFORE any real data was read;
they are NOT clauses of `17` and `17` is not edited to accommodate them).  Each
covers a case the frozen text does not decide.  All of them fail loudly rather
than guessing, because a silent default here is indistinguishable from a choice
made after seeing the number:

  C1  sigma == 0 (exact tie in the estimation-set class means): the fold is
      unusable and the arm stops.  No +-1 is picked.
  C2  no merge may cross OR absorb the median stratum.  A group holding the
      median can neither initiate a merge nor receive one, so the left and right
      halves of the sweep can never be joined through it (see `merge_strata`).
  C3  the per-subject slope sign-flip test requires EXACTLY 8 subjects; a
      dropped subject is a hard error, never a quietly smaller enumeration.
      The fold-level Spearman test may legitimately lose folds to 5.1's
      "fewer than 3 usable strata" rule, so there it enumerates 2^k over the k
      survivors AND reports k; k < 8 is stated, not absorbed.
  C4  an estimation-set MAD of exactly 0 makes the margin undefined -> hard error.
  C5  IRLS non-convergence, or a constant outcome within a subject, -> hard
      error.  Neither is imputed.
  C6  17 6.1(3) and 6.2(4) require the main effect to "still be there" after the
      file-order index is added as a covariate, without quantifying it.  Frozen
      here as: the eight beta_u|j must still average negative AND pass the same
      one-sided sign-flip test at p < 0.05.  If they do not, the layer CANNOT
      pass and the finding is reported as 17 5.4 words it - risk is collinear
      with file order and the two cannot be separated.  Without this, a result
      driven entirely by acquisition order would sail through: the covariate was
      computed but never allowed to change any verdict.
  C7  every quantity entering a test must be finite, and the inputs must be the
      COMPLETE target set, not merely a subset that joins cleanly.  A NaN slope
      or rho is a hard error, never a value that silently compares false.  The
      sign-flip enumeration is defined for any k >= 1 (at k = 1 the smallest
      attainable p is 0.5, reported honestly rather than as NaN); k == 0 is a
      hard error, because "nothing was computable" and "the test came out
      negative" must not land in the same cell of the 6.4 matrix.
  C8  17 5.3's "conventional cluster-robust 95% CI" is implemented as a genuine
      sandwich on the SAMPLE-LEVEL regression with subjects as clusters, CR1
      small-sample correction, and t_{G-1} = t_7 critical values - not as a
      t interval over the eight per-subject slopes, which is a different
      estimator with a different meaning.  Supplementary only; with 8 clusters
      the sandwich asymptotics are out of reach and the interval is reported
      with that caveat rather than leaned on.

Run modes (mutually exclusive):
  --selftest   synthetic verification of every formula.  Touches no real data.
  --track      stage 1: per-sample scores from the cache -> logs/w3_readout_scores.npz
  --analyze    stage 2: the actual L1/L2 verdict.  Reads the frozen decision
               matrix of 17 6.4 and nothing else.

`--analyze` is never the default: the preregistered order is code first, audit
second, real numbers last.

Usage: python tools/w3_readout_predict.py --selftest
"""
import argparse
import itertools
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm, spearmanr
from scipy.stats import t as t_dist

sys.path.insert(0, ".")
from uwb_osr.trajectory import (viterbi_track, _analytic, bspline_basis,       # noqa: E402
                                fit_trajectory, RADAR_POS, BIN_M)
# selftest only: renders a SYNTHETIC echo cube from a synthetic template, so the
# stage-1 scoring path is exercised without reading the cache or w2_pulse.npz
from uwb_osr.forward import render as fwd_render, clutter_filter as fwd_clutter  # noqa: E402

CACHE = r"D:\uwb_gestures\cache"
# resolved from this file's location, not the working directory: a required gate
# that silently misses its input because of where the script was launched from is
# no gate at all
ROOT = Path(__file__).resolve().parents[1]
RISK_NPZ = str(ROOT / "logs" / "w3_readout_risk.npz")
C0_NPZ = str(ROOT / "logs" / "w3_c0_8_9.npz")
SCORES_NPZ = str(ROOT / "logs" / "w3_readout_scores.npz")
PULSE_NPZ = str(ROOT / "logs" / "w2_pulse.npz")

# authoritative sensing constants (`00` 4.1) - do not re-derive
LAMBDA_M = 41.1e-3
DT = 0.05                               # 20 fps
# fast-time carrier, measured on the dataset (uwb_osr/forward.py header, `00` 4.1).
# Used only to de-rotate the readout coordinate (`18`); not a free parameter.
CARRIER_CYC_PER_BIN = 0.3203
# `18` 6, FIXED sensitivity value, never the primary one.  Provenance: it is the
# `carrier` FIELD stored in logs/w2_pulse.npz (7.6199 GHz) converted to
# cycles/sample, i.e. the metadata figure, NOT a spectrum of the template array -
# the template's own peak is 0.3265 on its native grid and 0.3218 zero-padded to
# 4096.  The 0.8% gap against the authoritative 7.56 GHz is recorded in `13` 4.2.
CARRIER_ALT = 0.3228
# |u_v . e_d| from the analytic sensitivity table (`00` 4.4 / `07` 1)
SENS = 0.774

# meta column layout of logs/w3_readout_risk.npz
GEST_C, SUBJ_C, ORD_C, BLK_C, ROW_C = 0, 1, 2, 3, 4

class ProtocolStop(SystemExit):
    """A convention guard fired.  Carries its tag so a test can assert WHICH one.

    Subclasses SystemExit so it still stops the process with a non-zero status;
    the tag exists because "some SystemExit was raised" is not evidence that the
    intended guard is the one that fired.
    """

    def __init__(self, tag, msg):
        self.tag = tag
        super().__init__(f"{tag}: {msg}")


# W1 pipeline hyper-parameters, needed only for the Rfit secondary channel
K_CTRL, LAM, MU = 30, 0.05, 0.05
# D        wrapped phase          (17 3.1 as amended by `18` 2)  - the L2 carrier
# Dunw     same phase, branch restored from the RAW range (`18` 5) - L2 control 1
# R        RAW Viterbi range      (17 3.2)                        - L2 control 2
# Rfit     FIT reprojection range (17 3.2 secondary)
CHANNELS = ("D", "Dunw", "R", "Rfit")

N_STRATA = 5
MEDIAN_STRATUM = 2
MIN_PER_CLASS = 10                      # 17 5.1
MIN_STRATA = 3                          # 17 5.1
ALPHA = 0.05

# 17 3.4 / 3.5.  `pos` is class +, `neg` is class -; the sign sigma itself is
# NOT fixed here - it is estimated per fold (3.5).  These only fix which gesture
# is called + so that the label l_i is well defined.
ARMS = {
    "H":  dict(pos=0, neg=1, risk="LR",  layer="extension"),   # G1 vs G2
    "V":  dict(pos=3, neg=2, risk="T",   layer="extension"),   # G4 vs G3
    "C0": dict(pos=9, neg=8, risk="LTR", layer="L1"),          # G10 vs G9
    "C1": dict(pos=9, neg=8, risk="LTR", layer="L2"),          # G10 vs G9
}

# 17 3.5's geometric expectation table.  Explicitly "a prediction on record, NOT
# a criterion": sigma always comes from the estimation set, and a disagreement is
# reported rather than corrected.  The C rows are conditional anyway - the dataset
# paper never says whether CW is seen from the subject or from the radars.
EXPECTED_SIGMA = {"H": +1.0, "V": +1.0, "C0": +1.0, "C1": +1.0}


# ===========================================================================
# 1. velocity channels (17 3.1, 3.2)
# ===========================================================================
def velocity_from_phase(z_bin, path, carrier=CARRIER_CYC_PER_BIN):
    """(T,) complex at the tracked bin + (T,) integer bins -> (T-1,) velocity.

    17 3.1 AS AMENDED BY `18`.  The principal value of
    dphi = arg(z(t+1) conj(z(t))) is exactly the wrapping this experiment is
    about: any true |v| above lambda/(4 dt) = 0.2055 m/s folds back and is
    unrecoverable from this channel alone.

    The amendment is the de-rotation by the bin coordinate.  The pulse is a
    carrier under an envelope, a(u) exp(i 2 pi f_s u), so reading it at bin b of
    a frame whose echo sits at n samples gives arg = 2 pi f_s (b - n).  When the
    track steps to a new bin, the b term injects 2 pi f_s = 2.012 rad of phase
    that has nothing to do with motion - about 0.132 m/s of apparent velocity per
    bin crossed, which is the same size as the whole gesture.  Multiplying by
    exp(-i 2 pi f_s b) removes it and leaves -2 pi f_s n, the range phase.

    Why this had to be fixed before running: bin-crossing rate rises with radial
    speed, and so does the risk score, so the two are collinear.  Without the
    de-rotation, "risk up -> D dies, R survives" could be produced entirely by
    bin crossings, which have nothing to do with the lambda/4 threshold - a false
    L2 PASS.  See `18` 1.  The correction is deterministic and motion-independent,
    so it cannot hide wrapping: wrapping happens in the principal value taken
    AFTER it.
    """
    return LAMBDA_M / (4 * np.pi * DT) * phase_increments(z_bin, path, carrier)


def phase_increments(z_bin, path, carrier=CARRIER_CYC_PER_BIN):
    """De-rotated principal-value phase increments, (T-1,).  `18` 2."""
    ph = np.asarray(z_bin) * np.exp(-2j * np.pi * carrier * np.asarray(path, float))
    return np.angle(ph[1:] * np.conj(ph[:-1]))


def velocity_from_phase_unwrapped(z_bin, path, r_raw, carrier=CARRIER_CYC_PER_BIN):
    """Channel D_unwrapped: the SAME phase, with the 2*pi*k branch restored.

    `18` 5.  This is the control that isolates wrapping.  D and R differ in more
    than the ambiguity - one is a complex phase on an integer bin, the other a
    sub-bin envelope peak - so they also differ in phase distortion, in how the
    clutter filter bites, and in how scattering-centre switching shows up.  Any
    of those growing with motion would reproduce "D dies, R survives" with no
    wrapping involved at all.

    D_unwrapped shares every one of those pathologies with D_wrapped and differs
    ONLY in the 2*pi ambiguity, so the three-way comparison separates them:

        D_wrapped dies, D_unwrapped and R survive  ->  wrapping
        D_wrapped and D_unwrapped both die         ->  phase-channel distortion
        D and R both die                           ->  tracking / motion confound

    The branch is chosen with the RAW range track, not np.unwrap: unwrapping by
    continuity assumes the very thing under test (that successive increments stay
    inside +-pi) and would silently invent the answer.  The range track picks the
    integer k; the value still comes from the phase.
    """
    dphi = phase_increments(z_bin, path, carrier)
    # in the de-rotated frame the phase runs as -2 pi f_s rho / BIN_M
    expected = -2 * np.pi * carrier * np.diff(np.asarray(r_raw)) / BIN_M
    k = np.round((expected - dphi) / (2 * np.pi))
    return LAMBDA_M / (4 * np.pi * DT) * (dphi + 2 * np.pi * k)


def velocity_from_range(r):
    """(T,) range track in metres -> (T-1,) radial velocity, 17 3.2.

    The falsification control.  Same pooling, same functional, same tracked
    scatterer as channel D; the only difference is that a range difference has
    no ambiguity limit.
    """
    return np.diff(r) / DT


def channel_velocities(frames, B=None):
    """(3,T,M) clutter-removed frames -> {channel: (3, T-1) radial velocities}.

    The phase must be read on the INTEGER Viterbi bin (17 7, appendix A): the
    sub-bin parabolic refinement in `r` is meaningful for range and meaningless
    for indexing a complex sample, so `return_path=True` exists for this.

    Channels:
      D     phase, on the tracked bin                      (17 3.1, primary)
      R     RAW Viterbi range differences                  (17 3.2, the control)
      Rfit  the same differences on the FIT reprojection    (17 3.2 "declared
            secondary analysis", included so the range channel can also be read
            in the smoothed geometry that the recorded 92.81% came from)
    """
    T = frames.shape[1]
    vD, vU, vR = (np.empty((3, T - 1)) for _ in range(3))
    rs, cs = [], []
    for v in range(3):
        r, conf, path = viterbi_track(frames[v], return_path=True)
        z = _analytic(frames[v].astype(np.float64))          # (T, M) complex
        zb = z[np.arange(T), path]
        vD[v] = velocity_from_phase(zb, path)
        vU[v] = velocity_from_phase_unwrapped(zb, path, r)
        vR[v] = velocity_from_range(r)
        rs.append(r)
        cs.append(conf)
    out = {"D": vD, "Dunw": vU, "R": vR}
    if B is not None:
        fit = fit_trajectory(np.stack(rs), np.stack(cs), (0, 1, 2),
                             B=B, lam=LAM, mu=MU)
        rho = np.linalg.norm(fit["p"][None, :, :] - RADAR_POS[:, None, :], axis=2)
        out["Rfit"] = np.stack([velocity_from_range(rho[v]) for v in range(3)])
    return out


def pool_views(vv):
    """(3, T-1) per-view radial velocities -> (vx, vy) in the pooled plane.

    17 3.3, inverting the analytic sensitivities (L +0.774 / T 0 / R -0.774 for
    horizontal; L 0 / T -0.774 / R 0 for vertical).  Note the two exact zeros:
    the Top view carries no horizontal information at all, and Left/Right carry
    no vertical information at all, so neither axis is an average over three
    views - each is a specific contrast.
    """
    vx = (vv[0] - vv[2]) / (2 * SENS)
    vy = -vv[1] / SENS
    return vx, vy


# ===========================================================================
# 2. score functionals (17 3.4)
# ===========================================================================
def score_h(vx):
    """Third moment of the horizontal velocity; T-1 terms."""
    return float(np.mean(vx ** 3))


def score_v(vy):
    return float(np.mean(vy ** 3))


def score_c1(vx, vy):
    """Signed loop area of the INTEGRATED planar path, start pinned at origin.

    17 3.4 freezes the offset convention as "start at zero", which is bitwise
    equivalent to explicitly closing the polygon: the closing term is
    0.5*(x[-1]*y[0] - x[0]*y[-1]) and both factors y[0], x[0] are zero by
    construction.  It is NOT equivalent to centring (that is what `08` did).
    The shoelace sum of an OPEN path is not translation invariant, so this is
    not a property that could be waved through - the integration is what makes
    the absolute offset never enter the formula at all.
    """
    x = np.concatenate([[0.0], np.cumsum(vx) * DT])
    y = np.concatenate([[0.0], np.cumsum(vy) * DT])
    return float(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def sample_scores(frames, B=None):
    """One (3,T,M) sample -> {(arm, channel): score} for H, V and C1."""
    out = {}
    for ch, vv in channel_velocities(frames, B=B).items():
        vx, vy = pool_views(vv)
        out[("H", ch)] = score_h(vx)
        out[("V", ch)] = score_v(vy)
        out[("C1", ch)] = score_c1(vx, vy)
    return out


# ===========================================================================
# 3. estimation-set quantities (17 3.5, 4, 5.1)
# ===========================================================================
def polarity(scores, is_pos, est):
    """sigma = sign(mean(s | class+) - mean(s | class-)) on the estimation set.

    17 3.5 removes the `max(acc, 100-acc)` global flip that the earlier scripts
    used; that flip was a free parameter chosen after seeing the accuracy.  Here
    the sign is a statistic of seven held-out-of-test subjects and nothing else.

    The frozen formula is `sign`, so an exact tie returns 0.0 - it is NOT rounded
    to +1 or -1.  Convention C1 below turns that 0 into an unusable fold rather
    than a coin flip: picking a direction on a tie would reintroduce exactly the
    free parameter that 3.5 was written to remove.
    """
    a = scores[est & is_pos].mean()
    b = scores[est & ~is_pos].mean()
    return float(np.sign(a - b))


def mad_scale(x):
    """Robust sd = 1.4826 * median absolute deviation (17 4)."""
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def strata_edges(risk_est):
    """Quintile cut points of the estimation-set risk; frozen per fold (17 5.1)."""
    return np.percentile(risk_est, [20, 40, 60, 80])


def assign_strata(risk, edges):
    """-> stratum id in 0..4."""
    return np.searchsorted(edges, risk, side="right")


def merge_strata(strata, is_pos, min_per_class=MIN_PER_CLASS,
                 n_strata=N_STRATA, median=MEDIAN_STRATUM):
    """Merge thin strata TOWARDS THE MEDIAN until each holds >= min_per_class
    of both classes (17 5.1).  Returns (usable_groups, missing_groups).

    Merging inward rather than outward is what keeps the extremes - the whole
    point of the stratification is the low-risk and high-risk ends, so a thin
    tail is absorbed into its neighbour instead of dissolving the contrast.

    CONVENTION C2, not a clause of 17: a group that already contains the median
    stratum has no "towards the median" direction left, so it is recorded
    missing rather than merged across the median.  17 5.1 specifies the
    direction and the re-check but is silent on this case.  The conservative
    reading is taken because the alternative - letting the median absorb its
    neighbours - could collapse the whole sweep into one group and quietly
    destroy the very contrast the stratification exists to measure.
    """
    def ok(g):
        m = np.isin(strata, g)
        return (is_pos[m].sum() >= min_per_class
                and (~is_pos[m]).sum() >= min_per_class)

    groups = [[i] for i in range(n_strata)]
    changed = True
    while changed:
        changed = False
        for gi, g in enumerate(groups):
            if ok(g):
                continue
            if max(g) < median:
                tgt = gi + 1
            elif min(g) > median:
                tgt = gi - 1
            else:
                continue                      # holds the median: cannot initiate
            if not 0 <= tgt < len(groups):
                continue
            if median in groups[tgt]:
                # C2: the target already holds the median.  Absorbing it would
                # put the median inside a group that a stratum from the OTHER
                # side could then merge into, joining the two halves through it -
                # which is what "never merge across the median" forbids.  Blocking
                # receipt as well as initiation is what actually enforces the rule;
                # blocking initiation alone leaves exactly that hole.
                continue
            groups[min(gi, tgt)] = sorted(groups[gi] + groups[tgt])
            del groups[max(gi, tgt)]
            changed = True
            break
    return [g for g in groups if ok(g)], [g for g in groups if not ok(g)]


def signed_dprime(scores, is_pos, sigma):
    """d' of class+ over class-, oriented by sigma so larger = better readout."""
    a, b = scores[is_pos], scores[~is_pos]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(sigma * (a.mean() - b.mean())
                 / np.sqrt(0.5 * (a.var() + b.var()) + 1e-30))


# ===========================================================================
# 4. per-subject models and the exact sign-flip test (17 5)
# ===========================================================================
def irls_logit(X, yb, max_iter=200, tol=1e-11):
    """Unpenalised logistic regression by IRLS -> (beta, converged).

    Hand-rolled rather than sklearn because sklearn regularises by default; an
    L2 penalty would shrink exactly the slope this experiment tests.  The
    selftest cross-checks this against sklearn with penalty=None.
    """
    yb = np.asarray(yb, float)
    if yb.min() == yb.max():
        return None, False                    # constant outcome: slope undefined
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(p * (1.0 - p), 1e-12)
        z = eta + (yb - p) / w
        WX = X * w[:, None]
        try:
            new = np.linalg.solve(X.T @ WX, WX.T @ z)
        except np.linalg.LinAlgError:
            return None, False
        if np.max(np.abs(new - beta)) < tol:
            return new, True
        beta = new
    return beta, False


def ols_slope(X, y):
    """Least-squares coefficients; X includes the intercept column."""
    return np.linalg.lstsq(X, np.asarray(y, float), rcond=None)[0]


def sign_flip_p(b):
    """Exact one-sided NEGATIVE sign-flip test over the 8 subjects (17 5.2).

        p = #{eps in {+-1}^n : mean(eps * b) <= mean(b)_obs} / 2^n

    The observed assignment is the all-+1 row, so p >= 1/2^n always; with n=8
    the smallest attainable p is 1/256 = 0.0039.  An earlier draft of `17` used
    absolute values, which is the two-sided test and disagreed with the stated
    one-sided direction; the corrected form is the one implemented here.
    """
    b = np.asarray(b, float)
    n = len(b)
    if n < 1:
        raise ProtocolStop("C7", "sign-flip test called with no slopes")
    if not np.all(np.isfinite(b)):
        raise ProtocolStop("C7", f"non-finite value in the sign-flip input: {b}")
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
    means = signs @ b / n
    obs = means[0]                            # the all-+1 row, bitwise identical
    return float(np.mean(means <= obs))


def cluster_robust_ci(X, y, groups, kind="ols", alpha=ALPHA):
    """C8: cluster-robust (sandwich) 95% CI on a SAMPLE-LEVEL slope.

    17 5.3 asks for a "conventional cluster-robust 95% CI" as a SUPPLEMENTARY
    report.  A t interval over the eight per-subject slopes is not that - it is a
    one-sample interval on eight numbers - so the sandwich is computed properly
    here:

        V = bread * (sum_g s_g s_g') * bread,    s_g = sum_{i in g} x_i u_i

    with bread = (X'X)^-1 for OLS and (X'WX)^-1 for the logit, u_i the residual
    (y_i - p_i for the logit).  Small-sample correction is the usual CR1,
    c = G/(G-1) * (N-1)/(N-K), and the critical value comes from t with G-1 = 7
    degrees of freedom, NOT the normal: with only eight clusters the asymptotics
    behind the sandwich are not in reach, and the interval is reported with that
    caveat attached.

    It is never a criterion.  The decision rests on the exact sign-flip
    enumeration, which needs no PARAMETRIC form - but it is not assumption
    free either: it assumes the per-subject slopes are sign-symmetric /
    exchangeable under the null.  That is a weaker and more transparent
    assumption than this interval's, not the absence of one.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    g_ids = np.unique(groups)
    G, N, K = len(g_ids), X.shape[0], X.shape[1]
    if G < 2 or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ProtocolStop("C7", "cluster-robust CI needs >= 2 finite clusters")

    if kind == "logit":
        beta, conv = irls_logit(X, y)
        if beta is None or not conv:
            raise ProtocolStop("C5", "pooled logistic fit for the cluster CI failed")
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        u = y - p
        bread = np.linalg.inv(X.T @ (X * np.maximum(p * (1 - p), 1e-12)[:, None]))
    else:
        beta = ols_slope(X, y)
        u = y - X @ beta
        bread = np.linalg.inv(X.T @ X)

    meat = np.zeros((K, K))
    for g in g_ids:
        s = (X[groups == g] * u[groups == g][:, None]).sum(axis=0)
        meat += np.outer(s, s)
    c = (G / (G - 1)) * ((N - 1) / (N - K))
    V = c * bread @ meat @ bread
    se = float(np.sqrt(V[1, 1]))
    h = t_dist.ppf(1 - alpha / 2, G - 1) * se
    return float(beta[1]), float(beta[1] - h), float(beta[1] + h), G


# ===========================================================================
# 5. selftest - synthetic data only, no cache access
# ===========================================================================
class _Check:
    def __init__(self):
        self.fails = 0

    def __call__(self, name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
        if not cond:
            self.fails += 1


def selftest():
    print("=" * 78)
    print("SELFTEST - synthetic data only, no cache, no real result")
    print("=" * 78)
    chk = _Check()
    rng = np.random.default_rng(0)

    # ---------------------------------------------------------------- 3.1 D
    print("\n17 3.1  channel D: phase -> radial velocity, and its wrapping limit")
    v_unamb = LAMBDA_M / (4 * DT)
    for v_true in (0.05, 0.12, 0.20):
        # a constant radial velocity advances phase by 4 pi v dt / lambda per frame
        dphi = 4 * np.pi * v_true * DT / LAMBDA_M
        z = np.exp(1j * dphi * np.arange(40))
        est = velocity_from_phase(z, np.zeros(40))
        chk(f"unwrapped v={v_true} m/s recovered", np.allclose(est, v_true, atol=1e-12),
            f"est={est[0]:.6f}")
    v_over = 0.30                              # above 0.2055 m/s: must fold back
    dphi = 4 * np.pi * v_over * DT / LAMBDA_M
    z = np.exp(1j * dphi * np.arange(40))
    est = velocity_from_phase(z, np.zeros(40))[0]
    folded = v_over - 2 * v_unamb
    chk("v above lambda/(4dt) folds to the aliased value",
        np.isclose(est, folded, atol=1e-12), f"{v_over} -> {est:.4f} (expected {folded:.4f})")
    chk("unambiguous limit is 0.2055 m/s", np.isclose(v_unamb, 0.2055, atol=5e-5),
        f"{v_unamb:.4f}")

    # ---------------------------------------------------------------- 3.2 R
    print("\n17 3.2  channel R: range differences, no ambiguity limit")
    r = np.cumsum(np.full(40, 0.30 * DT))
    chk("v=0.30 m/s recovered exactly where D would alias",
        np.allclose(velocity_from_range(r), 0.30), f"{velocity_from_range(r)[0]:.6f}")

    # ---------------------------------------------------------------- 3.3
    print("\n17 3.3  cross-view pooling inverts the analytic sensitivities")
    vx_t, vy_t = 0.11, -0.07
    vv = np.stack([np.full(5, SENS * vx_t),                    # Left  +0.774 x
                   np.full(5, -SENS * vy_t),                   # Top   -0.774 y
                   np.full(5, -SENS * vx_t)])                  # Right -0.774 x
    vx, vy = pool_views(vv)
    chk("pooled vx round-trips", np.allclose(vx, vx_t), f"{vx[0]:.6f} vs {vx_t}")
    chk("pooled vy round-trips", np.allclose(vy, vy_t), f"{vy[0]:.6f} vs {vy_t}")
    vv_h = np.stack([np.full(5, SENS * vx_t), np.zeros(5), np.full(5, -SENS * vx_t)])
    chk("a pure horizontal motion leaves vy at zero",
        np.allclose(pool_views(vv_h)[1], 0.0))

    # ---------------------------------------------------------------- 3.4 C1
    print("\n17 3.4  C1 signed loop area of the integrated path")
    n, a = 400, 0.15
    th = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    # velocities of a CCW circle of radius a, one full turn over n frames
    vx = -a * (2 * np.pi / (n * DT)) * np.sin(th)
    vy = a * (2 * np.pi / (n * DT)) * np.cos(th)
    area = score_c1(vx, vy)
    chk("CCW unit-circle area ~ +pi a^2", abs(area - np.pi * a ** 2) < 2e-3 * np.pi * a ** 2,
        f"{area:.6f} vs {np.pi * a ** 2:.6f}")
    chk("CW is the exact negative", np.isclose(score_c1(vx, -vy), -area, rtol=1e-12))
    # start-at-origin == explicit closure, and != centring
    x = np.concatenate([[0.0], np.cumsum(vx) * DT])
    y = np.concatenate([[0.0], np.cumsum(vy) * DT])
    closed = 0.5 * (np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
                    + x[-1] * y[0] - x[0] * y[-1])
    chk("start-at-origin is bitwise identical to explicit closure", closed == area)
    xc, yc = x - x.mean(), y - y.mean()
    centred = 0.5 * np.sum(xc[:-1] * yc[1:] - xc[1:] * yc[:-1])
    chk("on a CLOSED path the two conventions coincide (17 3.4b's own caveat)",
        np.isclose(centred, area), f"centred={centred:.6f} vs frozen={area:.6f}")
    # the separation only shows on an OPEN path - which is what real velocity
    # sequences are, and the reason 17 3.4 had to pick a convention at all
    th_o = np.linspace(0, 2.5 * np.pi, n + 1)[:-1]
    vx_o = -a * (2.5 * np.pi / (n * DT)) * np.sin(th_o)
    vy_o = a * (2.5 * np.pi / (n * DT)) * np.cos(th_o)
    area_o = score_c1(vx_o, vy_o)
    xo = np.concatenate([[0.0], np.cumsum(vx_o) * DT])
    yo = np.concatenate([[0.0], np.cumsum(vy_o) * DT])
    closed_o = 0.5 * (np.sum(xo[:-1] * yo[1:] - xo[1:] * yo[:-1])
                      + xo[-1] * yo[0] - xo[0] * yo[-1])
    xoc, yoc = xo - xo.mean(), yo - yo.mean()
    centred_o = 0.5 * np.sum(xoc[:-1] * yoc[1:] - xoc[1:] * yoc[:-1])
    chk("open path: start-at-origin still equals explicit closure",
        closed_o == area_o)
    chk("open path: centring is a DIFFERENT convention (17 3.4b)",
        not np.isclose(centred_o, area_o),
        f"centred={centred_o:.6f} vs frozen={area_o:.6f}")
    # open paths: the shoelace sum is not translation invariant
    op_x, op_y = np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0])
    sh = lambda X, Y: 0.5 * np.sum(X[:-1] * Y[1:] - X[1:] * Y[:-1])          # noqa: E731
    chk("open-path shoelace is NOT translation invariant (why the pin matters)",
        not np.isclose(sh(op_x, op_y), sh(op_x + 3.0, op_y + 2.0)))

    # ---------------------------------------------------------------- 3.5
    print("\n17 3.5  polarity from the estimation set only")
    s = np.array([2.0, 3.0, 2.5, -2.0, -3.0, -2.5, 99.0, -99.0])
    is_pos = np.array([True] * 3 + [False] * 3 + [False, True])
    est = np.array([True] * 6 + [False, False])       # the last two are the test subject
    chk("sigma=+1 when class+ scores higher on the estimation set",
        polarity(s, is_pos, est) == 1.0)
    chk("sigma flips with the estimation-set means",
        polarity(-s, is_pos, est) == -1.0)
    chk("the test subject cannot influence sigma",
        polarity(s, is_pos, est) == polarity(np.where(est, s, -s * 1000), is_pos, est))
    # C1: the frozen formula is sign(), so an exact tie is 0 - not rounded to +-1
    tie = np.array([1.0, -1.0, 2.0, 1.0, -1.0, 2.0, 7.0, -7.0])
    tie_pos = np.array([True] * 3 + [False] * 3 + [True, False])
    chk("an exact tie in the estimation-set means returns sigma = 0 (C1)",
        polarity(tie, tie_pos, est) == 0.0, f"sigma={polarity(tie, tie_pos, est)}")
    chk("sigma = 0 is falsy, so a caller cannot mistake it for a direction",
        not bool(polarity(tie, tie_pos, est)))

    # ---------------------------------------------------------------- 4 MAD
    print("\n17 4  robust scale and the margin")
    g = rng.normal(0.0, 3.0, 200000)
    chk("1.4826*MAD recovers sigma of a normal", abs(mad_scale(g) - 3.0) < 0.03,
        f"{mad_scale(g):.4f}")
    # Gross outliers do not blow the estimate up, but they do NOT leave it
    # untouched either, and the shift is deterministic rather than incidental:
    # a contamination fraction eps sitting entirely in the tail pushes the median
    # of the deviations from the 0.5 quantile of the clean |X| to the
    # 0.5/(1-eps) quantile.  Asserting that predicted value is a real test of the
    # estimator; a hand-picked tolerance around 3.0 would only have hidden it.
    eps = 200 / 2200
    pred = 1.4826 * 3.0 * norm.ppf((1 + 0.5 / (1 - eps)) / 2)
    sym_c = np.concatenate([rng.normal(0.0, 3.0, 2000), np.full(100, 1e6),
                            np.full(100, -1e6)])
    one_c = np.concatenate([rng.normal(0.0, 3.0, 2000), np.full(200, 1e6)])
    # `pred` is the ASYMPTOTIC value; with 2200 draws the estimate lands in a
    # finite-sample neighbourhood of it, not on it, so the check is stated that
    # way rather than as an exact prediction
    for nm, arr in (("symmetric", sym_c), ("one-sided", one_c)):
        chk(f"9% {nm} gross outliers move 1.4826*MAD into the finite-sample "
            f"neighbourhood of its asymptotic value",
            abs(mad_scale(arr) - pred) < 0.12,
            f"{mad_scale(arr):.4f} vs asymptotic {pred:.4f}")
        chk(f"...while the plain sd is destroyed ({nm})",
            mad_scale(arr) < arr.std() / 1e4,
            f"MAD {mad_scale(arr):.3f} vs sd {arr.std():.3g}")
    sc, sig, lab, sd = np.array([2.0, -1.0]), 1.0, np.array([1.0, 1.0]), 2.0
    chk("correct = 1[sigma*s*l>0]", list(sig * sc * lab > 0) == [True, False])
    chk("margin scales by the estimation-set sd",
        np.allclose(sig * sc * lab / sd, [1.0, -0.5]))

    # ---------------------------------------------------------------- 5 flip
    print("\n17 5.2  exact one-sided sign-flip test over 8 subjects")
    chk("all-negative slopes give the minimum p = 1/256",
        np.isclose(sign_flip_p(-np.arange(1.0, 9.0)), 1 / 256), f"{sign_flip_p(-np.arange(1.0, 9.0)):.6f}")
    chk("all-positive slopes give p = 1",
        np.isclose(sign_flip_p(np.arange(1.0, 9.0)), 1.0))
    sym = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0])
    chk("a symmetric set sits near p = 0.5", abs(sign_flip_p(sym) - 0.5) < 0.06,
        f"{sign_flip_p(sym):.4f}")
    chk("p is never below 1/256 (the observed row is always counted)",
        min(sign_flip_p(rng.normal(-5, 1, 8)) for _ in range(200)) >= 1 / 256)
    chk("one-sided: it is NOT symmetric under negation",
        not np.isclose(sign_flip_p(np.array([-3.0, -1, -2, -4, -1, -2, -3, -1])),
                       sign_flip_p(np.array([3.0, 1, 2, 4, 1, 2, 3, 1]))))

    # ---------------------------------------------------------------- 5.1
    print("\n17 5.1  stratum merging towards the median")
    strata = np.concatenate([np.full(4, 0), np.full(40, 1), np.full(40, 2),
                             np.full(40, 3), np.full(4, 4)])
    is_p = np.zeros(len(strata), bool)
    for st in range(5):                       # exactly half of each stratum is class+
        i = np.where(strata == st)[0]
        is_p[i[:len(i) // 2]] = True
    usable, missing = merge_strata(strata, is_p)
    chk("thin end strata are absorbed inward", usable == [[0, 1], [2], [3, 4]],
        f"usable={usable}")
    chk("nothing is reported missing here", missing == [], f"missing={missing}")
    chk("enough strata survive for a Spearman", len(usable) >= MIN_STRATA)
    fat = np.repeat(np.arange(5), 40)
    is_f = np.zeros(len(fat), bool)
    for st in range(5):
        i = np.where(fat == st)[0]
        is_f[i[:20]] = True
    chk("well-populated strata are left alone",
        merge_strata(fat, is_f)[0] == [[0], [1], [2], [3], [4]])
    one_sided = np.repeat(np.arange(5), 40)
    is_o = np.zeros(len(one_sided), bool)
    is_o[np.where(one_sided == 4)[0][:3]] = True          # stratum 4: only 3 class+
    for st in range(4):
        i = np.where(one_sided == st)[0]
        is_o[i[:20]] = True
    u2, _ = merge_strata(one_sided, is_o)
    chk("a one-CLASS shortage also triggers a merge", [4] not in u2, f"usable={u2}")
    dead = np.repeat(np.arange(5), 4)                     # every stratum too thin
    is_d = np.zeros(len(dead), bool)
    is_d[::2] = True
    u3, m3 = merge_strata(dead, is_d)
    chk("an all-thin fold ends up unusable, not silently merged into one",
        len(u3) < MIN_STRATA, f"usable={u3} missing={m3}")
    # C2 regression: counts 10/10, 4/4, 2/2, 4/4, 10/10.  Blocking only the
    # median group from INITIATING a merge let [1] join [2] and then [3] join
    # [1,2] - joining the two halves through the median and manufacturing the
    # third stratum that turns an excluded fold into an included one.
    cross = np.repeat(np.arange(5), 20)
    is_c = np.zeros(len(cross), bool)
    for st, npos in zip(range(5), (10, 4, 2, 4, 10)):
        i = np.where(cross == st)[0]
        is_c[i[:npos]] = True                             # npos +, npos - after trim
    keep = np.zeros(len(cross), bool)
    for st, npos in zip(range(5), (10, 4, 2, 4, 10)):
        i = np.where(cross == st)[0]
        keep[i[:2 * npos]] = True
    cross, is_c = cross[keep], is_c[keep]
    u5, m5 = merge_strata(cross, is_c)
    chk("C2: no group may merge ACROSS or INTO the median",
        all(not (min(g) < MEDIAN_STRATUM < max(g)) for g in u5 + m5)
        and all(MEDIAN_STRATUM not in g or g == [MEDIAN_STRATUM] for g in u5 + m5),
        f"usable={u5} missing={m5}")
    chk("...so this fold is excluded rather than scraping together 3 strata",
        len(u5) < MIN_STRATA, f"{len(u5)} usable")
    # C2: the median stratum alone is deficient - it has no direction to merge in
    med = np.repeat(np.arange(5), 40)
    is_m = np.zeros(len(med), bool)
    for st in range(5):
        i = np.where(med == st)[0]
        is_m[i[:(3 if st == MEDIAN_STRATUM else 20)]] = True
    u4, m4 = merge_strata(med, is_m)
    chk("a deficient MEDIAN stratum is recorded missing, not merged across (C2)",
        [MEDIAN_STRATUM] in m4 and all(MEDIAN_STRATUM not in g for g in u4),
        f"usable={u4} missing={m4}")
    chk("its neighbours survive on their own rather than being dragged in",
        [1] in u4 and [3] in u4, f"usable={u4}")

    # ---------------------------------------------------------------- 5.1 d'
    print("\n17 5.1  stratified d', oriented by sigma")
    a, b = rng.normal(1.0, 1.0, 4000), rng.normal(0.0, 1.0, 4000)
    sc = np.concatenate([a, b])
    ip = np.concatenate([np.ones(4000, bool), np.zeros(4000, bool)])
    chk("d' ~ 1 for unit-separated unit-variance classes",
        abs(signed_dprime(sc, ip, 1.0) - 1.0) < 0.06, f"{signed_dprime(sc, ip, 1.0):.4f}")
    chk("sigma=-1 flips the orientation",
        np.isclose(signed_dprime(sc, ip, -1.0), -signed_dprime(sc, ip, 1.0)))

    # ---------------------------------------------------------------- 5 IRLS
    print("\n17 5  per-subject slopes")
    n = 4000
    R = rng.uniform(0, 1, n)
    yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-(1.5 - 3.0 * R)))).astype(float)
    Xd = np.column_stack([np.ones(n), R])
    beta, conv = irls_logit(Xd, yb)
    chk("IRLS converges", conv)
    chk("IRLS recovers a known negative slope", abs(beta[1] - (-3.0)) < 0.35,
        f"beta={beta[1]:.4f} vs -3.0")
    try:
        from sklearn.linear_model import LogisticRegression
        # sklearn's lbfgs stops at tol=1e-4 by default, which is looser than the
        # IRLS convergence here; tighten it or the comparison measures sklearn's
        # stopping rule rather than agreement between the two estimators
        sk = LogisticRegression(penalty=None, max_iter=5000, tol=1e-10
                                ).fit(R[:, None], yb)
        chk("IRLS agrees with sklearn(penalty=None, tol=1e-10)",
            abs(beta[1] - sk.coef_[0, 0]) < 1e-6,
            f"{beta[1]:.8f} vs {sk.coef_[0, 0]:.8f}")
        sk_loose = LogisticRegression(penalty=None, max_iter=5000).fit(R[:, None], yb)
        chk("the earlier disagreement was sklearn's default tol, not the estimator",
            abs(beta[1] - sk_loose.coef_[0, 0]) > abs(beta[1] - sk.coef_[0, 0]),
            f"default-tol gap {abs(beta[1] - sk_loose.coef_[0, 0]):.2e} vs "
            f"tight {abs(beta[1] - sk.coef_[0, 0]):.2e}")
        skl2 = LogisticRegression(max_iter=2000).fit(R[:, None], yb)
        chk("default sklearn L2 WOULD have shrunk it (why IRLS is used)",
            abs(skl2.coef_[0, 0]) < abs(sk.coef_[0, 0]),
            f"L2={skl2.coef_[0, 0]:.4f} vs unpenalised={sk.coef_[0, 0]:.4f}")
    except ImportError:
        print("       (sklearn missing - cross-check skipped)")
    chk("a constant outcome yields no slope rather than a silent number",
        irls_logit(Xd, np.ones(n)) == (None, False))
    m = 2.0 - 4.0 * R + rng.normal(0, 0.5, n)
    chk("OLS recovers a known margin slope", abs(ols_slope(Xd, m)[1] - (-4.0)) < 0.1,
        f"{ols_slope(Xd, m)[1]:.4f}")

    # ---------------------------------------------------------------- 5.4
    print("\n17 5.4  file-order index as a covariate")
    j = rng.uniform(0, 1, n)
    R_col = 0.9 * j + 0.1 * rng.uniform(0, 1, n)          # risk nearly collinear with j
    yb2 = (rng.uniform(size=n) < 1 / (1 + np.exp(-(1.0 - 4.0 * j)))).astype(float)
    b_no, _ = irls_logit(np.column_stack([np.ones(n), R_col]), yb2)
    b_yes, _ = irls_logit(np.column_stack([np.ones(n), R_col, j]), yb2)
    chk("without j, a j-driven outcome looks like a risk effect", b_no[1] < -1.0,
        f"beta_R={b_no[1]:.4f}")
    chk("adding j collapses that spurious risk effect", abs(b_yes[1]) < abs(b_no[1]) / 2,
        f"beta_R|j={b_yes[1]:.4f} vs {b_no[1]:.4f}")
    yb3 = (rng.uniform(size=n) < 1 / (1 + np.exp(-(1.0 - 4.0 * R_col)))).astype(float)
    c_no, _ = irls_logit(np.column_stack([np.ones(n), R_col]), yb3)
    c_yes, _ = irls_logit(np.column_stack([np.ones(n), R_col, j]), yb3)
    chk("a genuinely risk-driven effect survives the same control",
        c_yes[1] < 0 and abs(c_yes[1]) > abs(c_no[1]) / 2,
        f"beta_R={c_no[1]:.4f} -> beta_R|j={c_yes[1]:.4f}")

    # ---------------------------------------------------------------- pipeline
    print("\nend-to-end: a synthetic CCW sample scores positive on C1, both channels")
    T = 90
    th = np.linspace(0, 2 * np.pi, T)
    amp, w = 0.05, 2 * np.pi / (T * DT)
    vx_t = -amp * w * np.sin(th)[:-1]
    vy_t = amp * w * np.cos(th)[:-1]
    vv = np.stack([SENS * vx_t, -SENS * vy_t, -SENS * vx_t])
    px, py = pool_views(vv)
    chk("C1 of a synthetic CCW loop is positive", score_c1(px, py) > 0,
        f"{score_c1(px, py):.6f}")
    chk("C1 of the mirrored CW loop is negative", score_c1(px, -py) < 0)
    chk("H of a rightward-biased stroke is positive",
        score_h(np.array([0.3, 0.3, -0.1, -0.1, -0.1, -0.1])) > 0)

    # ---------------------------------------------------------------- assembly
    print("\n17 2 / 6  assembly layer: risk pooling, fold driver, guard rails")
    rf = np.array([[0.10, 0.50, 0.30], [0.20, 0.60, 0.40]])
    chk("R^H = mean(Left, Right), Top excluded",
        np.allclose(paired_risk(rf, "LR"), [0.20, 0.30]))
    chk("R^V = Top only", np.allclose(paired_risk(rf, "T"), [0.50, 0.60]))
    chk("R^C = mean of all three",
        np.allclose(paired_risk(rf, "LTR"), [0.30, 0.40]))

    n_sub, n_per = 8, 200
    sub_s = np.repeat(np.arange(1, 9), n_per)
    ip_s = np.tile([True, False], n_sub * n_per // 2)
    R_s = rng.uniform(0, 1, n_sub * n_per)
    lab_s = np.where(ip_s, 1.0, -1.0)
    j_s = np.tile(np.arange(n_per), n_sub)
    # readout quality decays with risk: sigma*s*l = (1-R) + noise
    s_s = lab_s * (1.0 - R_s) + rng.normal(0, 0.35, n_sub * n_per)
    res = fold_analysis(s_s, R_s, j_s, ip_s, sub_s, "SYNTH", "D")
    chk("fold driver returns one slope per subject", len(res["beta"]) == 8)
    chk("a planted risk->failure effect gives 8 negative slopes",
        np.all(res["beta"] < 0), f"{np.round(res['beta'], 2)}")
    chk("...and the minimum attainable sign-flip p",
        np.isclose(sign_flip_p(res["beta"]), 1 / 256))
    chk("margin slopes are negative too", np.all(res["b"] < 0))
    chk("stratified rho is negative on all 8 folds",
        len(res["rho"]) == 8 and np.all(res["rho"] < 0), f"{np.round(res['rho'], 2)}")
    s_null = lab_s * 0.6 + rng.normal(0, 0.35, n_sub * n_per)     # no risk dependence
    res0 = fold_analysis(s_null, R_s, j_s, ip_s, sub_s, "SYNTH", "R")
    chk("a risk-independent readout is NOT significant",
        sign_flip_p(res0["beta"]) >= ALPHA, f"p = {sign_flip_p(res0['beta']):.4f}")

    def tag_of(fn):
        """Which guard fired - not merely 'something raised SystemExit'."""
        try:
            fn()
        except ProtocolStop as e:
            return e.tag
        except SystemExit:
            return "SystemExit(untagged)"
        return None

    chk("C3: fewer than 8 subjects fires C3 specifically",
        tag_of(lambda: fold_analysis(s_s[sub_s != 8], R_s[sub_s != 8], j_s[sub_s != 8],
                                     ip_s[sub_s != 8], sub_s[sub_s != 8], "S", "D")) == "C3")
    s_tie = np.ones(n_sub * n_per)                               # identical class means
    chk("C1: sigma = 0 fires C1 specifically",
        tag_of(lambda: fold_analysis(s_tie, R_s, j_s, ip_s, sub_s, "S", "D")) == "C1")
    # A genuine C4: the pooled estimation set is >50% one value, so the MAD is
    # exactly 0, while the class means still differ so C1 does NOT fire first.
    s_mad0 = np.zeros(n_sub * n_per)
    s_mad0[np.where(ip_s)[0][::10]] = 1.0
    chk("C4 construction really has MAD = 0 with a non-zero sigma",
        mad_scale(s_mad0[sub_s != 1]) == 0.0
        and polarity(s_mad0, ip_s, sub_s != 1) != 0.0,
        f"MAD={mad_scale(s_mad0[sub_s != 1])}, sigma={polarity(s_mad0, ip_s, sub_s != 1)}")
    chk("C4: a zero estimation-set MAD fires C4 specifically",
        tag_of(lambda: fold_analysis(s_mad0, R_s, j_s, ip_s, sub_s, "S", "D")) == "C4")
    chk("C5: a constant within-subject outcome fires C5 specifically",
        tag_of(lambda: fold_analysis(lab_s * 50.0 + rng.normal(0, .01, n_sub * n_per),
                                     R_s, j_s, ip_s, sub_s, "S", "D")) == "C5")
    chk("C7: a non-finite score is a hard error",
        tag_of(lambda: fold_analysis(np.where(np.arange(n_sub * n_per) == 3, np.nan, s_s),
                                     R_s, j_s, ip_s, sub_s, "S", "D")) == "C7")
    chk("C7: sign_flip_p refuses NaN instead of returning 0.0",
        tag_of(lambda: sign_flip_p([np.nan, -1.0, -2.0])) == "C7")
    chk("C7: the enumeration is defined at k = 1 (min p = 0.5, not NaN)",
        np.isclose(sign_flip_p([-3.0]), 0.5), f"{sign_flip_p([-3.0]):.4f}")
    chk("C7: k = 2 enumerates 4 assignments", sign_flip_p([-1.0, -2.0]) == 0.25)

    p = build_parser()
    chk("--track and --analyze are mutually exclusive",
        tag_of(lambda: p.parse_args(["--track", "--analyze"])) is not None)
    chk("no mode parses cleanly (defaults to selftest)",
        not p.parse_args([]).track and not p.parse_args([]).analyze)

    # ---------------------------------------------------------------- verdict
    print("\n17 6.1 / 6.2 / 6.4  the verdict as a pure function")
    good = dict(p_beta=0.004, beta_mean=-3.0, p_b=0.004, b_mean=-2.0,
                p_beta_j=0.004, beta_j_mean=-2.5, p_rho=0.004, rho_mean=-0.9, k=8)
    dead_ctrl = dict(good, p_beta=0.5, beta_mean=+0.1)
    chk("all criteria met -> PASS", layer_pass(good)[0])
    chk("a non-significant slope -> FAIL", not layer_pass(dict(good, p_beta=0.3))[0])
    chk("a positive stratified rho -> FAIL", not layer_pass(dict(good, rho_mean=+0.5))[0])
    chk("k = 0 never reaches the matrix: arm_stats stops with C7 instead",
        tag_of(lambda: arm_stats(dict(res, rho=np.array([])))) == "C7")
    chk("...which keeps 'nothing was computable' out of the FAIL cell",
        layer_pass(dict(good, k=0))[1]["strata"] is False)
    # C6, the bug this whole section exists for: an effect that is entirely file
    # order looks perfect until j is controlled for
    order_driven = dict(good, p_beta_j=0.5586, beta_j_mean=+0.742)
    chk("C6: an order-driven effect passed the OLD rule", layer_pass(
        {k: v for k, v in order_driven.items()}, None, alpha=ALPHA)[1]["slope"])
    chk("C6: ...but now FAILS on the covariate criterion",
        not layer_pass(order_driven)[0]
        and not layer_pass(order_driven)[1]["covariate"])
    chk("C6: beta_j significant but POSITIVE is still a failure",
        not main_effect_survives(dict(good, beta_j_mean=+2.0, p_beta_j=0.004)))
    chk("L2 control: a matched channel that also dies kills the layer",
        not layer_pass(good, controls={"R": good})[0]
        and not layer_pass(good, controls={"R": good})[1]["ctl:R"])
    chk("L2 control: a matched channel that survives lets the layer stand",
        layer_pass(good, controls={"R": dead_ctrl})[0])
    chk("decision matrix covers all four cells", len(DECISION_MATRIX) == 4
        and all((a, b) in DECISION_MATRIX for a in (True, False) for b in (True, False)))
    print("\n17 5.3 / C8  cluster-robust sandwich CI")
    nc, gsz = 8, 500
    gid = np.repeat(np.arange(nc), gsz)
    xc = rng.uniform(0, 1, nc * gsz)
    # a strong within-cluster shock: iid SEs would be far too narrow here
    # the shock must be a random SLOPE, not a random intercept: an intercept
    # shift is orthogonal to x within clusters, so it inflates the iid residual
    # variance without inflating the sandwich slope SE - which is the sandwich
    # being right, not the sandwich failing
    yc = (2.0 + (-3.0 + np.repeat(rng.normal(0, 1.0, nc), gsz)) * xc
          + rng.normal(0, .3, nc * gsz))
    Xc = np.column_stack([np.ones(nc * gsz), xc])
    b_c, lo_c, hi_c, G = cluster_robust_ci(Xc, yc, gid, kind="ols")
    chk("sandwich recovers the OLS point estimate", abs(b_c - ols_slope(Xc, yc)[1]) < 1e-9)
    chk("it reports the cluster count, not the sample count", G == nc)
    chk("it covers the true slope", lo_c < -3.0 < hi_c, f"[{lo_c:.3f}, {hi_c:.3f}]")
    resid = yc - Xc @ ols_slope(Xc, yc)
    se_iid = np.sqrt(np.linalg.inv(Xc.T @ Xc)[1, 1] * resid.var(ddof=2))
    se_cl = (hi_c - b_c) / t_dist.ppf(0.975, nc - 1)
    chk("clustering widens the SE versus the iid formula (the point of C8)",
        se_cl > 2 * se_iid, f"cluster {se_cl:.4f} vs iid {se_iid:.4f}")
    ybin = (rng.uniform(size=nc * gsz) < 1 / (1 + np.exp(-(1.0 - 2.0 * xc)))).astype(float)
    b_l, lo_l, hi_l, _ = cluster_robust_ci(Xc, ybin, gid, kind="logit")
    chk("the logit branch runs and covers its true slope", lo_l < -2.0 < hi_l,
        f"{b_l:+.3f} [{lo_l:.3f}, {hi_l:.3f}]")
    chk("C7: a single cluster is refused",
        tag_of(lambda: cluster_robust_ci(Xc[:10], yc[:10], np.zeros(10))) == "C7")

    # end-to-end version of the C6 bug: the effect is 100% acquisition order
    print("\nC6 end-to-end: an effect made entirely of file order must not PASS")
    jn = j_s / j_s.max()
    R_ord = 0.9 * jn + 0.1 * rng.uniform(0, 1, n_sub * n_per)
    s_ord = lab_s * (1.0 - jn) + rng.normal(0, 0.30, n_sub * n_per)
    res_ord = fold_analysis(s_ord, R_ord, j_s, ip_s, sub_s, "ORDER", "D")
    st_ord = arm_stats(res_ord)
    chk("unadjusted, the order-driven effect looks significant",
        st_ord["p_beta"] < ALPHA and st_ord["beta_mean"] < 0,
        f"p_beta={st_ord['p_beta']:.4f} mean={st_ord['beta_mean']:+.3f}")
    chk("controlling for j removes it",
        not main_effect_survives(st_ord),
        f"p_beta_j={st_ord['p_beta_j']:.4f} mean={st_ord['beta_j_mean']:+.3f}")
    chk("so the layer verdict is FAIL, not PASS", not layer_pass(st_ord)[0])
    chk("while the genuinely risk-driven arm still passes", layer_pass(arm_stats(res))[0])

    # ------------------------------------------------- stage 2 on a fixture
    print("\nstage 2 end-to-end on SYNTHETIC fixtures (never on the real artefacts)")
    tmp = tempfile.mkdtemp(prefix="w3_fixture_")
    gests, reps = [0, 1, 2, 3, 8, 9], 60
    rows, rmeta_l = [], []
    for g in gests:
        for su in range(1, 9):
            for jj2 in range(reps):
                rmeta_l.append((g, su, jj2, jj2 // 20, len(rows)))
                rows.append((g, su))
    rmeta_a = np.array(rmeta_l)
    nrow = len(rmeta_a)
    frng = np.random.default_rng(7)
    risk = frng.uniform(0.05, 0.95, (nrow, 3))
    for stem in ("raw", "clutterremoved"):
        np.save(f"{tmp}/y_{stem}.npy", rmeta_a[:, GEST_C])
        np.save(f"{tmp}/subj_{stem}.npy", rmeta_a[:, SUBJ_C])
    np.savez(f"{tmp}/risk.npz", meta=rmeta_a, risk_fit=risk,
             risk_raw=np.clip(risk + frng.normal(0, .05, risk.shape), 0, 1))
    labp = np.isin(rmeta_a[:, GEST_C], [0, 3, 9])          # the class+ gestures
    lab_f = np.where(labp, 1.0, -1.0)
    # D decays with risk; R/Rfit are risk-independent but must still produce BOTH
    # outcomes within every subject - a control channel at 99.9% accuracy has a
    # constant outcome and legitimately trips C5, which is the guard working, not
    # a wiring bug.  0.45/0.50 puts the control near 82% correct.
    scores = {}
    for arm in ("H", "V", "C1"):
        for ch in CHANNELS:
            if ch == "D":
                scores[f"{arm}_{ch}"] = (lab_f * (1.0 - risk.mean(axis=1))
                                         + frng.normal(0, .35, nrow))
            else:
                scores[f"{arm}_{ch}"] = lab_f * 0.45 + frng.normal(0, .50, nrow)
    np.savez(f"{tmp}/scores.npz", meta=rmeta_a, **scores)
    cm = np.isin(rmeta_a[:, GEST_C], [8, 9])
    np.savez(f"{tmp}/c0.npz", idx=rmeta_a[cm][:, ROW_C],
             area=(lab_f * (1.0 - risk.mean(axis=1)) + frng.normal(0, .35, nrow))[cm],
             y=rmeta_a[cm][:, GEST_C], subj=rmeta_a[cm][:, SUBJ_C])
    import io
    import contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            stage2_analyze(f"{tmp}/scores.npz", f"{tmp}/risk.npz", f"{tmp}/c0.npz", tmp)
        ran, err = True, ""
    except BaseException as e:                                        # noqa: BLE001
        ran, err = False, f"{type(e).__name__}: {e}"
    txt = buf.getvalue()
    chk("stage 2 runs end to end (catches wiring bugs before the real run)", ran, err)
    chk("it reaches the frozen decision matrix", "6.4 decision matrix" in txt)
    chk("it prints the 6.3 extension verdicts", "6.3 pre-declared extension" in txt)
    chk("it prints the declared sensitivity analyses",
        "DECLARED SENSITIVITY ANALYSES" in txt and "RAW Viterbi mean" in txt)
    chk("it validates the C0 join against BOTH X_raw and the risk metadata",
        "= the complete G9/G10 set" in txt)
    # assert the VERDICTS, not the presence of the letters "L1"/"L2": the fixture
    # plants a risk-driven D channel and a risk-independent R channel, so L1 and
    # L2 must both PASS and the falsification control must hold
    l1_line = next((l for l in txt.splitlines() if l.strip().startswith("L1 (")), "")
    l2_line = next((l for l in txt.splitlines() if l.strip().startswith("L2 (")), "")
    chk("L1 verdict line reports PASS on the planted effect",
        "PASS" in l1_line and "slope=ok" in l1_line and "covariate=ok" in l1_line,
        l1_line.strip())
    chk("L2 verdict line reports PASS with BOTH controls intact",
        "PASS" in l2_line and "ctl:Dunw=ok" in l2_line and "ctl:R=ok" in l2_line,
        l2_line.strip())
    chk("...and the `18` 5 diagnosis says WRAPPING", "diagnosis: WRAPPING" in txt)
    chk("the matched R channel is NOT significant in the fixture",
        "-> FAIL" in next((l for l in txt.splitlines()
                           if l.strip().startswith("  V (")), "-> FAIL"))
    def rerun_with(mutate, prefix):
        d = tempfile.mkdtemp(prefix=prefix)
        for f2 in ("risk.npz", "c0.npz", "y_raw.npy", "subj_raw.npy",
                   "y_clutterremoved.npy", "subj_clutterremoved.npy"):
            shutil.copy(f"{tmp}/{f2}", f"{d}/{f2}")
        np.savez(f"{d}/scores.npz", meta=rmeta_a, **mutate(dict(scores)))
        b = io.StringIO()
        with contextlib.redirect_stdout(b):
            stage2_analyze(f"{d}/scores.npz", f"{d}/risk.npz", f"{d}/c0.npz", d)
        return b.getvalue(), d

    def kill_r(s):
        for a in ("H", "V", "C1"):
            s[f"{a}_R"] = s[f"{a}_D"].copy()
        return s

    def kill_phase(s):
        for a in ("H", "V", "C1"):
            s[f"{a}_Dunw"] = s[f"{a}_D"].copy()
        return s

    t2, tmp2 = rerun_with(kill_r, "w3_fixture2_")
    l2b = next((l for l in t2.splitlines() if l.strip().startswith("L2 (")), "")
    chk("if the range control dies too, ctl:R triggers and L2 fails",
        "ctl:R=NO" in l2b and "FAIL" in l2b, l2b.strip())
    chk("...and the diagnosis names it rather than silently failing",
        "diagnosis: INCOHERENT" in t2 or "TRACKING" in t2)
    t4, tmp4 = rerun_with(kill_phase, "w3_fixture4_")
    l2c = next((l for l in t4.splitlines() if l.strip().startswith("L2 (")), "")
    chk("`18` 5 in situ: if Dunw dies with D, L2 FAILS even though R is fine",
        "ctl:Dunw=NO" in l2c and "ctl:R=ok" in l2c and "FAIL" in l2c, l2c.strip())
    chk("...and the diagnosis is PHASE DISTORTION, not wrapping",
        "diagnosis: PHASE DISTORTION" in t4)

    # integrity gate: a truncated score file must not be analysed
    tmp3 = tempfile.mkdtemp(prefix="w3_fixture3_")
    for src_f in ("risk.npz", "c0.npz", "y_raw.npy", "subj_raw.npy",
                  "y_clutterremoved.npy", "subj_clutterremoved.npy"):
        shutil.copy(f"{tmp}/{src_f}", f"{tmp3}/{src_f}")
    cut = rmeta_a[:, ORD_C] < reps - 5                 # drop 5 reps per cell
    np.savez(f"{tmp3}/scores.npz", meta=rmeta_a[cut],
             **{k: v[cut] for k, v in scores.items()})
    chk("C7: a truncated score file is refused even though all 8 subjects are present",
        tag_of(lambda: stage2_analyze(f"{tmp3}/scores.npz", f"{tmp3}/risk.npz",
                                      f"{tmp3}/c0.npz", tmp3)) == "C7")
    for d in (tmp, tmp2, tmp3, tmp4):
        shutil.rmtree(d, ignore_errors=True)

    # ---------------------------------------------------- stage 1 scoring path
    # sample_scores(..., B) - including the Rfit channel - is otherwise never
    # executed by any test, so it would first run on real echoes.
    print("\nstage 1 scoring path on a SYNTHETIC echo cube (no cache, no template file)")
    Msyn, Tsyn = 189, 90
    kk = np.arange(Msyn) - Msyn // 2
    tmpl = (np.exp(-(kk ** 2) / (2 * 3.0 ** 2)) * np.cos(2 * np.pi * 0.3203 * kk),
            Msyn // 2)
    th2 = np.linspace(0, 2 * np.pi, Tsyn)
    traj = np.stack([0.12 * np.cos(th2), 0.12 * np.sin(th2),
                     0.45 + 0.03 * np.sin(th2)], axis=1)
    frames = fwd_clutter(fwd_render(traj, template=tmpl, noise=0.01,
                                    rng=np.random.default_rng(3)))
    Bsyn = bspline_basis(Tsyn, K_CTRL)
    got = sample_scores(frames, B=Bsyn)
    chk(f"sample_scores returns all {3 * len(CHANNELS)} (arm, channel) keys",
        set(got) == {(a, c) for a in ("H", "V", "C1") for c in CHANNELS},
        f"{sorted(got)}")
    chk("every score is finite", all(np.isfinite(v) for v in got.values()))
    chk("the Rfit channel really ran (differs from RAW R)",
        got[("C1", "Rfit")] != got[("C1", "R")],
        f"Rfit={got[('C1', 'Rfit')]:.6g} vs R={got[('C1', 'R')]:.6g}")
    chk("a CCW synthetic loop gives a positive C1 area in the RANGE channels",
        got[("C1", "R")] > 0 and got[("C1", "Rfit")] > 0,
        f"R={got[('C1', 'R')]:.6f} Rfit={got[('C1', 'Rfit')]:.6f}")

    # ------------------------------------------------------------------ `18`
    print("\n`18` acceptance: the integer-bin phase correction")
    print("  both criteria are on SYNTHETIC echoes, before any real run")

    def probe(amp, strokes, seed=3, template=None, carrier=CARRIER_CYC_PER_BIN):
        """Render a synthetic gesture and measure D, Dunw and R against the truth."""
        th_p = np.linspace(0, 2 * np.pi * strokes, Tsyn)
        tr = np.stack([amp * np.cos(th_p), amp * np.sin(th_p),
                       0.45 + 0.03 * np.sin(np.linspace(0, 2 * np.pi, Tsyn))], axis=1)
        fr = fwd_clutter(fwd_render(tr, template=template or tmpl, noise=0.01,
                                    rng=np.random.default_rng(seed)))
        rho_p = np.linalg.norm(tr[None] - RADAR_POS[:, None], axis=2)
        vt = np.diff(rho_p, axis=1) / DT
        wrap = float((np.abs(np.diff(rho_p, axis=1)) > LAMBDA_M / 4).mean())
        cD, cU, cR, hops, vDmax, kbad, knz = [], [], [], [], [], [], []
        for v in range(3):
            r_v, _c, pth = viterbi_track(fr[v], return_path=True)
            zb = _analytic(fr[v].astype(np.float64))[np.arange(Tsyn), pth]
            vD = velocity_from_phase(zb, pth, carrier)
            vU = velocity_from_phase_unwrapped(zb, pth, r_v, carrier)
            cD.append(np.corrcoef(vD, vt[v])[0, 1])
            cU.append(np.corrcoef(vU, vt[v])[0, 1])
            cR.append(np.corrcoef(velocity_from_range(r_v), vt[v])[0, 1])
            hops.append(int((np.diff(pth) != 0).sum()))
            vDmax.append(float(np.abs(vD).max()))
            # how often the RAW-guided branch differs from the branch the NOISELESS
            # geometry would have chosen: the honest error rate of the `18` 5 control
            d_ph = phase_increments(zb, pth, carrier)
            k_used = np.round((-2 * np.pi * carrier * np.diff(r_v) / BIN_M - d_ph)
                              / (2 * np.pi))
            k_true = np.round((-2 * np.pi * carrier * np.diff(rho_p[v]) / BIN_M - d_ph)
                              / (2 * np.pi))
            knz.append(int((k_used != 0).sum()))
            kbad.append(int((k_used != k_true).sum()))
        return dict(wrap=wrap, vmax=float(np.abs(vt).max()), cD=cD, cU=cU, cR=cR,
                    hops=hops, vDmax=vDmax, kbad=kbad, knz=knz, n=Tsyn - 1)

    lim = LAMBDA_M / (4 * DT)
    lo = probe(0.12, 1.0)
    chk("(1) below the threshold: 0% wrap, yet the bin moves constantly",
        lo["wrap"] == 0.0 and min(lo["hops"]) > 20,
        f"wrap={100 * lo['wrap']:.1f}%  max|v|={lo['vmax']:.4f} < {lim:.4f}  "
        f"bin changes per view {lo['hops']}/{Tsyn - 1}")
    chk("(1) v^D now tracks the truth over ALL frames, not just static bins",
        all(abs(c) >= 0.9 for c in lo["cD"]), f"|corr| = {np.round(np.abs(lo['cD']), 3)}")
    hi = probe(0.15, 2.5)
    chk("(2) above the threshold the truth really does exceed the limit",
        hi["wrap"] > 0.3 and hi["vmax"] > lim,
        f"wrap={100 * hi['wrap']:.1f}%  max|v|={hi['vmax']:.4f} > {lim:.4f}")
    chk("(2) only the phase channel folds: |v^D| stays inside lambda/(4dt)",
        all(m <= lim + 1e-9 for m in hi["vDmax"]), f"max|v^D| = {np.round(hi['vDmax'], 4)}")
    chk("(2) the range channel keeps following the truth through the wrap",
        all(c > 0.9 for c in hi["cR"]), f"corr(R) = {np.round(hi['cR'], 3)}")
    chk("(2) ...while the phase channel's agreement collapses (this is the effect"
        " L2 exists to measure)",
        np.mean(np.abs(hi["cD"])) < np.mean(np.abs(lo["cD"])),
        f"|corr(D)| {np.mean(np.abs(hi['cD'])):.3f} vs {np.mean(np.abs(lo['cD'])):.3f} below")

    # ---- `18` 5: D_unwrapped, the control that isolates wrapping from the other
    # ways a phase channel can rot
    print("\n`18` 5  D_unwrapped: the same phase with the 2*pi branch restored")
    chk("below the threshold almost every frame needs no branch at all",
        sum(lo["knz"]) <= 0.05 * 3 * lo["n"],
        f"k != 0 on {lo['knz']} of {lo['n']} frames per view")
    chk("below the threshold D_unwrapped tracks the truth like D does",
        all(abs(c) >= 0.9 for c in lo["cU"]),
        f"D {np.round(np.abs(lo['cD']), 3)} vs Dunw {np.round(np.abs(lo['cU']), 3)}")
    chk("above the threshold D_unwrapped SURVIVES where D folds",
        all(abs(c) > 0.9 for c in hi["cU"]) and np.mean(np.abs(hi["cD"])) < 0.5,
        f"Dunw {np.round(np.abs(hi['cU']), 3)} vs D {np.round(np.abs(hi['cD']), 3)}")
    # the branch is picked from a NOISY range track, so it is not infallible; the
    # rate is measured and asserted rather than assumed away
    chk("the RAW-guided branch matches the noiseless-geometry branch >= 95% of frames",
        all(b <= 0.05 * lo["n"] for b in lo["kbad"] + hi["kbad"]),
        f"wrong branches: below {lo['kbad']}, above {hi['kbad']}, of {lo['n']}")
    # np.unwrap would assume every true increment is inside +-pi, i.e. exactly the
    # thing under test; with 4.0 rad steps it reconstructs the WRONG branch while
    # the range-guided choice gets it right
    step, npts = 4.0, 30
    z_fast = np.exp(1j * np.cumsum(np.full(npts, step)))
    dr = -step * BIN_M / (2 * np.pi * CARRIER_CYC_PER_BIN)     # range implying +4.0 rad
    r_fast = np.cumsum(np.full(npts, dr))
    v_branch = velocity_from_phase_unwrapped(z_fast, np.zeros(npts), r_fast)
    v_unwrap = LAMBDA_M / (4 * np.pi * DT) * np.diff(np.unwrap(np.angle(z_fast)))
    chk("range-guided branching recovers a >pi increment that np.unwrap cannot",
        np.allclose(v_branch, LAMBDA_M / (4 * np.pi * DT) * step)
        and not np.allclose(v_branch, v_unwrap),
        f"branch {v_branch[0]:+.5f} vs np.unwrap {v_unwrap[0]:+.5f} "
        f"(truth {LAMBDA_M / (4 * np.pi * DT) * step:+.5f})")
    # the three-way table itself, on synthetic summaries
    die = dict(p_beta=0.004, beta_mean=-3.0)
    live = dict(p_beta=0.6, beta_mean=-0.1)
    chk("`18` 5 table: D dies alone -> WRAPPING",
        l2_diagnosis(die, live, live)[0] == "WRAPPING")
    chk("`18` 5 table: D and Dunw both die -> PHASE DISTORTION, not wrapping",
        l2_diagnosis(die, die, live)[0] == "PHASE DISTORTION")
    chk("`18` 5 table: everything dies -> TRACKING / MOTION CONFOUND",
        l2_diagnosis(die, die, die)[0] == "TRACKING / MOTION CONFOUND")
    chk("`18` 5 table: D never degrades -> NO EFFECT",
        l2_diagnosis(live, live, live)[0] == "NO EFFECT")
    good2 = dict(p_beta=0.004, beta_mean=-3.0, p_b=0.004, b_mean=-2.0,
                 p_beta_j=0.004, beta_j_mean=-2.5, p_rho=0.004, rho_mean=-0.9, k=8)
    chk("a Dunw that dies alongside D now FAILS the layer (R alone could not see it)",
        not layer_pass(good2, controls={"Dunw": good2, "R": live})[0])
    chk("...and with Dunw intact the layer stands",
        layer_pass(good2, controls={"Dunw": live, "R": live},
                   paired={"Dunw": (0.004, -2.0), "R": (0.004, -2.0)})[0])

    # ---- `18` 5b: the paired difference, not two separate significance verdicts
    print("\n`18` 5b  paired per-subject channel differences")
    big = dict(beta=np.array([-3.0, -3.2, -2.8, -3.1, -2.9, -3.3, -2.7, -3.0]))
    smallish = dict(beta=np.array([-2.6, -2.9, -2.5, -2.7, -2.6, -3.0, -2.4, -2.7]))
    none_ = dict(beta=np.array([0.1, -0.2, 0.3, 0.0, -0.1, 0.2, -0.3, 0.1]))
    p_u, m_u, d_u = paired_delta(big, none_)
    chk("a real gap between channels gives a significant paired delta",
        p_u < ALPHA and m_u < 0, f"mean {m_u:+.3f}, p = {p_u:.4f}")
    p_s, m_s, _ = paired_delta(big, smallish)
    chk("a SMALL but consistent gap is still detected (paired, so noise cancels)",
        p_s < ALPHA and m_s < 0, f"mean {m_s:+.3f}, p = {p_s:.4f}")
    p_n, m_n, _ = paired_delta(big, big)
    chk("identical channels give no difference at all", p_n == 1.0 and m_n == 0.0,
        f"mean {m_n:+.3f}, p = {p_n:.4f}")
    chk("C3: the paired test refuses anything but 8 paired slopes",
        tag_of(lambda: paired_delta(big, dict(beta=big["beta"][:7]))) == "C3")
    # the exact trap: D clears 0.05, the control does not, yet the DIFFERENCE is
    # not established.  Old code called that WRAPPING; it must not.
    marg = dict(p_beta=0.004, beta_mean=-1.0)
    ctl_marg = dict(p_beta=0.07, beta_mean=-0.9)
    chk("'one significant, one not' alone no longer reads as WRAPPING",
        l2_diagnosis(marg, ctl_marg, live,
                     paired={"Dunw": (0.55, -0.1), "R": (0.004, -2.0)})[0]
        == "NOT SEPARATED")
    chk("...and the layer fails on that same ground",
        not layer_pass(good2, controls={"Dunw": live, "R": live},
                       paired={"Dunw": (0.55, -0.1), "R": (0.004, -2.0)})[0])
    chk("with both paired differences established it reads as WRAPPING",
        l2_diagnosis(marg, live, live,
                     paired={"Dunw": (0.004, -2.0), "R": (0.004, -2.0)})[0] == "WRAPPING")

    # ---- non-circular acceptance: the de-rotation must not be validated with the
    # same constant that generated the pulse
    print("\n`18` 6  non-circular carrier acceptance")
    for f_gen in (0.28, 0.36):
        kk2 = np.arange(Msyn) - Msyn // 2
        t_alt = (np.exp(-(kk2 ** 2) / (2 * 3.0 ** 2))
                 * np.cos(2 * np.pi * f_gen * kk2), Msyn // 2)
        pr = probe(0.12, 1.0, template=t_alt, carrier=f_gen)
        chk(f"a pulse generated AND de-rotated at f_s={f_gen} still tracks",
            all(abs(c) >= 0.9 for c in pr["cD"]), f"|corr| = {np.round(np.abs(pr['cD']), 3)}")
    mism = probe(0.12, 1.0, carrier=CARRIER_ALT)
    chk(f"de-rotating with the alternative constant {CARRIER_ALT} (the `carrier` "
        "FIELD of w2_pulse.npz, 7.6199 GHz converted; `13` 4.2) barely moves it",
        all(abs(c) >= 0.9 for c in mism["cD"]),
        f"|corr| = {np.round(np.abs(mism['cD']), 3)} vs "
        f"{np.round(np.abs(lo['cD']), 3)} at {CARRIER_CYC_PER_BIN}")
    # `18` 7 lists this as a MUST-PASS gate, so a missing or malformed template is
    # a hard failure.  Printing SKIPPED and still reporting ALL PASS would let the
    # freeze proceed on a gate that never ran.
    if not Path(PULSE_NPZ).is_file():
        raise ProtocolStop("C7", f"{PULSE_NPZ} is missing; `18` 7 lists the "
                           "measured-template acceptance as mandatory")
    meas = np.load(PULSE_NPZ)
    for fld in ("template", "W"):
        if fld not in meas.files:
            raise ProtocolStop("C7", f"{PULSE_NPZ} has no '{fld}' field")
    tm = np.real(meas["template"])
    Wm = int(meas["W"])
    if tm.ndim != 1 or tm.shape[0] != 2 * Wm + 1 or not np.all(np.isfinite(tm)):
        raise ProtocolStop("C7", f"{PULSE_NPZ}: template shape {tm.shape} is not "
                           f"2*W+1 = {2 * Wm + 1}, or is non-finite")
    tm = tm / np.abs(tm).max()
    full = np.zeros(Msyn)
    full[Msyn // 2 - Wm:Msyn // 2 + Wm + 1] = tm
    pr = probe(0.12, 1.0, template=(full, Msyn // 2))
    chk("the MEASURED pulse template (logs/w2_pulse.npz - a calibration artefact: "
        "one pulse waveform, no gesture or subject labels, no risk scores) tracks",
        all(abs(c) >= 0.9 for c in pr["cD"]),
        f"|corr| = {np.round(np.abs(pr['cD']), 3)}")
    pr_hi = probe(0.15, 2.5, template=(full, Msyn // 2))
    sep = np.mean(np.abs(pr_hi["cU"])) - np.mean(np.abs(pr_hi["cD"]))
    chk("...and with it, D still folds above the threshold while Dunw survives",
        np.mean(np.abs(pr_hi["cD"])) < 0.5 and all(abs(c) > 0.85 for c in pr_hi["cU"])
        and sep > 0.5,
        f"D {np.round(np.abs(pr_hi['cD']), 3)} Dunw {np.round(np.abs(pr_hi['cU']), 3)}"
        f"  separation {sep:.3f}")
    chk("the correction is a no-op when the bin never moves (`18` 2)",
        np.array_equal(velocity_from_phase(np.exp(1j * np.arange(20.0)), np.zeros(20)),
                       LAMBDA_M / (4 * np.pi * DT) * np.angle(
                           np.exp(1j * np.arange(1.0, 20)) * np.conj(np.exp(1j * np.arange(19.0))))))

    # the phase-sign convention is left alone (`18` 3); that it cannot move a
    # verdict is proved, not hoped:
    chk("a global sign flip leaves the C1 area EXACTLY unchanged",
        score_c1(-px, -py) == score_c1(px, py))
    s_flip = -s_s
    chk("a global sign flip leaves correct/margin unchanged (sigma absorbs it)",
        np.array_equal(polarity(s_flip, ip_s, sub_s != 1) * s_flip,
                       -polarity(s_s, ip_s, sub_s != 1) * -s_s))
    chk("so H/V and C1 verdicts are invariant to the phase convention",
        np.allclose(fold_analysis(s_flip, R_s, j_s, ip_s, sub_s, "S", "D")["beta"],
                    res["beta"]))

    print("\n" + "=" * 78)
    print(f"  SELFTEST: {'ALL PASS' if chk.fails == 0 else f'{chk.fails} FAILURE(S)'}")
    print("=" * 78)
    if chk.fails:
        raise SystemExit(1)
    print("  Formulas verified against 17 3-5 on synthetic inputs.")
    print("  No real data was read; no L1/L2 result exists yet.")


# ===========================================================================
# 6. stage 1 - per-sample scores.  Reads the echo cache and NOTHING else.
# ===========================================================================
def cell_rows(y, subj, g, s, reps=100):
    """The rows of one (gesture, subject) cell, in cache order.

    Reproduces the selection of tools/traj_w2_readout_risk.py verbatim so that
    the file-order index j means the same thing in both files.  Derived here
    rather than copied out of the risk npz so that stage 1 never opens it;
    stage 2 asserts the two agree row for row.
    """
    return np.where((y == g) & (subj == s))[0][:reps]


def stage1_track():
    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    subj = np.load(f"{CACHE}/subj_clutterremoved.npy")

    gestures = sorted({a["pos"] for a in ARMS.values()} | {a["neg"] for a in ARMS.values()})
    keys = [(a, c) for a in ("H", "V", "C1") for c in CHANNELS]
    meta, cols = [], {k: [] for k in keys}
    B = bspline_basis(90, K_CTRL)

    print("=" * 78)
    print("STAGE 1  per-sample readout scores (17 3)")
    print("=" * 78)
    print(f"  gestures {gestures}   channels {CHANNELS}   no risk score is read here\n")
    t0 = time.time()
    for g in gestures:
        for s in range(1, 9):
            for j, i in enumerate(cell_rows(y, subj, g, s)):
                sc = sample_scores(np.asarray(X[i]), B=B)
                for k in keys:
                    cols[k].append(sc[k])
                meta.append((g, s, j, j // 20, i))
        print(f"  G{g + 1} done  ({time.time() - t0:.0f}s)", flush=True)

    out = {f"{a}_{c}": np.array(cols[(a, c)]) for a, c in keys}
    np.savez(SCORES_NPZ, meta=np.array(meta), **out)
    print(f"\n  {len(meta)} samples x {len(keys)} scores   [saved] {SCORES_NPZ}")
    print("  stage 1 produced scores only; no risk-readout association exists yet.")


# ===========================================================================
# 7. stage 2 - the L1/L2 analysis (17 4, 5, 6)
# ===========================================================================
def paired_risk(risk, how, agg="mean"):
    """17 2: R^H = mean(L,R), R^V = T, R^C = mean(L,T,R).

    `agg="max"` is the declared sensitivity analysis of 17 2 ("以 max 替代均值");
    it never replaces the primary criterion.  A single view id ("L"/"T"/"R") is
    the C-arm per-view breakdown that the same clause asks for.
    """
    if how in ("L", "T", "R"):
        return risk[:, {"L": 0, "T": 1, "R": 2}[how]]
    cols = [0, 2] if how == "LR" else [0, 1, 2]
    sub = risk[:, cols]
    return sub.max(axis=1) if agg == "max" else sub.mean(axis=1)


def fold_analysis(s, R, j, is_pos, sub, arm, chan):
    """One arm/channel -> per-subject slopes and per-fold stratified rho (17 4, 5)."""
    subjects = np.unique(sub)
    if len(subjects) != 8:
        raise ProtocolStop("C3", f"{arm}/{chan} has {len(subjects)} subjects, expected 8")
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(R))):
        raise ProtocolStop("C7", f"{arm}/{chan} has non-finite scores or risks")

    betas, bs, betas_j, rhos, notes = [], [], [], [], []
    sigmas, miss_counts, pool = [], [], []
    for u in subjects:
        est, test = sub != u, sub == u
        sigma = polarity(s, is_pos, est)
        if sigma == 0.0:
            raise ProtocolStop("C1", f"{arm}/{chan} fold {u} has sigma = 0 (exact "
                               "tie); the arm is unusable and no direction may be chosen")
        sd = mad_scale(s[est])
        if sd == 0.0:
            raise ProtocolStop("C4", f"{arm}/{chan} fold {u} has an estimation-set "
                               "MAD of 0; the margin is undefined")
        edges = strata_edges(R[est])

        lab = np.where(is_pos, 1.0, -1.0)
        correct = (sigma * s * lab > 0).astype(float)[test]
        margin = (sigma * s * lab / sd)[test]
        Rt, jt = R[test], j[test].astype(float)

        b_log, conv = irls_logit(np.column_stack([np.ones(test.sum()), Rt]), correct)
        if b_log is None or not conv:
            raise ProtocolStop("C5", f"{arm}/{chan} fold {u} logistic fit failed "
                               f"(converged={conv}); not imputed")
        b_lin = ols_slope(np.column_stack([np.ones(test.sum()), Rt]), margin)
        b_cov, conv_c = irls_logit(np.column_stack([np.ones(test.sum()), Rt, jt]), correct)
        if b_cov is None or not conv_c:
            raise ProtocolStop("C5", f"{arm}/{chan} fold {u} covariate fit failed")
        betas.append(b_log[1])
        bs.append(b_lin[1])
        betas_j.append(b_cov[1])
        sigmas.append(sigma)
        # each sample is pooled exactly once, from the fold where it was held out
        pool.append((Rt, correct, margin, np.full(test.sum(), u, float)))

        # stratified d' on the test subject, edges frozen from the estimation set
        st = assign_strata(Rt, edges)
        usable, missing = merge_strata(st, is_pos[test])
        miss_counts.append(len(missing))
        if len(usable) < MIN_STRATA:
            notes.append(f"subject {u}: only {len(usable)} usable strata "
                         f"(missing {missing}) - fold excluded per 5.1")
            continue
        if missing:
            notes.append(f"subject {u}: kept with {len(usable)} strata, "
                         f"{len(missing)} recorded missing {missing} (5.1)")
        xs = [Rt[np.isin(st, g)].mean() for g in usable]
        ds = [signed_dprime(s[test][np.isin(st, g)], is_pos[test][np.isin(st, g)], sigma)
              for g in usable]
        if not np.all(np.isfinite(ds)):
            raise ProtocolStop("C7", f"{arm}/{chan} fold {u} produced a non-finite "
                               f"stratified d': {ds}")
        rho_u = float(spearmanr(xs, ds).statistic)
        if not np.isfinite(rho_u):
            raise ProtocolStop("C7", f"{arm}/{chan} fold {u} produced a non-finite "
                               "Spearman rho (constant d' across strata?)")
        rhos.append(rho_u)

    Rp, cp, mp, gp = (np.concatenate([p[i] for p in pool]) for i in range(4))
    return dict(beta=np.array(betas), b=np.array(bs), beta_j=np.array(betas_j),
                rho=np.array(rhos), notes=notes, sigma=np.array(sigmas),
                missing=np.array(miss_counts),
                pool_X=np.column_stack([np.ones(len(Rp)), Rp]),
                pool_correct=cp, pool_margin=mp, pool_sub=gp)


# ---------------------------------------------------------------------------
# the verdict, as pure functions of summary numbers so it can be tested on
# synthetic input without any of the machinery above
# ---------------------------------------------------------------------------
def arm_stats(res):
    """Everything a verdict is allowed to look at, reduced to plain numbers.

    k == 0 is a hard C7 error rather than a NaN that flows on into the four-cell
    matrix: "the experiment answered no" and "there was nothing to compute" are
    different conclusions, and a FAIL cell cannot be allowed to mean either one.
    """
    if len(res["rho"]) == 0:
        raise ProtocolStop("C7", "no fold produced a usable stratified rho; the "
                           "analysis is undefined, which is NOT the same as a FAIL")
    st = dict(
        p_beta=sign_flip_p(res["beta"]), beta_mean=float(res["beta"].mean()),
        p_b=sign_flip_p(res["b"]), b_mean=float(res["b"].mean()),
        p_beta_j=sign_flip_p(res["beta_j"]), beta_j_mean=float(res["beta_j"].mean()),
        k=len(res["rho"]),
        p_rho=sign_flip_p(res["rho"]), rho_mean=float(res["rho"].mean()),
    )
    st["ci_beta"] = cluster_robust_ci(res["pool_X"], res["pool_correct"],
                                      res["pool_sub"], kind="logit")
    st["ci_b"] = cluster_robust_ci(res["pool_X"], res["pool_margin"],
                                   res["pool_sub"], kind="ols")
    return st


def main_effect_survives(st, alpha=ALPHA):
    """C6: is the effect still there once file order is controlled for?"""
    return bool(st["beta_j_mean"] < 0 and st["p_beta_j"] < alpha)


def degrades(st, alpha=ALPHA):
    """Does this channel's readout get worse as risk rises?  (`18` 5)"""
    return bool(st["p_beta"] < alpha and st["beta_mean"] < 0)


def paired_delta(res_main, res_ctrl):
    """`18` 5b: the WITHIN-SUBJECT difference of slopes, and its sign-flip p.

    "D is significant and the control is not" is not evidence that they differ -
    it is the classic comparison of two significance verdicts instead of a test
    of the difference.  The subjects are the same eight people in the same order,
    so the slopes pair exactly and the difference can be tested directly:

        delta_u = beta_u^D - beta_u^control,   H1: mean(delta) < 0

    Same one-sided exact enumeration as everything else (17 5.2).
    """
    a, b = res_main["beta"], res_ctrl["beta"]
    if len(a) != len(b) or len(a) != 8:
        raise ProtocolStop("C3", f"paired test needs 8 slopes on each side, "
                           f"got {len(a)} and {len(b)}")
    d = a - b
    return sign_flip_p(d), float(d.mean()), d


def layer_pass(st, controls=None, paired=None, alpha=ALPHA):
    """17 6.1 / 6.2 (as amended by `18` 5, 5b) as a pure predicate.

    `controls` maps a name to that channel's summary; every control must fail to
    degrade.  `paired` maps a name to (p, mean_delta) from `paired_delta`; every
    paired difference must be significantly negative.  Both are required: the
    control clause alone accepts "D significant, control not", which is not a
    test of any difference (`18` 5b).
    """
    r = dict(
        slope=st["p_beta"] < alpha,
        strata=bool(st["k"]) and st["rho_mean"] < 0 and st["p_rho"] < alpha,
        covariate=main_effect_survives(st, alpha),
    )
    for name, cst in (controls or {}).items():
        r[f"ctl:{name}"] = not degrades(cst, alpha)
    for name, (p, m) in (paired or {}).items():
        r[f"diff:{name}"] = bool(p < alpha and m < 0)
    return all(r.values()), r


L2_DIAGNOSIS = {
    ("die", "live", "live"): ("WRAPPING", "only the ambiguous channel degrades; no "
                              "significant degradation was detected in the same phase "
                              "with the branch restored, nor in the range channel "
                              "-> 17 6.2 attribution stands"),
    ("die", "die", "live"): ("PHASE DISTORTION", "the phase channel rots with or "
                             "without the ambiguity, so the cause is not wrapping "
                             "-> L2 FAIL (`18` 5)"),
    ("die", "die", "die"): ("TRACKING / MOTION CONFOUND", "everything degrades "
                            "together, so the risk score is tracking vigour of "
                            "motion, not wrapping -> L2 FAIL (17 6.5)"),
    ("die", "live", "die"): ("INCOHERENT", "the wrapped phase and the range channel "
                             "die while the unwrapped phase survives; not a pattern "
                             "the design anticipates -> L2 FAIL, report as such"),
}


def l2_diagnosis(d_st, dunw_st, r_st, paired=None, alpha=ALPHA):
    """Which of `18` 5's explanations the three channels actually support.

    A WRAPPING call additionally requires the paired differences of `18` 5b: the
    per-channel verdicts alone cannot distinguish "D degrades more than the
    controls" from "D happened to clear 0.05 and they happened not to".
    """
    key = tuple("die" if degrades(s, alpha) else "live"
                for s in (d_st, dunw_st, r_st))
    if key[0] == "live":
        return "NO EFFECT", "the wrapped phase channel does not degrade with risk"
    out = L2_DIAGNOSIS[key]
    if out[0] == "WRAPPING" and paired is not None:
        weak = [n for n, (p, m) in paired.items() if not (p < alpha and m < 0)]
        if weak:
            return ("NOT SEPARATED",
                    f"D degrades and the controls do not, but the paired difference "
                    f"is not significant for {weak} - `18` 5b forbids reading that as "
                    "a demonstrated difference between channels")
    return out


DECISION_MATRIX = {
    (True, True): "complete claim stands: the recorded failure was predictable AND "
                  "the mechanism is wrapping; framework 1/3 -> 2/3, the third "
                  "research point has grounds",
    (True, False): "predictable, but the mechanism is NOT attributable to wrapping; "
                   "rewrite `07` 2 and demote dead end #10's rationale from "
                   "'wrapping' to 'empirical fact'",
    (False, True): "wrapping holds under the matched definition but does NOT connect "
                   "to the recorded failure ledger; weak, insufficient on its own",
    (False, False): "negative: keep as 'partial failure predictability and its "
                    "theoretical limits'; do NOT prop up the third point",
}


def report_arm(name, chan, res):
    """Print one arm/channel and return its summary numbers."""
    st = arm_stats(res)
    exp = EXPECTED_SIGMA.get(name)
    flipped = [int(u) for u, s in zip(range(1, 9), res["sigma"]) if exp and s != exp]
    print(f"\n  {name}/{chan}")
    print(f"    sigma per fold       : {np.round(res['sigma'], 0).astype(int)}"
          + (f"   (17 3.5 geometric expectation {exp:+.0f}"
             + (f"; DIFFERS on subjects {flipped} - recorded as required, the "
                "estimation set governs)" if flipped else "; all agree)")
             if exp else ""))
    print(f"    beta_u (correct ~ R) : {np.round(res['beta'], 3)}")
    print(f"      mean {st['beta_mean']:+.4f}   sign-flip p = {st['p_beta']:.4f}"
          f"   [{'p<0.05' if st['p_beta'] < ALPHA else 'n.s.'}]")
    print(f"      cluster-robust (C8, G={st['ci_beta'][3]}, t_{st['ci_beta'][3] - 1}) "
          f"pooled slope {st['ci_beta'][0]:+.3f}  95% CI [{st['ci_beta'][1]:+.3f}, "
          f"{st['ci_beta'][2]:+.3f}]")
    print(f"        CAVEAT: only {st['ci_beta'][3]} clusters - the sandwich's "
          "asymptotics are NOT in reach here.  Supplementary; the decision rests")
    print("        on the exact sign-flip enumeration, which needs no parametric")
    print("        form - though it does assume per-subject sign symmetry under H0.")
    print(f"    b_u (margin ~ R)     : {np.round(res['b'], 3)}")
    print(f"      mean {st['b_mean']:+.4f}   sign-flip p = {st['p_b']:.4f}"
          f"   [{'p<0.05' if st['p_b'] < ALPHA else 'n.s.'}]")
    print(f"      cluster-robust pooled slope {st['ci_b'][0]:+.3f}  95% CI "
          f"[{st['ci_b'][1]:+.3f}, {st['ci_b'][2]:+.3f}]")
    print(f"    beta_u | file-order j: {np.round(res['beta_j'], 3)}")
    print(f"      mean {st['beta_j_mean']:+.4f}   sign-flip p = {st['p_beta_j']:.4f}"
          f"   -> main effect {'SURVIVES' if main_effect_survives(st) else 'GONE (C6)'}")
    print(f"    stratified rho       : k = {st['k']}/8 folds"
          + ("" if st["k"] == 8 else "   (C3: reduced enumeration, reported not absorbed)")
          + f"   missing strata per fold {res['missing'].tolist()}")
    if st["k"]:
        print(f"      {np.round(res['rho'], 3)}   mean {st['rho_mean']:+.4f}"
              f"   sign-flip p = {st['p_rho']:.4f}")
    for n in res["notes"]:
        print(f"      ! {n}")
    return st


def _report_layer(tag, arm, st, controls=None, paired=None):
    ok, why = layer_pass(st, controls, paired)
    bits = "  ".join(f"{k}={'ok' if v else 'NO'}" for k, v in why.items())
    print(f"\n  {tag} ({arm}): {bits}   -> {'PASS' if ok else 'FAIL'}")
    if not why["covariate"]:
        print("     C6 / 17 5.4: the effect does not survive the file-order control, so")
        print("     the finding is 'risk is collinear with file order, not separable'.")
    for name, cst in (controls or {}).items():
        if degrades(cst):
            print(f"     control '{name}' degraded too (beta {cst['beta_mean']:+.3f}, "
                  f"p = {cst['p_beta']:.4f}) - see the diagnosis below")
    for name, (p, m) in (paired or {}).items():
        print(f"     paired delta_u = beta^D - beta^{name}: mean {m:+.4f}, "
              f"sign-flip p = {p:.4f}"
              f"   [{'D degrades MORE' if p < ALPHA and m < 0 else 'difference NOT shown'}]")
    return ok


def stage2_analyze(scores_npz=SCORES_NPZ, risk_npz=RISK_NPZ, c0_npz=C0_NPZ,
                   cache=CACHE):
    """The paths are parameters so the whole path can be exercised on synthetic
    fixtures.  Defaults are the real artefacts; the selftest passes a temp dir.
    Without this the reporting and verdict wiring would only ever run for the
    first time on the real numbers, which is precisely when a crash is worst."""
    sc = np.load(scores_npz)
    rk = np.load(risk_npz)
    c0 = np.load(c0_npz)
    smeta, rmeta = sc["meta"], rk["meta"]

    print("=" * 78)
    print("STAGE 2  L1/L2 analysis (17 4-6).  Decision matrix frozen in 6.4.")
    print("=" * 78)

    # ---------------------------------------------------------------- integrity
    # "every score row was found in the risk table" is a much weaker statement
    # than "the score table IS the target set".  A truncated stage 1 that still
    # happens to contain all eight subjects would pass the weak version and be
    # analysed as if complete, so the target set is reconstructed and compared.
    gest_needed = sorted({a["pos"] for a in ARMS.values()} | {a["neg"] for a in ARMS.values()})
    if len(np.unique(smeta[:, ROW_C])) != len(smeta):
        raise ProtocolStop("C7", "duplicate cache rows in the score file")
    if len(np.unique(rmeta[:, ROW_C])) != len(rmeta):
        raise ProtocolStop("C7", "duplicate cache rows in the risk file")
    # the target set is rebuilt from the CACHE labels with the same rule stage 1
    # uses, not read off the risk file - deriving the expectation from one of the
    # files being checked would make the check partly self-fulfilling
    ycache = np.load(f"{cache}/y_clutterremoved.npy")
    scache = np.load(f"{cache}/subj_clutterremoved.npy")
    want = np.sort(np.concatenate([cell_rows(ycache, scache, g, su)
                                   for g in gest_needed for su in range(1, 9)]))
    if not np.array_equal(np.sort(smeta[:, ROW_C]), want):
        raise ProtocolStop("C7", f"score file is not the complete target set: "
                           f"{len(smeta)} rows vs {len(want)} rebuilt from the cache "
                           f"for gestures {gest_needed}")
    if not np.array_equal(np.sort(rmeta[np.isin(rmeta[:, GEST_C], gest_needed)][:, ROW_C]),
                          want):
        raise ProtocolStop("C7", "the risk file's own target rows disagree with the "
                           "cache-rebuilt set")
    for key in [f"{a}_{c}" for a in ("H", "V", "C1") for c in CHANNELS]:
        if key not in sc.files:
            raise ProtocolStop("C7", f"score file is missing channel {key}")
        if len(sc[key]) != len(smeta) or not np.all(np.isfinite(sc[key])):
            raise ProtocolStop("C7", f"{key}: length mismatch or non-finite values")
    for key in ("risk_fit", "risk_raw"):
        if rk[key].shape != (len(rmeta), 3) or not np.all(np.isfinite(rk[key])):
            raise ProtocolStop("C7", f"{key}: shape mismatch or non-finite values")

    pos = {int(r): i for i, r in enumerate(rmeta[:, ROW_C])}
    take = np.array([pos[int(r)] for r in smeta[:, ROW_C]])
    if not np.array_equal(rmeta[take][:, [GEST_C, SUBJ_C, ORD_C]],
                          smeta[:, [GEST_C, SUBJ_C, ORD_C]]):
        raise ProtocolStop("C7", "stage 1 meta disagrees with the risk npz")
    risk_fit = rk["risk_fit"][take]
    print(f"  {len(smeta)} scored samples = the complete row set for gestures "
          f"{gest_needed}; rows unique; gesture/subject/file-order all agree")

    # C0 lives in the X_raw indexing; validate the join row by row before using it
    yr = np.load(f"{cache}/y_raw.npy")
    sr = np.load(f"{cache}/subj_raw.npy")
    if not (np.array_equal(c0["y"], yr[c0["idx"]])
            and np.array_equal(c0["subj"], sr[c0["idx"]])):
        raise ProtocolStop("C7", "C0 npz labels disagree with X_raw at its own rows")
    rowpos = {int(r): i for i, r in enumerate(rmeta[:, ROW_C])}
    if len(np.unique(c0["idx"])) != len(c0["idx"]):
        raise ProtocolStop("C7", "duplicate rows in the C0 file")
    c0_want = np.sort(np.concatenate(
        [cell_rows(ycache, scache, g, su)
         for g in (ARMS["C0"]["pos"], ARMS["C0"]["neg"]) for su in range(1, 9)]))
    if not np.array_equal(np.sort(c0["idx"]), c0_want):
        raise ProtocolStop("C7", f"C0 file is not the complete G9/G10 set: "
                           f"{len(c0['idx'])} rows vs {len(c0_want)} expected")
    if len(c0["area"]) != len(c0["idx"]) or not np.all(np.isfinite(c0["area"])):
        raise ProtocolStop("C7", "C0 areas: length mismatch or non-finite values")
    c0_sel = np.array([rowpos[int(i)] for i in c0["idx"]])
    if not (np.array_equal(rmeta[c0_sel][:, GEST_C], c0["y"])
            and np.array_equal(rmeta[c0_sel][:, SUBJ_C], c0["subj"])):
        raise ProtocolStop("C7", "C0 rows do not line up with the risk metadata")
    print(f"  C0: {len(c0_sel)} rows = the complete G9/G10 set, validated against "
          "X_raw AND the risk metadata")

    def run(arm, chan, s_all, risk_mat, how, is_pos, sub, j_all, agg="mean"):
        return fold_analysis(s_all, paired_risk(risk_mat, how, agg), j_all,
                             is_pos, sub, arm, chan)

    results, primary = {}, {}
    for arm, spec in ARMS.items():
        if arm == "C0":
            results[("C0", "C0")] = run("C0", "C0", c0["area"], rk["risk_fit"][c0_sel],
                                        spec["risk"], c0["y"] == spec["pos"],
                                        c0["subj"], rmeta[c0_sel][:, ORD_C])
            continue
        m = np.isin(smeta[:, GEST_C], [spec["pos"], spec["neg"]])
        ip, sb, jj = smeta[m][:, GEST_C] == spec["pos"], smeta[m][:, SUBJ_C], smeta[m][:, ORD_C]
        for chan in CHANNELS:
            results[(arm, chan)] = run(arm, chan, sc[f"{arm}_{chan}"][m],
                                       risk_fit[m], spec["risk"], ip, sb, jj)

    print("\n" + "=" * 78)
    print("A  per-arm results.  D and R are 17 3.1/3.2; Rfit is the declared")
    print("   secondary analysis of 3.2 and is NOT the falsification control.")
    print("=" * 78)
    for k, v in results.items():
        primary[k] = report_arm(k[0], k[1], v)

    # ------------------------------------------------------------------ verdict
    print("\n" + "=" * 78)
    print("B  VERDICT (17 6.1, 6.2, 6.3, 6.4 - frozen before these numbers existed)")
    print("=" * 78)
    l1 = _report_layer("L1", "C0", primary[("C0", "C0")])
    print("     L1 has NO falsification control by construction: C0's range companion")
    print("     is not a matched functional (17 3.4b), so a PASS means the recorded")
    print("     failure was foreseeable, NOT that wrapping caused it.")
    pair_c1 = {n: paired_delta(results[("C1", "D")], results[("C1", n)])[:2]
               for n in ("Dunw", "R")}
    l2 = _report_layer("L2", "C1 D vs Dunw and R", primary[("C1", "D")],
                       controls={"Dunw": primary[("C1", "Dunw")],
                                 "R": primary[("C1", "R")]},
                       paired=pair_c1)
    verdict, why_txt = l2_diagnosis(primary[("C1", "D")], primary[("C1", "Dunw")],
                                    primary[("C1", "R")], paired=pair_c1)
    print(f"\n     `18` 5 three-way diagnosis: {verdict}")
    print(f"     {why_txt}")

    print(f"\n  6.4 decision matrix -> L1 {'PASS' if l1 else 'FAIL'}, "
          f"L2 {'PASS' if l2 else 'FAIL'}")
    print(f"     {DECISION_MATRIX[(l1, l2)]}")

    print("\n  6.3 pre-declared extension arms (they can neither rescue nor overturn")
    print("  C0/C1; reported with an explicit verdict so they are not left dangling):")
    hv = {}
    for arm in ("H", "V"):
        hv[arm] = _report_layer(
            f"  {arm}", f"{arm} D vs Dunw and R", primary[(arm, "D")],
            controls={"Dunw": primary[(arm, "Dunw")], "R": primary[(arm, "R")]},
            paired={n: paired_delta(results[(arm, "D")], results[(arm, n)])[:2]
                    for n in ("Dunw", "R")})
    if not (l1 or l2) and any(hv.values()):
        print("\n     17 6.3: risk applies to directional readouts, but the original")
        print("     chirality attribution of `07` 2 is unsupported.")
    if (l1 or l2) and not any(hv.values()):
        print("\n     17 6.3: the framework's scope is limited to chirality-type readouts.")

    # -------------------------------------------------------------- sensitivity
    print("\n" + "=" * 78)
    print("C  DECLARED SENSITIVITY ANALYSES (17 2, 3.2) - reported, never criteria")
    print("=" * 78)
    raw_all = rk["risk_raw"][take]
    print("  every arm that carries or controls a layer is re-run under each declared")
    print("  variant.  All of it is descriptive; the primary row is FIT/mean.\n")
    for arm, chan in (("C0", "C0"), ("C1", "D"), ("C1", "R"), ("H", "D"),
                      ("H", "R"), ("V", "D"), ("V", "R")):
        spec = ARMS[arm]
        if arm == "C0":
            s_all, fitm, rawm = c0["area"], rk["risk_fit"][c0_sel], rk["risk_raw"][c0_sel]
            ip, sb, jj = c0["y"] == spec["pos"], c0["subj"], rmeta[c0_sel][:, ORD_C]
        else:
            m = np.isin(smeta[:, GEST_C], [spec["pos"], spec["neg"]])
            s_all, fitm, rawm = sc[f"{arm}_{chan}"][m], risk_fit[m], raw_all[m]
            ip, sb, jj = (smeta[m][:, GEST_C] == spec["pos"],
                          smeta[m][:, SUBJ_C], smeta[m][:, ORD_C])
        variants = [("FIT mean (primary)", fitm, spec["risk"], "mean"),
                    ("FIT max over views", fitm, spec["risk"], "max"),
                    ("RAW Viterbi mean", rawm, spec["risk"], "mean")]
        if spec["risk"] == "LTR":            # 17 2: the C arm gets a per-view split
            variants += [(f"FIT {v} only", fitm, v, "mean") for v in ("L", "T", "R")]
        print(f"  {arm}/{chan}")
        for label, mat, how, agg in variants:
            note = ""
            if agg == "max" and how == "T":
                note = "   (identical to mean: R^V is a single view)"
            try:
                st = arm_stats(run(arm, chan, s_all, mat, how, ip, sb, jj, agg))
            except ProtocolStop as e:
                print(f"    {label:22s} stopped: {e}")
                continue
            print(f"    {label:22s} beta mean {st['beta_mean']:+.4f}  p = {st['p_beta']:.4f}"
                  f"   rho mean {st['rho_mean']:+.4f}  p = {st['p_rho']:.4f}"
                  f"  k={st['k']}{note}")
    print("\n  none of the above changes a verdict; 17 2 and 3.2 declared them in")
    print("  advance so that they cannot be selected after the primary row is known.")

    print("\n  C0 and C1 share the same 1600 G9/G10 samples, so they are NOT two")
    print("  independent confirmations (17 4).")


# ===========================================================================
def build_parser():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--selftest", action="store_true",
                      help="synthetic verification of every formula; no cache access")
    mode.add_argument("--track", action="store_true",
                      help="stage 1: extract per-sample scores from the cache")
    mode.add_argument("--analyze", action="store_true",
                      help="stage 2: the real L1/L2 verdict (17 6.4)")
    return ap


def main():
    args = build_parser().parse_args()

    if args.track:
        stage1_track()
    elif args.analyze:
        stage2_analyze()
    else:
        selftest()
        if not args.selftest:
            print("\n  (no mode given - ran --selftest.  --track and --analyze are")
            print("   explicit on purpose: code first, audit second, real numbers last.)")


if __name__ == "__main__":
    main()
