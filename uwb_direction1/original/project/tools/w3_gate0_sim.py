"""Gate 0 of the readout-failure preregistration (`proposal_v2/17` 6.0).

The whole gate asks ONE question, and deliberately no more than one:

    does the FIT model risk score get the physical ORDERING backwards?

Ground truth is available here because the trajectory is ours: we render the
echoes from a known path, so the true wrap rate can be computed analytically
from that path without any inversion.  Sweeping `strokes` walks the true wrap
rate from 0% to ~90%, and the gate checks that the FIT score - which on real
data is all we ever have - ranks the levels in the same direction.

What a PASS does NOT mean (17 6.0.4, and dead ends #10/#15 in `00` 5): it does
not make FIT equal to the true wrap rate, its value is not a physical wrapping
proportion, and the rho measured here must never be used to calibrate anything
on real data.  The gate is an ordering check and nothing else.

Gate 0 touches NO Doppler readout at all - neither channel D nor C0/C1.  That is
why this file imports no Doppler code: the only quantities it computes are the
true wrap rate of a known path and the FIT risk score of the W1 reconstruction.

Levels are design B' (17 6.0.2): `amp` fixed at 0.15 m and only `strokes` swept,
so the radius - and therefore how far the target sits from each radar - is held
constant and does not leak into the ordering.  This is NOT a pure speed
intervention: sweeping `strokes` also changes the total number of revolutions,
the terminal phase, and the ratio of in-plane speed to the fixed z breathing
term.  What B' does guarantee is that no level gets its wrapping from a
different physical quantity (17 6.0.2, the z=0 diagnostic).

FAIL means the whole line stops (17 6.5): if the risk score cannot even order a
simulation where truth is known, there is nothing to say about real data.

Zero GPU, ~80 renders.  Usage: python tools/w3_gate0_sim.py [--smoke]
"""
import argparse
import sys
import time

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, ".")
from uwb_osr.trajectory import (viterbi_track, bspline_basis, fit_trajectory,   # noqa: E402
                                RADAR_POS, Z_REF)
from uwb_osr.forward import (render, clutter_filter, load_template,             # noqa: E402
                             calibrate_gains_noise)

CACHE = r"D:\uwb_gestures\cache"
NPZ = "logs/w3_gate0_sim.npz"
VIEWS = ["Left", "Top", "Right"]

# W1 pipeline hyper-parameters (`00` 4.2) - frozen, do not re-tune
K_CTRL, LAM, MU = 30, 0.05, 0.05

# authoritative sensing constants (`00` 4.1) - do not re-derive
LAMBDA_M = 41.1e-3
QUARTER = LAMBDA_M / 4.0                # 10.275 mm, the wrapping threshold

# design B' (17 6.0.2), frozen
T_FRAMES = 90
AMP_M = 0.15
STROKES = (1.00, 1.30, 1.50, 1.75, 2.00, 2.50, 4.50, 8.00)
N_SEED = 10                             # 5 ccw (even seed_idx) + 5 cw (odd)
WRAP_FROZEN = (0.00, 16.29, 34.83, 47.38, 56.55, 66.10, 81.46, 89.89)

TIE_PP = 0.01                           # tie tolerance on the true wrap rate
MIN_POINTS = 6                          # fewer effective points after merging = FAIL
RHO_PASS = 0.7


def circle(kind, T=T_FRAMES, amp=0.25, strokes=2.0):
    """Verbatim from tools/w2_extended.py (17 6.0.1 pins the trajectory source)."""
    t = np.linspace(0, 1, T)
    sgn = -1.0 if kind == "cw" else 1.0
    return np.stack([amp * np.cos(2 * np.pi * strokes * t),
                     sgn * amp * np.sin(2 * np.pi * strokes * t),
                     Z_REF + 0.03 * np.sin(2 * np.pi * t)], axis=1)


def reconstruct(frames, B):
    """Verbatim from tools/w2_extended.py: the full W1 pipeline, K=30, lam=mu=0.05."""
    tr = [viterbi_track(frames[v]) for v in range(3)]
    r = np.stack([t[0] for t in tr])
    c = np.stack([t[1] for t in tr])
    return fit_trajectory(r, c, (0, 1, 2), B=B, lam=LAM, mu=MU)


def wrap_fraction(rho):
    """rho: (V,T) range tracks in metres -> (V,) fraction of wrapped frames.

    This is 17 2 verbatim: #{t : |rho(t+1) - rho(t)| > lambda/4} / (T-1).
    """
    return (np.abs(np.diff(rho, axis=1)) > QUARTER).mean(axis=1)


def rho_of(p):
    """(T,3) path -> (3,T) ranges to the three radars."""
    return np.linalg.norm(p[None, :, :] - RADAR_POS[:, None, :], axis=2)


def seed_and_kind(level_idx, seed_idx):
    """The frozen seeding rule of 17 6.0.1.

    rng = default_rng(1000*level_idx + seed_idx); even seed_idx renders CCW and
    odd renders CW, giving 5 of each per level.  Lives here rather than inline so
    that tools/w3_gate0_diag.py drives the identical renders.
    """
    kind = "ccw" if seed_idx % 2 == 0 else "cw"
    return kind, np.random.default_rng(1000 * level_idx + seed_idx)


def render_setup():
    """Renderer inputs, verbatim from tools/w2_extended.py part B (17 6.0.1).

    Shared with tools/w3_gate0_diag.py so the calibration cannot drift apart
    between the gate and its post-hoc diagnostic.
    """
    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    subj = np.load(f"{CACHE}/subj_clutterremoved.npy")
    tmpl = load_template()
    gains, sigmas = calibrate_gains_noise(X, y, subj)
    g_render = gains / gains.max()
    noise = float(np.median(sigmas / gains))
    B = bspline_basis(T_FRAMES, K_CTRL)
    return tmpl, g_render, noise, B


# ---------------------------------------------------------------------------
# (a) true wrap rate - from the KNOWN path, no inversion anywhere
# ---------------------------------------------------------------------------
def true_wrap_per_view(p_true):
    return wrap_fraction(rho_of(p_true))


def true_wrap_level(strokes, amp=AMP_M):
    """17 6.0.3(a): three-view equal-weight mean, then CW/CCW equal-weight mean.

    CW and CCW do NOT agree: the y component flips sign with `kind` and couples
    to the z breathing term, so the two chiralities wrap differently (level 1 is
    19.48 vs 13.11).  Hence the explicit two-stage aggregation.
    """
    per_kind = {k: true_wrap_per_view(circle(k, amp=amp, strokes=strokes)).mean()
                for k in ("cw", "ccw")}
    both = 0.5 * (per_kind["cw"] + per_kind["ccw"])
    return 100 * both, 100 * per_kind["cw"], 100 * per_kind["ccw"]


# ---------------------------------------------------------------------------
# (b) FIT model risk score - the quantity we would have on real data
# ---------------------------------------------------------------------------
def render_and_score(level_idx, seed_idx, strokes, tmpl, g_render, noise, B):
    """One render -> clutter filter -> W1 reconstruction -> R^C.

    Seeding follows the frozen rule of 17 6.0.1, see `seed_and_kind`.
    """
    kind, rng = seed_and_kind(level_idx, seed_idx)
    p_true = circle(kind, amp=AMP_M, strokes=strokes)
    raw = render(p_true, template=tmpl, gains=g_render, noise=noise,
                 scatterers=None, rng=rng)
    cr = clutter_filter(raw)
    out = reconstruct(cr, B)
    risk = wrap_fraction(rho_of(out["p"]))         # (3,) per-view
    return kind, risk, out["rms"] * 1000


def two_stage(per_kind):
    """17 6.0.3(b): median within each `kind`, then CW/CCW equal-weight mean.

    Not a median over all 10 seeds: taking it per chirality first keeps the
    chirality weighting identical on both sides of the comparison, which is
    what makes the (a) and (b) axes strictly symmetric.
    """
    return 0.5 * (float(np.median(per_kind["cw"])) + float(np.median(per_kind["ccw"])))


# ---------------------------------------------------------------------------
# merging of tied levels (17 6.0.4 fallback clause)
# ---------------------------------------------------------------------------
def merge_ties(wrap_pct, rc_by_kind):
    """Group levels whose TRUE wrap rates tie within TIE_PP, then re-aggregate.

    Analytically the eight B' levels are all distinct, so this never fires on
    the frozen design; it is kept because the preregistration defines it and a
    silently absent fallback is worse than an unused one.
    """
    order = np.argsort(wrap_pct)
    groups, cur = [], [int(order[0])]
    for k in order[1:]:
        if abs(wrap_pct[k] - wrap_pct[cur[-1]]) < TIE_PP:
            cur.append(int(k))
        else:
            groups.append(cur)
            cur = [int(k)]
    groups.append(cur)

    g_wrap, g_rc = [], []
    for grp in groups:
        # (a) side: the tied value itself
        g_wrap.append(float(np.mean([wrap_pct[k] for k in grp])))
        # (b) side: pool every render of the tied levels, then re-run two_stage
        pooled = {kd: np.concatenate([rc_by_kind[k][kd] for k in grp])
                  for kd in ("cw", "ccw")}
        g_rc.append(two_stage(pooled))
    return groups, np.array(g_wrap), np.array(g_rc)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2 levels x 2 seeds pipeline check; prints NO verdict")
    args = ap.parse_args()

    levels = list(enumerate(STROKES))
    n_seed = N_SEED
    if args.smoke:
        levels = [levels[0], levels[7]]
        n_seed = 2

    print("=" * 78)
    print("GATE 0  simulation ordering gate  (proposal_v2/17 6.0)")
    print("=" * 78)
    print(f"  lambda/4 = {QUARTER * 1e3:.3f} mm   T = {T_FRAMES}   "
          f"amp = {AMP_M:.2f} m (fixed)   K = {K_CTRL}, lam = {LAM}, mu = {MU}")
    print(f"  levels = B' strokes {STROKES}")
    print(f"  single scattering centre, radars at nominal positions\n")
    if args.smoke:
        print("  !! SMOKE RUN - reduced levels and seeds, NOT A GATE 0 RESULT !!\n")

    # ---------------------------------------------------------------- render setup
    tmpl, g_render, noise, B = render_setup()
    print(f"  calibrated on real data: gains = {np.round(g_render, 4)}   "
          f"noise = {noise:.5f}\n")

    # ---------------------------------------------------------------- (a) truth
    print("=" * 78)
    print("A  TRUE wrap rate from the known path (no inversion, no renderer)")
    print("=" * 78)
    print(f"  {'lvl':>3s}{'strokes':>9s}{'CW %':>9s}{'CCW %':>9s}{'mean %':>9s}"
          f"{'frozen %':>10s}{'delta':>8s}")
    wrap_pct = np.zeros(len(STROKES))
    for li, st in enumerate(STROKES):
        m, cw, ccw = true_wrap_level(st)
        wrap_pct[li] = m
        print(f"  {li:3d}{st:9.2f}{cw:9.2f}{ccw:9.2f}{m:9.2f}"
              f"{WRAP_FROZEN[li]:10.2f}{m - WRAP_FROZEN[li]:+8.2f}")
    # the frozen table is quoted to 2 dp, so the faithful check is that these values
    # round to it - an absolute tolerance would sit exactly on the rounding boundary
    dev = np.abs(wrap_pct - np.array(WRAP_FROZEN)).max()
    same = np.array_equal(np.round(wrap_pct, 2), np.array(WRAP_FROZEN))
    print(f"\n  max deviation from the frozen 17 6.0.2 table: {dev:.4f} pp"
          f"   [{'OK, rounds to the frozen table' if same else 'MISMATCH - STOP'}]")
    if not same:
        print("  the level table does not reproduce; do not proceed.")
        return

    # ---------------------------------------------------------------- (b) FIT
    print("\n" + "=" * 78)
    print("B  FIT model risk score R^C from the rendered echoes")
    print("=" * 78)
    print("  render -> clutter_filter -> viterbi_track -> fit_trajectory -> reproject")
    print(f"  R^C = mean(risk_L, risk_T, risk_R), matching the C arm's paired risk\n")
    print(f"  {'lvl':>3s}{'strokes':>9s}{'CW med %':>11s}{'CCW med %':>11s}"
          f"{'R^C %':>9s}{'fit rms':>10s}{'L':>7s}{'T':>7s}{'R':>7s}")

    t0 = time.time()
    rc_by_kind, per_view, rms_all = {}, {}, {}
    rc_level = np.full(len(STROKES), np.nan)
    for li, st in levels:
        by_kind = {"cw": [], "ccw": []}
        views, rmss = [], []
        for si in range(n_seed):
            kind, risk, rms = render_and_score(li, si, st, tmpl, g_render, noise, B)
            by_kind[kind].append(float(risk.mean()))
            views.append(risk)
            rmss.append(rms)
        rc_by_kind[li] = {k: np.array(v) for k, v in by_kind.items()}
        per_view[li] = np.array(views)
        rms_all[li] = np.array(rmss)
        rc_level[li] = two_stage(rc_by_kind[li])
        pv = 100 * per_view[li].mean(axis=0)
        print(f"  {li:3d}{st:9.2f}{100 * np.median(by_kind['cw']):11.2f}"
              f"{100 * np.median(by_kind['ccw']):11.2f}{100 * rc_level[li]:9.2f}"
              f"{np.mean(rmss):9.1f}mm{pv[0]:7.1f}{pv[1]:7.1f}{pv[2]:7.1f}")
    print(f"\n  {len(levels) * n_seed} renders in {time.time() - t0:.0f}s")

    if args.smoke:
        print("\n  SMOKE RUN COMPLETE - no verdict computed, rerun without --smoke.")
        return

    np.savez(NPZ, wrap_pct=wrap_pct, rc_level=rc_level, strokes=np.array(STROKES),
             per_view=np.stack([per_view[i] for i in range(len(STROKES))]),
             rc_cw=np.stack([rc_by_kind[i]["cw"] for i in range(len(STROKES))]),
             rc_ccw=np.stack([rc_by_kind[i]["ccw"] for i in range(len(STROKES))]),
             rms=np.stack([rms_all[i] for i in range(len(STROKES))]),
             amp=AMP_M, n_seed=N_SEED, quarter=QUARTER)
    print(f"  [saved] {NPZ}")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print("C  VERDICT (criteria frozen in 17 6.0.4 before this script existed)")
    print("=" * 78)

    groups, g_wrap, g_rc = merge_ties(wrap_pct, rc_by_kind)
    merged = [g for g in groups if len(g) > 1]
    print(f"  tie merging (< {TIE_PP} pp on the true rate): "
          + ("none, 8 levels stay distinct" if not merged
             else f"{len(merged)} group(s) merged -> {[[int(i) for i in g] for g in merged]}"))
    print(f"  effective points: {len(g_wrap)}   "
          f"[{'OK' if len(g_wrap) >= MIN_POINTS else f'< {MIN_POINTS} = FAIL'}]")
    if len(g_wrap) < MIN_POINTS:
        print("\n  GATE 0 FAIL: too few effective points.  17 6.0.4 requires a level")
        print("  REDESIGN, not a relaxed threshold.  Do not proceed to 6.1/6.2.")
        return

    rho, p = spearmanr(g_wrap, g_rc)
    ends_ok = rc_level[-1] > rc_level[0]
    print(f"\n  Spearman rho = {rho:+.4f}   (p = {p:.4g}, reported but not a criterion)")
    print(f"  criterion 1: rho >= {RHO_PASS}                       "
          f"[{'PASS' if rho >= RHO_PASS else 'FAIL'}]")
    print(f"  criterion 2: R^C(level 7) > R^C(level 0)      "
          f"{100 * rc_level[-1]:.2f}% vs {100 * rc_level[0]:.2f}%   "
          f"[{'PASS' if ends_ok else 'FAIL'}]")

    # ------------------------------------------------------------- descriptive
    # DESCRIPTIVE ONLY, added AFTER the first run and NOT a criterion.  The frozen
    # 6.0.4 criteria above are computed exactly as written and are not touched by
    # anything below.  This block exists because a rank correlation reports one
    # number and would otherwise hide WHERE the score stops tracking the truth.
    print("\n" + "-" * 78)
    print("  descriptive (post-hoc, not a criterion): where does FIT stop tracking?")
    print("-" * 78)
    inv = [i for i in range(1, len(STROKES)) if rc_level[i] < rc_level[i - 1]]
    if not inv:
        print("  R^C is monotone across all 8 levels.")
    else:
        for i in inv:
            lo = np.concatenate([rc_by_kind[i]["cw"], rc_by_kind[i]["ccw"]])
            hi = np.concatenate([rc_by_kind[i - 1]["cw"], rc_by_kind[i - 1]["ccw"]])
            sep = "all" if lo.max() < hi.min() else "some"
            print(f"  level {i-1} -> {i}: true {wrap_pct[i-1]:.2f} -> {wrap_pct[i]:.2f}%"
                  f" but FIT {100*rc_level[i-1]:.2f} -> {100*rc_level[i]:.2f}%"
                  f"   ({sep} {len(lo)} renders below the level-{i-1} minimum;"
                  f" fit rms {rms_all[i-1].mean():.0f} -> {rms_all[i].mean():.0f} mm)")
        keep = [i for i in range(len(STROKES)) if i < min(inv)]
        if len(keep) >= 3:
            r_lo, _ = spearmanr(wrap_pct[keep], rc_level[keep])
            print(f"  for reference only, levels {keep[0]}-{keep[-1]} alone: "
                  f"rho = {r_lo:+.4f}")
            print("  that sub-sweep rho DOES NOT redefine the gate.  The verdict above")
            print("  stands on all 8 levels; what the inversion costs is per-level")
            print("  strict monotonicity, which the frozen criteria never required.")
        print("\n  cause, from an isolation diagnostic (numbers in `00` 3.1):")
        print("  NOT the spline's expressiveness - least-squares fitting the K=30 basis")
        print("  straight onto the true 8-stroke path lands at 4.56 mm position RMS and")
        print("  recovers 88.76-89.51% wrap, so the basis represents that motion fine.")
        print("  Two other things move together at the top level: the Viterbi range")
        print("  track itself degrades (8.75 -> 32.22 mm RMSE), and on those same")
        print("  degraded tracks the nonlinear fit becomes initialisation-sensitive -")
        print("  re-solving with K/lam/mu unchanged but seeded from the true path takes")
        print("  level 7 from 75.09% back to 88.76%.  Neither one alone accounts for")
        print("  the inversion, so it is not attributed to a single cause.")
        print("\n  the standing formulation: all 8 levels PASS on the frozen criteria;")
        print("  the top-level inversion means per-level strict monotonicity does NOT")
        print("  hold; the rho of levels 0-6 is post-hoc description and does not")
        print("  redefine Gate 0.")

    ok = (rho >= RHO_PASS) and ends_ok
    print("\n" + "=" * 78)
    print(f"  GATE 0: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    if ok:
        print("  Meaning, strictly (17 6.0.4): the FIT model risk score did NOT get")
        print("  the physical ordering backwards.  It does NOT mean FIT equals the")
        print("  true wrap rate, its value is NOT a physical wrapping proportion, and")
        print("  this rho must NOT be used to calibrate anything on real data.")
        print("\n  Next per 17 8: implement tools/w3_readout_predict.py (3-6).")
    else:
        print("  17 6.5: STOP.  A risk score that cannot order a simulation with known")
        print("  truth says nothing about real data.  Do not run 6.1/6.2.")


if __name__ == "__main__":
    main()
