"""POST-HOC diagnostic for the Gate 0 level 6 -> 7 inversion.

    NOT part of the preregistration.  NOT a Gate 0 criterion.  Running it, or
    any number it prints, does NOT change the Gate 0 PASS recorded in
    `logs/w3_gate0_sim.log`.  It was written AFTER the gate was decided and
    exists only to explain one observation in it.

`proposal_v2/17` 6.0.4 fixed the verdict on all eight levels; that verdict
stands.  What this script explains is a descriptive fact underneath it: the FIT
model risk score falls at the highest level (80.34% -> 75.09%) while the true
wrap rate still rises (81.46% -> 89.89%), deterministically, in every one of the
ten renders.

The first written explanation - "K=30 cannot represent 8 strokes" - was WRONG
and is retained here only as the hypothesis this script falsifies.

Three manipulations, each isolating one candidate cause:

  1  pure approximation - least-squares fit the K=30 basis straight onto the TRUE
     path.  No renderer, no tracker, no solver.  If the basis were the limit,
     this alone would fail.  It does not.
  2  tracking quality   - RMSE of the Viterbi range track against the true ranges,
     reported as the MEAN over the ten fixed renders of that level (a mean, not
     the two-stage aggregation, because it is a per-render measurement error and
     never enters any 17 quantity).
  3  basin choice       - re-solve the SAME tracks with K/lam/mu unchanged and
     only the initialisation swapped, from the default flat z = Z_REF to the true
     path.  Because the measurements are held fixed, whatever changes is the
     solver's convergence basin and not a measurement effect.

R^C here uses the FORMAL two-stage aggregation of 17 6.0.3(b) - median within
each chirality, then an equal-weight CW/CCW mean - so that the default-init
column is directly comparable to the gate.  It reproduces the gate's R^C exactly,
which the script asserts on every run.  An earlier draft averaged the ten renders
directly; those numbers (80.26 / 81.91 / 75.13 / 88.80) are NOT R^C in the
frozen sense and must not be quoted as such.

Levels, seeds, trajectory and aggregation are all imported from w3_gate0_sim so
the two files cannot drift apart.

Zero GPU, 20 renders.  Usage: python tools/w3_gate0_diag.py
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from uwb_osr.trajectory import viterbi_track, fit_trajectory                    # noqa: E402
from uwb_osr.forward import render, clutter_filter                              # noqa: E402
from w3_gate0_sim import (circle, wrap_fraction, rho_of, two_stage,             # noqa: E402
                          seed_and_kind, render_setup, reconstruct,
                          AMP_M, N_SEED, STROKES, K_CTRL, LAM, MU)

# the two levels that bracket the inversion
LEVELS = (6, 7)
# the gate's own recorded R^C, read from its artefact rather than transcribed:
# a hand-copied constant only carries the printed 2 dp and would false-alarm here
GATE_NPZ = "logs/w3_gate0_sim.npz"


def approximation_only(strokes, B):
    """Fit the basis straight onto the true path: no echoes, no tracking."""
    out = []
    for kind in ("cw", "ccw"):
        p = circle(kind, amp=AMP_M, strokes=strokes)
        C, *_ = np.linalg.lstsq(B, p, rcond=None)
        pf = B @ C
        out.append((kind,
                    np.sqrt(np.mean(np.sum((pf - p) ** 2, axis=1))) * 1000,
                    100 * wrap_fraction(rho_of(pf)).mean(),
                    100 * wrap_fraction(rho_of(p)).mean()))
    return out


def track_and_refit(level_idx, strokes, tmpl, g_render, noise, B):
    """Render the level's ten fixed samples; measure tracking, then refit twice."""
    trk, by_kind_def, by_kind_init = [], {"cw": [], "ccw": []}, {"cw": [], "ccw": []}
    for si in range(N_SEED):
        kind, rng = seed_and_kind(level_idx, si)
        p = circle(kind, amp=AMP_M, strokes=strokes)
        cr = clutter_filter(render(p, template=tmpl, gains=g_render, noise=noise,
                                   scatterers=None, rng=rng))
        tr = [viterbi_track(cr[v]) for v in range(3)]
        r = np.stack([t[0] for t in tr])
        c = np.stack([t[1] for t in tr])
        trk.append(np.sqrt(np.mean((r - rho_of(p)) ** 2)) * 1000)

        # default initialisation - identical to what the gate itself ran
        by_kind_def[kind].append(wrap_fraction(rho_of(reconstruct(cr, B)["p"])).mean())
        # same measurements, same K/lam/mu, only the starting point changes
        C0, *_ = np.linalg.lstsq(B, p, rcond=None)
        seeded = fit_trajectory(r, c, (0, 1, 2), B=B, lam=LAM, mu=MU, init=C0)
        by_kind_init[kind].append(wrap_fraction(rho_of(seeded["p"])).mean())

    return (np.mean(trk),
            {k: np.array(v) for k, v in by_kind_def.items()},
            {k: np.array(v) for k, v in by_kind_init.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-npz", default=GATE_NPZ,
                    help="artefact to cross-check against; exists so the failure "
                         "branch can actually be exercised")
    gate_npz = ap.parse_args().gate_npz

    print("=" * 78)
    print("POST-HOC DIAGNOSTIC - Gate 0 level 6 -> 7 inversion")
    print("=" * 78)
    print("  NOT preregistered, NOT a criterion, does NOT change the Gate 0 PASS.")
    print(f"  K = {K_CTRL}, lam = {LAM}, mu = {MU}; levels, seeds and aggregation")
    print("  imported from tools/w3_gate0_sim.py\n")

    tmpl, g_render, noise, B = render_setup()

    print("=" * 78)
    print("1  PURE APPROXIMATION: is the K=30 basis the limit?")
    print("=" * 78)
    print("  least-squares fit of the basis onto the TRUE path - no renderer, no")
    print("  tracker, no solver.  If the basis could not hold 8 strokes, it fails here.\n")
    print(f"  {'strokes':>8s}{'kind':>6s}{'pos RMS':>10s}{'wrap(fit)':>12s}{'wrap(true)':>12s}")
    for li in LEVELS:
        for kind, rms, wf, wt in approximation_only(STROKES[li], B):
            print(f"  {STROKES[li]:8.2f}{kind:>6s}{rms:9.2f}mm{wf:11.2f}%{wt:11.2f}%")
    print("\n  => the basis represents both levels.  The first written explanation")
    print("     ('K=30 cannot represent 8 strokes') is FALSIFIED.")

    print("\n" + "=" * 78)
    print("2/3  TRACKING QUALITY, and R^C under a true-path initialisation")
    print("=" * 78)
    print("  Viterbi RMSE = MEAN over the 10 fixed renders of the level (a per-render")
    print("  measurement error; it is not a 17 quantity and gets no two-stage rule).")
    print("  R^C = the FORMAL two-stage aggregation of 17 6.0.3(b): median within")
    print("  each chirality, then equal-weight CW/CCW mean.\n")
    print(f"  {'lvl':>3s}{'strokes':>9s}{'Viterbi RMSE':>14s}"
          f"{'R^C default':>13s}{'R^C true-init':>15s}{'recovered':>11s}")
    rows = {}
    for li in LEVELS:
        trk, d_def, d_init = track_and_refit(li, STROKES[li], tmpl, g_render, noise, B)
        rc_def, rc_init = two_stage(d_def), two_stage(d_init)
        rows[li] = (trk, rc_def, rc_init)
        print(f"  {li:3d}{STROKES[li]:9.2f}{trk:11.2f} mm{100 * rc_def:12.2f}%"
              f"{100 * rc_init:14.2f}%{100 * (rc_init - rc_def):+10.2f} pp")

    print(f"\n  cross-check against the gate's recorded R^C ({gate_npz}):")
    gate_rc = np.load(gate_npz)["rc_level"]
    ok = True
    for li in LEVELS:
        d = abs(rows[li][1] - gate_rc[li])
        ok &= d == 0.0
        print(f"    level {li}: R^C default = {100 * rows[li][1]:.6f}%  vs gate "
              f"{100 * gate_rc[li]:.6f}%   [{'BITWISE MATCH' if d == 0.0 else 'MISMATCH'}]")
    if not ok:
        # a hard failure, not a quiet return: this check is the script's claim to
        # be driving the gate's own renders, so it must be able to fail the process
        raise SystemExit("  the diagnostic is not reproducing the gate's own "
                         "renders - stop.")

    print("\n" + "=" * 78)
    print("  READING")
    print("=" * 78)
    print("  The basis is not the bottleneck (part 1).  Two things move together at")
    print("  the top level: the Viterbi range track degrades "
          f"({rows[6][0]:.2f} -> {rows[7][0]:.2f} mm),")
    print("  and on those same degraded tracks the fit becomes initialisation-")
    print("  sensitive - holding the measurements fixed and changing only the start")
    print(f"  point recovers {100 * (rows[7][2] - rows[7][1]):.2f} pp at level 7 "
          f"against {100 * (rows[6][2] - rows[6][1]):.2f} pp at level 6.")
    print("  Neither alone accounts for the inversion; it is not attributed to a")
    print("  single cause, and in particular not to the spline's expressiveness.")
    print("\n  Gate 0 itself is unaffected: all 8 levels PASS on the frozen criteria;")
    print("  the inversion means per-level strict monotonicity does NOT hold; the rho")
    print("  of levels 0-6 is post-hoc description and does not redefine Gate 0.")


if __name__ == "__main__":
    main()
