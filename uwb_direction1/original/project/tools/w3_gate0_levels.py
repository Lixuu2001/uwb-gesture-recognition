"""Analytic coverage check for the Gate 0 speed levels (proposal_v2/17 6.0.2).

Pure geometry: the true wrap rate is computed from the KNOWN trajectory, so this
touches neither the renderer, nor the FIT risk score, nor any Doppler readout.
It is therefore safe to run before the preregistration is sealed.

The acceptance criteria printed here are DESIGN-SELF-CONTAINED on purpose -
span, distinctness, gap uniformity.  An earlier version also scored the levels
against a "real measured band 25.8-57.3%".  That argument has been withdrawn
(17 6.0.2) for three independent reasons: the band came from a 1920-sample
subset (full-sample value is 28.0-58.1%), it was aggregated max-over-views while
Gate 0 uses the three-view mean R^C, and it is a FIT MODEL RISK SCORE while this
axis is the TRUE wrap rate - pairing the two presupposes the calibration that
Gate 0 explicitly refuses to claim.

The z-column exists because `strokes` is not a pure speed intervention: the
z breathing term of w2_extended.circle is a fixed 3 cm / 1 cycle oscillation, so
its 2.12 mm per-frame step decides the levels that sit right at the 10.27 mm
threshold.  Re-running with the term zeroed shows which levels are speed-driven.

Usage: python tools/w3_gate0_levels.py
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from uwb_osr.trajectory import RADAR_POS, Z_REF                     # noqa: E402

LAMBDA_M = 41.1e-3
QUARTER = LAMBDA_M / 4.0
T = 90
Z_AMP = 0.03                        # w2_extended.circle breathing term, verbatim

# candidate designs
DESIGNS = {
    "A  amp+strokes co-varying (original 17 6.0.2 proposal, REJECTED)": [
        (0.05, 1.0), (0.10, 1.0), (0.15, 1.5), (0.20, 2.0),
        (0.25, 2.0), (0.30, 2.5), (0.35, 3.0), (0.40, 3.0)],
    "B  amp fixed 0.15 m, strokes only (SUPERSEDED by B', level 1 was z-driven)": [
        (0.15, s) for s in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 4.5, 8.0)],
    "B' amp fixed, level 1 at 1.30 (ADOPTED, 17 6.0.2)": [
        (0.15, s) for s in (1.0, 1.30, 1.5, 1.75, 2.0, 2.5, 4.5, 8.0)],
}


def circle(kind, T=T, amp=0.25, strokes=2.0, z_amp=Z_AMP):
    """Verbatim from tools/w2_extended.py, plus a z_amp knob for diagnosis only.

    z_amp defaults to the frozen 0.03 m; Gate 0 itself always uses the default.
    """
    t = np.linspace(0, 1, T)
    sgn = -1.0 if kind == "cw" else 1.0
    return np.stack([amp * np.cos(2 * np.pi * strokes * t),
                     sgn * amp * np.sin(2 * np.pi * strokes * t),
                     Z_REF + z_amp * np.sin(2 * np.pi * t)], axis=1)


def true_wrap(p):
    """Per-view wrap rate of a KNOWN trajectory p (T,3) -> (3,)."""
    rho = np.linalg.norm(p[None, :, :] - RADAR_POS[:, None, :], axis=2)   # (3,T)
    return (np.abs(np.diff(rho, axis=1)) > QUARTER).mean(axis=1)          # (3,)


def level_wrap(amp, strokes, z_amp=Z_AMP):
    """The frozen two-stage aggregation of 17 6.0.3(a): 3-view mean, then CW/CCW."""
    w = {k: true_wrap(circle(k, amp=amp, strokes=strokes, z_amp=z_amp))
         for k in ("cw", "ccw")}
    per_view = 0.5 * (w["cw"] + w["ccw"]) * 100
    return 100 * w["cw"].mean(), 100 * w["ccw"].mean(), per_view


def report(name, levels):
    print("=" * 86)
    print(name)
    print("=" * 86)
    print(f"  {'lvl':>3s}{'amp':>7s}{'strk':>6s}{'CW %':>8s}{'CCW %':>8s}"
          f"{'mean %':>9s}{'z=0 %':>8s}{'L':>7s}{'T':>7s}{'R':>7s}")
    means = []
    for i, (amp, st) in enumerate(levels):
        mcw, mccw, pv = level_wrap(amp, st)
        m = 0.5 * (mcw + mccw)
        z0 = 0.5 * sum(level_wrap(amp, st, z_amp=0.0)[:2])
        means.append(m)
        print(f"  {i:3d}{amp:7.2f}{st:6.2f}{mcw:8.2f}{mccw:8.2f}{m:9.2f}{z0:8.2f}"
              f"{pv[0]:7.1f}{pv[1]:7.1f}{pv[2]:7.1f}")

    means = np.array(means)
    uniq = len(np.unique(np.round(means, 6)))
    print(f"\n  span            : {means.min():.2f}% .. {means.max():.2f}%")
    print(f"  distinct levels : {uniq}/8   (ties merge -> {uniq} effective points)")
    print(f"  >= 6 effective points   : {'YES' if uniq >= 6 else 'NO'}")
    print(f"  max gap between levels  : {np.diff(np.sort(means)).max():.2f} pp\n")
    return means


def z_sensitivity():
    """Which levels are speed-driven and which are depth-driven?"""
    print("=" * 86)
    print("z-sensitivity of the fixed-radius sweep (amp = 0.15 m)")
    print("=" * 86)
    print("  the z breathing term is a FIXED 3 cm / 1 cycle oscillation: its largest")
    print("  per-frame step is 2.12 mm against the 10.27 mm threshold, so it only")
    print("  decides levels sitting right at the knife edge.\n")
    print(f"  {'strokes':>8s}{'with z %':>10s}{'z = 0 %':>10s}{'delta pp':>10s}   driver")
    for st in (1.00, 1.10, 1.20, 1.25, 1.30, 1.35, 1.50, 1.75, 2.00, 2.50, 4.50, 8.00):
        a = 0.5 * sum(level_wrap(0.15, st)[:2])
        b = 0.5 * sum(level_wrap(0.15, st, z_amp=0.0)[:2])
        if a < 1e-9:
            drv = "-  (no wrap at all)"
        elif b < 1e-9:
            drv = "DEPTH  (all of it is the z term)"
        else:
            drv = "in-plane speed"
        print(f"  {st:8.2f}{a:10.2f}{b:10.2f}{a - b:+10.2f}   {drv}")
    print()


if __name__ == "__main__":
    print(f"lambda/4 = {QUARTER * 1e3:.2f} mm   T = {T}   radars = nominal   "
          f"z_amp = {Z_AMP * 1e2:.0f} cm\n")
    for nm, lv in DESIGNS.items():
        report(nm, lv)
    z_sensitivity()
