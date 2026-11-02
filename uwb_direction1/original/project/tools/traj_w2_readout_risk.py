"""W3 power check: does the phase-wrapping RISK SCORE have usable dynamic range?

This runs BEFORE any preregistration of the readout-failure experiment
(`proposal_v2/16`).  Increment A of the third research point claims:

    risk(gesture, view) = fraction of frames whose inter-frame radial range
    change exceeds lambda/4 = 10.28 mm   ->   predicts Doppler readout failure

That claim can only be tested if the risk score actually VARIES across gestures.
The recorded number is a single global 32.2% (`00` 4.1).  If every gesture sits
at 30-34% the regressor has no variance and the experiment is dead on arrival,
no matter how rigorous the protocol.  Three gates are checked here:

  A  dynamic range : do at least two gestures have non-overlapping risk IQRs?
  B  confounding   : is between-gesture variance larger than between-subject
                     variance?  (if subjects dominate, the score measures WHO,
                     not WHAT)
  C  low-risk regime : does an unwrapped population exist at all?  Without it
                     "switch to the range domain when risk is high" has no
                     counterpart regime and collapses to the known answer
                     (dead end #10: always use the range domain).

Risk is computed twice, because it matters which trajectory it is read from:
  raw  - straight off the Viterbi range track
  fit  - reprojected from the W1 B-spline trajectory (K=30, lam=0.05)
Neither is a clean velocity measurement.  `raw` carries the Viterbi path prior
(jump_penalty=0.15) which CLIPS velocity, and single-bin tracking jitter is
6.35 mm against a 10.28 mm threshold, so two bins of jitter already count as a
wrap.  `fit` is smoothed but is the only one constrained to a single 3-D path
consistent with all three views.  Section 3 decides between them by testing
which one reproduces the analytic per-view sensitivity (07 P1): a horizontal
swipe must wrap on Left/Right and NOT on Top, because u_Top . e_x = 0 exactly.

Zero GPU.  Usage: python tools/traj_w2_readout_risk.py [--reps 20]
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from uwb_osr.trajectory import (viterbi_track, bspline_basis, fit_trajectory,   # noqa: E402
                                RADAR_POS, FPS)

CACHE = r"D:\uwb_gestures\cache"
NPZ = "logs/w3_readout_risk.npz"
VIEWS = ["Left", "Top", "Right"]
# meta column layout: gesture, subject, file-order index in cell, block, cache row
GEST_C, SUBJ_C, ORD_C, BLK, ROW_C = 0, 1, 2, 3, 4
K_CTRL, LAM, MU = 30, 0.05, 0.05

# authoritative sensing constants (`proposal_v2/00` 4.1) - do not re-derive
LAMBDA_M = 41.1e-3                      # carrier 7.56 GHz measured
QUARTER = LAMBDA_M / 4.0                # 10.275 mm, the wrapping threshold
V_UNAMB = LAMBDA_M * FPS / 4.0          # 0.2055 m/s

GEST = ["G1 LR", "G2 RL", "G3 UD", "G4 DU", "G5 LR-UD", "G6 LR-DU",
        "G7 RL-UD", "G8 RL-DU", "G9 CW", "G10 CCW", "G11 push", "G12 empty"]


def wrap_fraction(rho):
    """rho: (V,T) range tracks in metres -> (V,) fraction of wrapped frames."""
    return (np.abs(np.diff(rho, axis=1)) > QUARTER).mean(axis=1)


# ---------------------------------------------------------------------------
def collect(reps):
    X = np.load(f"{CACHE}/X_clutterremoved.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_clutterremoved.npy")
    subj = np.load(f"{CACHE}/subj_clutterremoved.npy")
    B = bspline_basis(90, K_CTRL)

    meta, risk_raw, risk_fit = [], [], []
    for g in range(12):
        for s in range(1, 9):
            idx = np.where((y == g) & (subj == s))[0][:reps]
            for j, i in enumerate(idx):
                unit = np.asarray(X[i])
                tr = [viterbi_track(unit[v]) for v in range(3)]
                r = np.stack([t[0] for t in tr])          # (3,T) measured range
                c = np.stack([t[1] for t in tr])
                risk_raw.append(wrap_fraction(r))

                out = fit_trajectory(r, c, (0, 1, 2), B=B, lam=LAM, mu=MU)
                rho = np.linalg.norm(out["p"][None, :, :] - RADAR_POS[:, None, :],
                                     axis=2)              # (3,T) reprojected
                risk_fit.append(wrap_fraction(rho))
                # col 2 = file-order index within this (gesture, subject) cell
                #         (0..99; NOT verified to equal recording time)
                # col 3 = block index = col2 // 20, for sampling-stability checks
                # col 4 = original cache row, so any later analysis can re-join
                meta.append((g, s, j, j // 20, i))
        print(f"  [{GEST[g]:9s}] done", flush=True)
    return np.array(meta), np.array(risk_raw), np.array(risk_fit)


# ---------------------------------------------------------------------------
def cell_means(meta, risk):
    """(12,8,3) mean risk per gesture x subject x view."""
    M = np.full((12, 8, 3), np.nan)
    for g in range(12):
        for s in range(8):
            m = (meta[:, 0] == g) & (meta[:, 1] == s + 1)
            if m.any():
                M[g, s] = risk[m].mean(axis=0)
    return M


def eta_squared(M):
    """Variance decomposition on a (12,8) table of cell means."""
    grand = np.nanmean(M)
    ss_g = 8 * np.nansum((np.nanmean(M, axis=1) - grand) ** 2)
    ss_s = 12 * np.nansum((np.nanmean(M, axis=0) - grand) ** 2)
    ss_t = np.nansum((M - grand) ** 2)
    return ss_g / ss_t, ss_s / ss_t


def gate_a(meta, risk, v, gests=range(12), mask=None):
    """Non-overlapping IQR pairs for view v over `gests`.  (n_pairs, n_tot, best)."""
    gests = list(gests)
    q = {}
    for g in gests:
        m = meta[:, 0] == g
        if mask is not None:
            m &= mask
        r = risk[m, v]
        q[g] = (np.percentile(r, 25), np.percentile(r, 50), np.percentile(r, 75))
    pairs, gap_best, best = 0, -1.0, ""
    for ia, a in enumerate(gests):
        for b in gests[ia + 1:]:
            lo, hi = (a, b) if q[a][1] < q[b][1] else (b, a)
            gap = q[hi][0] - q[lo][2]              # low.p75 vs high.p25
            if gap > 0:
                pairs += 1
                if gap > gap_best:
                    gap_best, best = gap, (f"{GEST[lo]} (p75={q[lo][2]:.3f}) "
                                           f"< {GEST[hi]} (p25={q[hi][0]:.3f}), "
                                           f"gap={gap:.3f}")
    n_tot = len(gests) * (len(gests) - 1) // 2
    return pairs, n_tot, best


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100,
                    help="samples per (gesture, subject) cell; 100 = the full cache")
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--fig-only", action="store_true",
                    help="redraw the figure from the saved npz")
    ap.add_argument("--analyze-only", action="store_true",
                    help="rerun the reporting from the saved npz (no tracking)")
    args = ap.parse_args()

    if args.fig_only:
        d = np.load(NPZ)
        make_fig(d["meta"], d["risk_raw"], d["risk_fit"])
        return

    print("=" * 78)
    print("W3 power check: dynamic range of the phase-wrapping risk score")
    print("=" * 78)
    print(f"  lambda = {LAMBDA_M * 1e3:.1f} mm   lambda/4 = {QUARTER * 1e3:.2f} mm"
          f"   v_unamb = {V_UNAMB:.4f} m/s @ {FPS:.0f} fps")
    print(f"  risk = fraction of frames with |d(range)| > lambda/4\n")

    if args.analyze_only:
        d = np.load(NPZ)
        meta, raw, fit = d["meta"], d["risk_raw"], d["risk_fit"]
        print(f"\n[data] {len(meta)} samples reloaded from {NPZ}\n")
    else:
        meta, raw, fit = collect(args.reps)
        # save BEFORE any analysis - collection costs ~50 min at reps=100 and must
        # never be lost to a bug in the reporting code
        np.savez(NPZ, meta=meta, risk_raw=raw, risk_fit=fit,
                 lambda_m=LAMBDA_M, quarter=QUARTER, reps=args.reps)
        print(f"\n[data] {len(meta)} samples x 3 views   [saved] {NPZ}\n")

    # ------------------------------------------------------- sanity vs 32.2%
    print("=" * 78)
    print("0  SANITY: reproduce the recorded global wrap fraction (00 4.1 = 32.2%)")
    print("=" * 78)
    for nm, R in (("raw Viterbi track", raw), ("fitted B-spline reprojection", fit)):
        per_sample = R.mean(axis=1)
        print(f"  {nm:30s} global = {100 * R.mean():6.2f}%"
              f"   per-sample p10/p90 = {100 * np.percentile(per_sample, 10):.1f}%"
              f" / {100 * np.percentile(per_sample, 90):.1f}%")
    print("\n  NOTE: one bin = 6.35 mm against a 10.28 mm threshold, so TWO bins of")
    print("  tracking jitter already register as a wrap.  Whether either number is a")
    print("  motion measurement or a tracker artefact is decided in section 3.")

    # ------------------------------------------------------- per-gesture table
    for nm, R in (("RAW", raw), ("FIT", fit)):
        print("\n" + "=" * 78)
        print(f"1  per-gesture x per-view wrap fraction [{nm}]  (mean +- sd over samples)")
        print("=" * 78)
        print(f"  {'gesture':10s}" + "".join(f"{v:>18s}" for v in VIEWS) + f"{'max-view':>12s}")
        for g in range(12):
            m = meta[:, 0] == g
            row = f"  {GEST[g]:10s}"
            for v in range(3):
                row += f"{100 * R[m, v].mean():12.1f}+-{100 * R[m, v].std():4.1f}"
            row += f"{100 * R[m].max(axis=1).mean():12.1f}"
            print(row)

    # ------------------------------------------------------- gates, both scores
    # G12 has no hand target - the tracker follows clutter, so its "risk" is not a
    # motion quantity.  Gates are reported with AND without it; the frozen protocol
    # uses the G12-excluded numbers (16 4.4).
    SETS = (("all 12", range(12)), ("no G12", range(11)))
    for nm, R in (("RAW", raw), ("FIT (model risk score)", fit)):
        print("\n" + "=" * 78)
        print(f"GATE A  dynamic range: non-overlapping IQRs between gesture pairs [{nm}]")
        print("=" * 78)
        for sn, gs in SETS:
            line = f"  {sn:8s}"
            for v in range(3):
                pairs, tot, _ = gate_a(meta, R, v, gs)
                line += f"   {VIEWS[v]} {pairs:3d}/{tot}"
            print(line)
        _, _, best = gate_a(meta, R, 2, range(11))
        if best:
            print(f"    widest pair (Right, no G12): {best}")

        print("\n" + "=" * 78)
        print(f"GATE B  confounding: gesture variance vs subject variance [{nm}]")
        print("=" * 78)
        M = cell_means(meta, R)
        for sn, gs in SETS:
            line = f"  {sn:8s}"
            for v in range(3):
                eg, es = eta_squared(M[list(gs), :, v])
                line += f"   {VIEWS[v]} eta2_g={eg:.3f} eta2_s={es:.3f}"
            print(line)

        print("\n" + "=" * 78)
        print(f"C  DESCRIPTIVE (not a gate): how much of the data is low-risk? [{nm}]")
        print("=" * 78)
        print("  NOTE: this section is DESCRIPTIVE ONLY.  Whether the Doppler readout")
        print("  beats the range domain in the low-risk stratum has NOT been measured")
        print("  here; 08 4 puts the simulated crossover at roughly 15-40% wrap, which")
        print("  the observed spread straddles.  Do not conclude anything about readout")
        print("  switching from these counts - that is the core experiment's job.")
        m11 = meta[:, 0] != 11
        for op, lbl in ((np.max, "max over views (all 3 quiet)"),
                        (np.min, "min over views (any 1 quiet)")):
            agg = op(R[m11], axis=1)
            row = f"  {lbl:30s}"
            for thr in (0.05, 0.10, 0.20, 0.30):
                row += f"  <{100 * thr:.0f}%: {100 * (agg < thr).mean():5.2f}%"
            print(row)
        print("  per-view lowest-risk gesture (no G12): "
              + ", ".join(f"{VIEWS[v]} {GEST[int(np.argmin([R[meta[:, 0] == g, v].mean() for g in range(11)]))]}"
                          for v in range(3)))

    # ------------------------------------------------------- geometric coherence
    print("\n" + "=" * 78)
    print("3  DECIDER: which risk score carries the geometry? (vs 07 P1)")
    print("=" * 78)
    print("  u_Top . e_x = 0 exactly  -> a HORIZONTAL swipe must not wrap on Top")
    print("  u_L . e_y = u_R . e_y = 0 -> a VERTICAL swipe must not wrap on L/R")
    print("  ratio = blind-view risk / sensitive-view risk;  0 = perfect geometry,")
    print("  1 = the score is isotropic noise and carries no geometric information.\n")
    print(f"  {'score':6s}{'horizontal (G1,G2)':>26s}{'vertical (G3,G4)':>22s}")
    for nm, R in (("RAW", raw), ("FIT", fit)):
        mh = np.isin(meta[:, 0], [0, 1])
        mv = np.isin(meta[:, 0], [2, 3])
        rh = R[mh, 1].mean() / (0.5 * (R[mh, 0].mean() + R[mh, 2].mean()))
        rv = (0.5 * (R[mv, 0].mean() + R[mv, 2].mean())) / R[mv, 1].mean()
        print(f"  {nm:6s}{rh:26.3f}{rv:22.3f}")
    print("\n  A ratio near 1 means the wrap count is dominated by tracking jitter")
    print("  (1 bin = 6.35 mm vs a 10.28 mm threshold), NOT by radial hand motion.")
    print("\n  !! ASYMMETRIC READING.  fit_trajectory FORCES all three views onto one")
    print("  3-D path, so a low FIT ratio is partly guaranteed by construction and is")
    print("  NOT evidence that FIT equals the true wrap rate.  What this test does")
    print("  establish is the NEGATIVE half: RAW fails to carry the per-view geometric")
    print("  pattern, so the RAW count (and the recorded 32.2%) is contaminated by")
    print("  tracking jitter.  FIT is therefore a MODEL RISK SCORE, not ground truth.")

    # ------------------------------------------------------- sampling stability
    print("\n" + "=" * 78)
    print("4  SAMPLING STABILITY: do disjoint blocks of 20 give the same answer?")
    print("=" * 78)
    nb = int(meta[:, BLK].max()) + 1
    if nb < 2:
        print("  only one block collected - rerun with --reps 100 to check stability")
    else:
        print(f"  {nb} disjoint blocks of <=20 samples per (gesture, subject) cell\n")
        print(f"  {'block':7s}{'n':>6s}{'globalRAW':>11s}{'globalFIT':>11s}"
              f"{'A/Right(noG12)':>16s}{'eta2_g Right':>14s}")
        for b in range(nb):
            m = meta[:, BLK] == b
            pairs, tot, _ = gate_a(meta[m], fit[m], 2, range(11))
            eg, _ = eta_squared(cell_means(meta[m], fit[m])[:11, :, 2])
            print(f"  {b:<7d}{m.sum():6d}{100 * raw[m].mean():10.2f}%"
                  f"{100 * fit[m].mean():10.2f}%{f'{pairs}/{tot}':>16s}{eg:14.3f}")
        print("\n  the frozen protocol may only be based on quantities that are stable")
        print("  across these blocks; anything that moves is a file-order artefact.")

    # ------------------------------------------------------- chirality focus
    print("\n" + "=" * 78)
    print("2  frozen target readout: G9/G10 chirality (the d' that must be predicted)")
    print("=" * 78)
    for g in (8, 9):
        m = meta[:, 0] == g
        print(f"  {GEST[g]:10s} per-view risk = "
              + "  ".join(f"{VIEWS[v]} {100 * raw[m, v].mean():5.1f}%" for v in range(3)))

    if not args.no_fig:
        make_fig(meta, raw, fit)


def make_fig(meta, raw, fit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4), sharey=True, sharex=True)
    for row, (nm, R) in enumerate((("RAW Viterbi track", raw),
                                   ("FIT B-spline reprojection", fit))):
        for v in range(3):
            ax = axes[row, v]
            ax.boxplot([100 * R[meta[:, 0] == g, v] for g in range(12)],
                       showfliers=False, medianprops=dict(color="crimson"))
            ax.axhline(32.2, color="grey", ls="--", lw=1,
                       label="recorded global 32.2%")
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels([g.split()[0] for g in GEST], rotation=45, fontsize=8)
            ax.set_title(f"{nm} - {VIEWS[v]} view", fontsize=10)
            ax.grid(alpha=.3, axis="y")
            if v == 0:
                ax.set_ylabel(r"wrapped frames [%]  ($|\Delta\rho|>\lambda/4$)")
                if row == 0:
                    ax.legend(fontsize=8)
    fig.suptitle("Phase-wrapping risk per gesture and view.  Only the FIT row "
                 "reproduces the geometry:\nG1/G2 (horizontal) wrap on Left/Right "
                 "but not Top;  G3/G4 (vertical) the reverse.", fontsize=11)
    fig.tight_layout()
    out = r"proposal_v2\pilot\w3_readout_risk.png"
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
