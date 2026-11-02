# Radial-velocity projection-norm candidate gate

**Date:** 2026-10-25 UTC  
**Decision:** not cleared for fitting or training; no new model experiment.

## Candidate and falsifiable mechanism

Screened whether three radar views could yield a view-complementary radial-motion statistic that approximates gesture speed magnitude, then explain Right-only errors beyond the fixed Right baseline. A valid physical velocity norm would require simultaneous per-frame radial-velocity observations and calibrated line-of-sight vectors. If only independently repeated gesture clips are paired, their aggregate energies are not components of one observed velocity vector. The falsifiable claim therefore cannot be tested as a physical reconstruction from the current package.

## Available inputs and acquisition constraint

The source dataset paper (Ahmed et al., *Scientific Data* 8, 102 (2021), DOI [10.1038/s41597-021-00876-0](https://doi.org/10.1038/s41597-021-00876-0)) states that left/top/right X4 radars operate independently in monostatic mode and each transmits/receives independently. It describes 90 slow-time rows per 4.5-second sample, but the released project artifacts do not provide a shared trigger, per-frame timestamps, or calibrated line-of-sight/extrinsic records. The restored cache is clutter-removed real-valued amplitude-like range-time maps. Same-index view pairing is documented as soft synchronization, not simultaneous sample-level velocity measurement. No labels, tensor values, or checkpoint contents were read for this gate.

## Prior art and narrow difference

- The dataset paper already describes the left/right range-time signature of a horizontal swipe moving away from the left radar and toward the right radar. This is a published view-dependent radial-motion cue, not a new mechanism.
- Two-radar velocity reconstruction from distinct radial projections is established in mmWave gesture sensing; see the *Sensors* framework “A Low-Complexity Hand Gesture Recognition Framework via Dual mmWave FMCW Radar System” (2023), [DOI 10.3390/s23208551](https://doi.org/10.3390/s23208551). It uses radial velocities and azimuth geometry to recover velocity components. Its sensor differs, but the projection principle itself is prior art.
- The same-corpus Xue et al. 2026 paper already models left/top/right TR views with multiview fusion. Project gates on multi-radar trajectory reconstruction likewise found the current cache lacks synchronization, calibrated geometry, raw complex echoes, and external hand-pose truth (see `results/cloud_20261011_three_radar_trajectory_prior_art/RUN_REPORT.md` and `research/PAIRWISE_TIME_ALIGNMENT_FEASIBILITY_PROTOCOL_20261008.md`).

The narrow proposal to sum per-view statistics from separately recorded repetitions might be numerically computable, but it would not be an observed vector norm. Without a signal-specific hypothesis beyond generic multiview aggregation, it does not clear the method-contribution gate.

## Decision and stop condition

Do not implement a feature or train a classifier from this proposal. Reconsider only if an already-authorized data source contains simultaneous timestamps plus surveyed/calibrated view geometry, and a primary-source review establishes a specific method-level gap. Otherwise, a score comparison on the existing repeated-development cohort would test an empirical aggregate feature, not the claimed physical mechanism or a validated novel method.

## Execution record

The cloud executor was running at spec revision 1385 with restricted/enforced networking, 5 CPUs, and no added capabilities, secrets, runtime variables, or outbound identities. The restored data directory and receipt were present; no Python training process or checkpoint file was found within the inspected workspace scope. No training, data download, or array/label read occurred.
