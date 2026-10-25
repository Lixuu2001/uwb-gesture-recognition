# Bounded research decision review after Stage A run02

**Date:** 2026-10-25 UTC  
**Review boundary:** literature and existing project records only. No new model training, fitting, parameter search, dataset download, or label/array read was performed for this review. Branch at review start: `018361d349699e5a114505c443cf9b2e6e9f8dd6`.

## Decision

No candidate currently clears all three conditions for another method experiment: (1) a mechanism that is materially different from documented prior art and the project's closed routes, (2) an observable present in the current clutter-removed maps, and (3) a valid, frozen evaluation with the current repeated-development users. Two plausible signal-level angles were screened below and both are closed at the novelty/measurement gate. Do not start either experiment from this review.

The latest L/T privileged-strata result is a narrow descriptive screen, not a contribution: incremental CMI was 0.011062 bits/sample, but only 1/12 classes met the coverage criterion and 1/12 met the cross-user error-direction criterion. The 8/12 continuation gates failed. This closes that frozen representation/probe; it does not refute all extra-view learning. The source trial/session dependence remains unknown, so no inference was used.

## Two candidate angles reviewed and rejected

### 1. Paired-view range-rate reciprocity for horizontal gestures

**Question and falsifiable mechanism.** For a horizontal swipe, does the sign of the target range-time slope in the left radar oppose that in the right radar, while the top view has a smaller signed change; does a frozen reciprocity residual explain a Right-only model's errors beyond a fixed Right baseline? A falsifying outcome would be unstable signs across held users, no incremental held-user performance over the fixed Right-only control, or dependence on post-hoc class/view choices.

**Available input.** The restored cache contains clutter-removed, real-valued per-view maps (L/T/R, 90 slow-time rows by 189 range bins). It has no raw complex samples, shared frame trigger, per-view timing calibration, or synchronized hand-pose trajectory. Pairing is by participant/gesture/repetition and was reviewed as soft synchronization; it does not establish framewise correspondence.

**Nearest original literature and scope read.** Ahmed et al., *Scientific Data* 8, 102 (2021), DOI [10.1038/s41597-021-00876-0](https://doi.org/10.1038/s41597-021-00876-0): publisher full text sections on acquisition, data records, and signal-pattern analysis. It documents three independent monostatic left/top/right radars, their nominal geometry, and explicitly illustrates opposite left/right range-time motion for a horizontal swipe. Xue, Yang & Zhu, *Frontiers in Neuroscience* 20 (2026), DOI [10.3389/fnins.2026.1864448](https://doi.org/10.3389/fnins.2026.1864448): original full-text abstract, introduction, signal model, and method/contribution sections reviewed; same-corpus three-view motion-enhanced TR early fusion, local/global branches, and gated/bilinear feature fusion are already proposed. The project also has a 2026-10-11 paired-event candidate screen; it found same-event correspondence is soft-synchronized and that a held-view split alone is not a method contribution.

**Narrow difference and why it does not clear.** A hand-coded signed reciprocity residual would be more specific than generic fusion, but the dataset paper already shows this basic left/right signal behavior. The proposed physical interpretation needs calibrated, synchronized range tracks or external motion truth that are absent. Calling a view-derived feature a calibrated physical invariant would overstate the input. No distinctive method-level contribution remains after these constraints.

**Minimal comparison if a new, technically distinct mechanism were later established.** Use the existing eight held-user outer folds, with all transforms and any thresholds fit on each fold's training users only. Compare one fixed Right-only baseline against the same baseline plus one preregistered L/R reciprocity statistic; include a fixed standard three-view fusion control only if its score is already available under the identical folds. Freeze class mapping, metric, thresholds, and all row pairing before reading any held-user result. This comparison is not authorized now and would still be within the repeatedly developed eight-user corpus.

**Stop conditions.** Close before fitting if source data do not contain trustworthy event-level view alignment/calibration; close after a preregistered run if the increment is absent or does not recur across held users. Do not rescue it by searching classes, slopes, offsets, or filters.

### 2. Speed-normalized range-time path descriptor for single-view Right deployment

**Question and falsifiable mechanism.** Does a compact range-time path descriptor (for example, a fixed dominant-range trajectory and its signed segment durations) capture a stable speed-normalized signal property that explains Right-only errors beyond existing image-like features? The claim would require a repeatable held-user gain over the frozen Right baseline, not just a visually plausible path or an in-sample class correlation.

**Available input.** The current maps provide time and range axes after clutter removal, so a numerical range-time path can be derived. They do not provide hand-pose ground truth or calibrated physical displacement. A ridge/centroid track would be an algorithmic summary of echo energy, not verified hand kinematics.

**Nearest original literature and scope read.** Khan et al., *Sensors* 17(4), 833 (2017), DOI [10.3390/s17040833](https://doi.org/10.3390/s17040833): publisher abstract and indexed method description reviewed; it uses TOA variation as a motion-range cue and clustering with distance/orientation robustness. Leem, Khan & Cho, *IEEE Transactions on Instrumentation and Measurement* 69(4), 1066–1081 (2020), DOI [10.1109/TIM.2019.2909249](https://doi.org/10.1109/TIM.2019.2909249): author-institution abstract reviewed; it localizes, tracks, and renders IR-UWB hand trajectories for recognition. Kim et al., *IEEE Access* 14 (2026), DOI [10.1109/ACCESS.2026.3671243](https://doi.org/10.1109/ACCESS.2026.3671243): official DGIST record/abstract read, not full text; same-corpus speed-induced scale variation is addressed with range-Doppler multi-scale extraction. The same-corpus Xue et al. (2026) paper also models motion-enhanced TR maps. The project's 2026-10-11 range-trajectory prior-art gate already closed generic range-ridge/centroid classification, and the historical temporal-scale/moment-guided probes did not clear their continuation criteria.

**Narrow difference and why it does not clear.** Restricting the descriptor to Right view or replacing a network with a hand-engineered path does not distinguish a physical mechanism from TOA/trajectory/speed-scale prior art. No trial-level speed or hand path is available to test the asserted invariance. This would be a new feature extractor and model-selection exercise, not a supported original mechanism.

**Minimal comparison if a distinct observable were later established.** On the same eight held-user folds, compare the frozen Right-only baseline to one preregistered descriptor and a matched-capacity non-trajectory control; fit normalization and any thresholds on training users only. Use one predeclared aggregate metric and report all eight folds. This is not authorized now, and a result on these repeatedly reused eight users would remain within-corpus evidence.

**Stop conditions.** Close before fitting unless a primary-source review finds a specific method-level gap and an independent observable validates the physical interpretation. If a future authorized run shows no aggregate gain or the gain is concentrated in a previously developed user/class, stop; no class-specific tuning or descriptor grid.

## Leakage, dependence, and external-validity limits

- Users HV3–HV8 have prior project-development use; even a clean leave-one-user-out replay is not untouched external validation. The 900-s/900-s run02 was a single descriptive audit, not a new independent cohort.
- HV7/HV8 labels were used only as post-fit diagnostic class-conditioning in Stage A and are not deployment-time inputs.
- The 100 exact duplicate triplets map to participant 2 = sealed HV2, outside source users HV3–HV8; run02's pre-fit scan found 0 exact aliases and 0 digest collisions across 7,100 retained source rows. Duplicate provenance remains unresolved, and no trial/session identifiers establish record-level exchangeability.
- The incremental CMI compares L/T only against the fixed binary Right K=2 partition, not all information in Right. Clusters are statistical strata, not verified kinematic groups.
- Same-corpus accuracy and alternate release records cannot establish external generalization; the public Figshare item is a processed release of the same UWB-Gestures corpus.

## Evidence-supported scope choices

1. **Method-contribution paper:** not supportable from the reviewed candidates and current cache. Keep the method-contribution field false. Resume method development only after a materially distinct observable clears primary-source review and a protocol can avoid tuning on the repeatedly used eight people.
2. **Empirical/reproducibility paper:** possible as a different scope, not as a method claim. It could report deterministic cache restoration, faithful baseline replay, cross-user evaluation protocol, and carefully bounded negative/heterogeneity findings. It would need a literature-positioned question and full, transparent baseline/protocol comparison; the Stage A result alone is not sufficient and carries no inferential claim.
3. **Engineering result:** the verified cloud restore, source-only artifact separation, hard-budget runner, hashed protocol, and reproducible descriptive report are concrete engineering outputs. They demonstrate execution integrity, not a new sensing algorithm.

No new data are required to make this decision review. This review did not authorize any training or ask for a dataset re-upload. The current study line is paused for owner review of research scope; no method contribution is claimed as complete.
