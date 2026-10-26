# Motion-primitive label factorization candidate gate

**Date:** 2026-10-26 00:55:36 UTC  
**Decision:** not cleared for fitting or training; no model experiment.

## Candidate and falsifiable mechanism

Screened a label-structured learner that decomposes the eight documented swipe classes into shared ordered motion factors (horizontal/vertical direction components), then composes class scores. The testable mechanism would be that shared supervision improves held-user recognition of diagonal and direction-opposed swipe classes over a capacity-matched direct 12-class readout, with a frozen factor-target permutation control. Single-view or three-view inputs could be used; the method would not claim calibrated hand kinematics.

This is a plausible engineering experiment, but this review does **not** establish a research contribution. A generic hierarchical/multitask classifier or a relabeling of known gesture primitives is not novel by itself, and no specific radar signal mechanism was identified beyond the known class ontology.

## Literature reviewed and scope

- Ahmed et al., *Scientific Data* 8, 102 (2021), DOI [10.1038/s41597-021-00876-0](https://doi.org/10.1038/s41597-021-00876-0): publisher full text read for the gesture vocabulary, acquisition, and signal-pattern sections. It defines the eight directed swipe classes, two rotations, inward push, and empty class; it also documents the independently operating monostatic L/T/R radars and view-dependent range-time patterns. This already supplies the ontology that the candidate would factorize.
- Chen et al., *Electronics* 15(2), 437 (2026), DOI [10.3390/electronics15020437](https://doi.org/10.3390/electronics15020437): publisher article/abstract and gesture-definition section reviewed. It explicitly aligns radar gestures to motion primitives such as lateral swipe and radial push/pull. Its FMCW data and classes differ, but it further narrows any broad “primitive-based radar gesture recognition” claim.
- Li et al., *Sensors* 23(20), 8551 (2023), DOI [10.3390/s23208551](https://doi.org/10.3390/s23208551): article description reviewed for dual-radar radial-velocity decomposition and direction recovery. This is physical direction inference, not semantic label factorization; it preempts generic claims that combining radar views to recover motion direction is itself new.
- Xue et al., *Frontiers in Neuroscience* 20 (2026), DOI [10.3389/fnins.2026.1864448](https://doi.org/10.3389/fnins.2026.1864448): same-corpus tri-view recognition/fusion prior art is already recorded and remains applicable. This candidate is not a fusion architecture claim.

This was a bounded screen, not an exhaustive search or full review of all related work. The search did not establish a directly identical factorized-label algorithm, but it also did not find a sufficiently narrow gap that clears a novelty claim. “Not found” is not evidence of novelty.

## Observable inputs and evaluation limits

The current package has clutter-removed, real-valued range-time maps plus class and participant metadata. It does not provide synchronized hand-pose labels or calibrated physical trajectory truth. Label factors can be derived from the published class instructions, but that would test a semantic decomposition of existing labels—not a measured physical primitive. The cache may support a descriptive/classification probe, but all eight users have already appeared in project development; held-user folds on this cohort cannot be presented as untouched independent validation. No tensor values or label values were read for this screen.

A valid future experiment, if separately authorized after a stronger novelty case, would need a preregistered matched-capacity direct-class baseline, one fixed factorized head, and one fixed factor-target permutation control on the same user folds. All choices would be confined to training/validation users, with the held user scored once. Stop if the factorized head does not beat both controls on the frozen aggregate and recur across users; no alternate primitive maps, class subsets, loss weights, or seeds to rescue a miss. This experiment is **not authorized** by this screen or the current status.

## Execution record and disposition

Current managed executor: revision 1389, running, restricted/enforced egress, 5 CPUs, PyTorch unavailable. Restored data at `/workspace/uwb-restored-20261020` and the 789-file source subtree at `/workspace/uwb-cloud-current/uwb_direction1/original/project` are present. No active UWB training process or serialized checkpoint was found. No data download, array/label read, fitting, or training occurred.

**Disposition:** candidate remains uncleared; do not implement/train it yet. The method-contribution goal remains unresolved. Continue bounded literature-only screening or await a genuinely distinct mechanism; do not reopen closed feature, calibration, generic fusion, or time-reversal routes.
