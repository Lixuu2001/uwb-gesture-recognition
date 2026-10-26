# Range/time augmentation candidate: bounded prior-art and observability gate

**Date:** 2026-10-26 22:26 UTC  
**Parent before update:** 795157646528b86b4e0a9679ae91c5e9d606167b  
**Decision:** do not start an experiment for generic range/aspect/time augmentation on the current UWB-Gestures clutter-removed maps.

## Candidate examined

A candidate would augment each training gesture with range translation/scaling, aspect-angle change, or duration/time scaling, intending to simulate cross-user execution variation and improve held-user recognition.

## Primary literature check

- Hassab, Stadelmayer & Lurz, “Physically-Interpretable Data Augmentation for Multi-Range Hand Gesture Recognition Using FMCW Radar Time Series,” *IEEE Transactions on Radar Systems* 1 (2023), 571–582, DOI [10.1109/TRS.2023.3320869](https://doi.org/10.1109/TRS.2023.3320869). The paper manipulates extracted physical time series (range, azimuth, elevation) to simulate range/aspect variation and reports multi-range gesture experiments. The signal modality differs, but generic physical range/aspect augmentation is established prior art.
- Fusco et al., “Enhancing FMCW Radar Gesture Classification With Physically Interpretable Data Augmentation,” *IEEE Access* 13 (2025), 60556–60569, DOI [10.1109/ACCESS.2025.3556565](https://doi.org/10.1109/ACCESS.2025.3556565). Its stated transformations include time scaling, range/angle changes, and noise on physically interpretable radar features; this further preempts presenting those operations alone as new.
- Kim et al., “Robust UWB Radar Gesture Recognition Addressing Speed-Induced Scale Variations via Multi-Scale Feature Extraction,” *IEEE Access* 14 (March 2026), 40034–40041, DOI [10.1109/ACCESS.2026.3671243](https://doi.org/10.1109/ACCESS.2026.3671243). The official DGIST record describes the same UWB-Gestures corpus and a range-Doppler multi-scale treatment of speed-induced scale mismatch. Generic speed/scale normalization is therefore especially close to existing work on this corpus.

This is a targeted source screen, not an exhaustive literature review or a claim of protocol equivalence.

## Input feasibility in this cloud

The restored input inventory records only a real-valued clutter-removed range-time tensor, `X_clutterremoved[9593,3,90,189]`; it does not include raw complex echoes, calibrated range/angle trajectories, or synchronized hand-motion measurements. A shift or warp can be numerically applied to a map, but the current files do not establish that its amplitude, clutter residue, and target motion correspond to the intended physical range/aspect transformation. Calling such a heuristic “physical” would be unsupported.

No array values, labels, or checkpoints were read for this gate; no fitting, training, or synthetic perturbation was run.

## Disposition

Close generic range/aspect/time augmentation on the present maps as not cleared for a novel-method experiment. This does not show that every UWB-specific augmentation is ineffective. A future candidate would need a distinct observable mechanism and a mapping whose physical interpretation is identified by available measurements, then a frozen user-held-out control. No such mechanism is established here. The overall method-contribution goal remains unresolved.

## Runtime check

Executor revision 1437 is running at the current workspace with five CPU cores; NumPy/SciPy/scikit-learn are importable and PyTorch is absent. The restored data and 789-file source snapshot are present. The metadata-only process/checkpoint scan found no active UWB/training job and zero common serialized checkpoint files within workspace depth 7. No raw-data restore or Git source-snapshot change was made.
