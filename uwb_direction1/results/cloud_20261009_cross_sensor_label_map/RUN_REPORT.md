# 2026-10-09 — cross-sensor gesture label-map feasibility

## New finding

A metadata-only, coarse two-family mapping is possible: eight UWB swipe labels and Pantomime left/right swipe labels map to "swipe"; UWB clockwise/counterclockwise rotations and Pantomime clockwise/counter-clockwise circles map to "circular". This corrects the prior report's statement that no metadata-defined common labels existed. UWB inward push is deliberately excluded because the label alone does not establish equivalence to Pantomime's push. Empty/no-gesture and all nonmatching Pantomime gestures are excluded. This creates a closed-set two-family candidate endpoint; it does not create a common 12-class task.

The dataset papers themselves motivate swiping/cyclic rotation as common gesture families, and existing work includes general RGB/depth adaptive cross-modality gesture sensing, cross-device FMCW adaptation, and UWB-to-FMCW transfer for digit recognition. Therefore neither the label map nor generic domain adaptation is a novelty claim. A bounded search found no direct IR-UWB/X4-to-FMCW-point-cloud dynamic hand-gesture transfer match; this is not exhaustive prior-art clearance. The possible contribution remains a research question about transfer under participant-disjoint evaluation using a properly defined cross-modal representation, not a demonstrated method.

## Execution state

The Pantomime Zenodo record lists a public 107.6 MB dataset archive and 937.5 kB source archive. This cloud executor remains CPU-only with restricted egress and no configured capabilities, secrets, or outbound identities. Archive license/internal manifest still need confirmation before use. No archive downloaded, no external sample labels or UWB labels/arrays read, and no fitting or training occurred. Process inspection found no active experiment; prior UWB data and the 789-file original-source snapshot remain present and untouched.

## Next gate

1. Review direct prior art specifically for family-level cross-radar gesture transfer and inspect the permitted archive's license/manifest path.
2. If access is valid, freeze participant-disjoint folds, source/target direction, target-label budget (including zero-shot), class aggregation, and a cross-modal representation before reading target labels or fitting.
3. Stop if class balance/sequence semantics do not support the frozen endpoint or if a matching prior method is found. Do not report native UWB accuracy as cross-sensor transfer.

The overall research goal remains unvalidated. No training is running.


## Additional prior art check (2026-10-09)

Hazra et al. (SAM 2022; DOI 10.1109/SAM53842.2022.9827785) report cross-learning from camera point clouds to 60-GHz FMCW radar point clouds for gesture recognition. This directly preempts a broad cross-modal point-cloud gesture learning claim. It does not test IR-UWB/X4 CIR against FMCW radar point clouds, so that specific transfer endpoint remains unvalidated rather than novelty-cleared. This is a bounded metadata/abstract check only; no archive download, labels, arrays, or model fitting.
