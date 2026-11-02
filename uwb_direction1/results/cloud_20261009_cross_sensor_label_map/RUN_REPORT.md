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


## Direct same-dataset multi-view and cross-frequency prior-art audit (2026-10-09T07:46:13Z)

### Exact UWB-Gestures overlap

A full-text review of Xue, Yang & Zhu, *Neuromorphic-inspired multi-view global-local fusion for IR-UWB radar dynamic gesture recognition* (Frontiers in Neuroscience, 19 June 2026; DOI 10.3389/fnins.2026.1864448) found direct overlap with the user's UWB-Gestures source and the obvious one-/three-radar direction. The paper identifies Ahmed et al.'s public 12-class UWB-Gestures dataset, uses the same XeThru/X4 collection with left, top, and right views, and reports 9,600 samples from eight subjects. It evaluates subject-independent 6-subject train / 2-subject test plus single-, dual-, and tri-view inputs. Its Table 6 reports MV-GLFNet 88.13% left-only, 81.29% right-only, 77.82% top-only, 91.06% left+right, 95.12% left+top, 96.71% right+top, and 98.29% three-view. Therefore basic one-versus-three view comparison, same-dataset multi-view fusion, and the named global/local + gated first-/bilinear second-order fusion idea are already prior art. Do not pitch a comparable architecture or view-count result as novel. The paper's single 6/2 subject split is not directly score-comparable to this project's frozen eight-fold protocol.

### Cross-frequency radar transfer prior art

Full text of Gurbuz et al., *Cross-Frequency Training with Adversarial Learning for Radar Micro-Doppler Signature Classification* (SPIE Radar Sensor Technology XXIV; DOI 10.1117/12.2559155) confirms a synchronized/co-located dataset using XeThru X4 IR-UWB (about 7–10 GHz), 24-GHz FMCW, and 77-GHz TI IWR1443, with six participants and 11 activities/gaits. It explicitly studies cross-frequency/sensor pretraining and transfer, including CAE, optical initialization, and kinematically filtered GAN synthetic data. This preempts a broad claim that X4-to-FMCW radar transfer or cross-frequency radar-domain adaptation is new. Its listed tasks are activities/gaits rather than the UWB-Gestures/Pantomime swipe/circle endpoint, so that exact cross-dataset hand-gesture family question remains unvalidated; the narrower pairing does not itself establish novelty.

### Archive and execution gate

The official Pantomime Zenodo record (10.5281/zenodo.4459969) lists a 107.6-MB dataset ZIP and a 937.5-kB source ZIP, but the visible Rights → License value is blank. The dataset preview explicitly says it does not show all files; opening the source ZIP via the preview tool failed before exposing its contents. No archive was downloaded. The cloud executor is CPU-only; its enforced egress policy has no Zenodo host allowance and no configured capabilities, secrets, or outbound identities. No external samples/labels, UWB labels, or radar arrays were opened, and no fitting/training process was started. The restored UWB data and 789-file source snapshot remain present and untouched.

### Decision

No current candidate now clears both novelty and feasibility. Standard one-/three-view fusion on UWB-Gestures is directly covered by 2026 prior art; broad cross-frequency transfer is also covered. Pantomime-based cross-dataset transfer cannot proceed until a valid license/permission and archive manifest are verified, and the exact task-specific novelty remains open. A future run should first resolve rights/data access, then complete direct-prior-art review and freeze participant-disjoint folds, direction, target-label budget, semantic class map, and representation before any target labels are read or any model is fitted. Do not start the old multi-view or generic domain-adaptation grid.

Primary sources: Xue et al. full article https://doi.org/10.3389/fnins.2026.1864448 ; Gurbuz et al. article https://doi.org/10.1117/12.2559155 ; official Pantomime record https://zenodo.org/records/4459969 ; CI4R-MULTI3 author dataset page https://research.ece.ncsu.edu/ci4r/ci4r-multi3/ .
