# 2026-10-29 — bounded method-candidate refresh after Betti STOP

## Decision

No new candidate cleared the current-data method gate. No training or fitting is authorized by this literature refresh. The method-contribution goal remains unresolved; this does not prove that all methods are impossible.

## Bounded evidence reviewed

- Ahmed et al., Scientific Data 8, 102 (2021), DOI [10.1038/s41597-021-00876-0](https://doi.org/10.1038/s41597-021-00876-0): the dataset/acquisition paper documents three independent monostatic left/top/right UWB views and opposite left/right range-time motion for horizontal swipes. This closes a generic left/right reciprocity observation as a new signal mechanism; framewise calibration and shared timing are absent in the current restored package.
- Xue, Yang & Zhu, Frontiers in Neuroscience 20 (2026), DOI [10.3389/fnins.2026.1864448](https://doi.org/10.3389/fnins.2026.1864448): the same UWB-Gestures corpus is used for three-view early motion fusion, local/global representations, and gated/bilinear feature fusion. Generic multi-view, global/local, or view-subset repackaging is preempted.
- Khan et al., Sensors 17(4), 833 (2017), DOI [10.3390/s17040833](https://doi.org/10.3390/s17040833): IR-UWB hand gesture descriptors include magnitude statistics, frequency, and time-of-arrival variation. Leem et al., IEEE TIM 69(4), 1066–1081 (2020), DOI [10.1109/TIM.2019.2909249](https://doi.org/10.1109/TIM.2019.2909249): IR-UWB hand localization/tracking/rendered trajectories. These close generic range/TOA path summaries as a distinct physical contribution.
- Kim et al., IEEE Access (2026), DOI [10.1109/ACCESS.2026.3671243](https://doi.org/10.1109/ACCESS.2026.3671243): the official DGIST abstract describes speed-induced scale handling on UWB-Gestures with range-Doppler multi-scale extraction. Speed/scale normalization is not an unoccupied method direction here.
- Maršálek et al., Sensors 23(4), 2224 (2023), DOI [10.3390/s23042224](https://doi.org/10.3390/s23042224): full-text radar persistent-homology target-detection precedent was reviewed earlier. The exact preregistered Betti-curve representation has now failed its fixed gate (see the run03 report); do not repackage variants of that descriptor.

This was a bounded targeted review, not an exhaustive systematic search. It does not claim exhaustive novelty clearance.

## Current observable and excluded claims

The restored predictor input is the real-valued clutter-removed map cache, X_clutterremoved with shape [9593, 3, 90, 189]. It contains no raw complex/IQ echoes, synchronized per-view frame timing, calibrated line-of-sight trajectories, or independent hand-pose/kinematic truth. The views are separate monostatic recordings with soft event-level pairing, not a measured synchronized velocity vector. Therefore Doppler-phase, calibrated 3-D motion, framewise cross-view reciprocity, or hand-kinematic invariance claims are not supported by these inputs. The independent-cohort screen also found no protocol-compatible external dynamic hand-gesture dataset.

The latest Betti feasibility result remains STOP_CANDIDATE; its detailed source-only metrics are in RUN03_REPORT.md. No labels from participants 7–8 were used in this refresh. No map values, labels, or raw arrays were read for this literature note. No process/checkpoint is running.

## Disposition

Keep the study method-first. Do not launch another fit, tune prior descriptors, or claim a validated contribution. Continue only if a materially different mechanism can be tied to an observable in the existing cache and distinguished from the cited primary work before a frozen protocol is written. If none is found, the concrete supported output is a reproducibility/quality-audit scope, which is not equivalent to the requested new algorithm. No new data request is made.
