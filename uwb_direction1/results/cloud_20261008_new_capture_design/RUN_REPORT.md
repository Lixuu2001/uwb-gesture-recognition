# 2026-10-08: synchronized external-truth capture protocol

## Outcome

Completed a bounded literature review and wrote `research/NEW_CAPTURE_PROTOCOL_20261008.md`. This is a design artifact only, not a new experiment, validated method, or contribution claim. No training process was started. The restored data remained in place and was not re-downloaded.

## Latest branch state and environment

- Branch head read from GitHub before this update: `7418489711410419acf6c23fa869d0b263ab4638` (2026-10-08 08:46:59 UTC). This additive report and protocol were then committed and pushed normally as `245ffaea313b12b2553f4268ad0723d84d61a60c`.
- Latest status says the paired-time alignment screen failed its continuation gate; no labels/model fitting were used for that screen. The direct same-dataset prior-art report records the 2026 three-view paper and duplicate audit, while explicitly leaving the research goal unmet.
- Restore directory `/workspace/uwb-restored-20260930` exists and its restore verification record is present. Existing cloud process scan found no live training process; historical PIDs shown as defunct are not active jobs.
- No X4 capture interface, optical-motion-capture feed, or new cohort/session/placement data is attached to this cloud runtime.

## Literature evidence and implications

- Ahmed et al. (2021) document the UWB-Gestures dataset as 12 classes, eight participants, three independently operated monostatic X4 views saved separately, with 90 frames per 4.5-second sample (20 frames/s). This tells us what the existing dataset contains; it does not validate a new method. DOI: `10.1038/s41597-021-00876-0`.
- Kern et al. (2022) give a direct precedent for common-trigger radar/stereo capture, camera-radar spatial calibration using a shared target, and extension to multiple radars. Their four-radar 79-GHz setup is different hardware. Official paper PDF: `2022_GeMiC_Kern_Ground_Truth_System_for_Radar_Measurements_of_Humans.pdf`.
- Engel et al. (2025) demonstrate synchronized optical motion capture and mmWave radar with hardware triggering and shared-target coordinate registration, plus participant-held-out pose evaluation. The paper reports 12 participants; it uses mmWave FMCW rather than X4 IR-UWB. Official repository PDF and dataset DOI are recorded in the protocol.
- Xue et al. (2026) already cover three-view fusion on the same UWB-Gestures dataset. The protocol therefore excludes generic fusion as a novelty claim.

## Decision

The only currently defensible forward action is a new, synchronized capture that can answer a physical observability question under participant/session/placement holdout. This experiment cannot run in the cloud without capture hardware and new measurements. The proposed single-vs-three-view hand-trajectory/gesture question remains a candidate pending direct prior-art review; no novelty claim or training queue is approved by this report.

## Reproducibility and scope

No arrays, labels, fitting, or optimizer runs were used in writing the protocol. No new training, test, or validation scores are reported. No existing data, source snapshot, or model artifact was modified.
