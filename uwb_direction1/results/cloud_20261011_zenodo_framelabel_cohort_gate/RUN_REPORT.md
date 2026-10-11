# Zenodo frame-labeled 60 GHz gesture cohort gate

Date: 2026-10-11

## Finding

The public Zenodo record `10.5281/zenodo.15178095` describes a BGT60TR13C 60 GHz FMCW corpus with 49,000 recordings: 25,000 nominal recordings and 24,000 anomalous recordings. Nominal labels are swipe left/right/up/down and push, with 12 participants. The anomaly conditions are fast, slow, and wrist execution; those samples use eight participants and were all collected at location e1. Each recording has 100 frames, raw inputs shaped [recording, frame, 3 antennas, 32 chirps, 64 samples], and per-frame targets. The record says raw data is unchanged from the earlier IEEE DataExplorer release, so those releases are one cohort, not two independent datasets. The archive is listed as 36.5 GB with MD5 `d63a9645f27ebded613807b674bba63a`.

Primary source: [Zenodo record and dataset metadata](https://zenodo.org/records/15178095). The record's visible Rights/License field was blank during this review; an OpenAIRE index labels it CC BY, but that secondary metadata does not resolve the record-level license. Treat reuse terms as unverified until the authoritative Zenodo/API metadata or deposit terms confirm them.

## Relevance and limits

The five nominal class names overlap semantically with some X4 gestures, but names alone do not verify gesture instructions, orientation conventions, or one-to-one event labels. Its FMCW raw ADC representation, 60 GHz sensor, 100-frame sequences, and single-radar setup differ from the restored X4 IR-UWB three-view clutter-removed range-time data. The anomaly subsets are also not balanced across participants or rooms, so an evaluation must keep those acquisition factors explicit.

Cross-sensor transfer itself is prior art: Jung et al., *Digit Recognition Using FMCW and UWB Radar Sensors: A Transfer Learning Approach*, IEEE Sensors Journal (2023), DOI [10.1109/JSEN.2023.3289232](https://doi.org/10.1109/JSEN.2023.3289232), applies transfer between FMCW and UWB radar to air-written digits using range-time and Doppler-time representations. This is not the same hand-gesture task, but it blocks a broad claim that transferring between UWB and FMCW is novel. DI-Gesture and the newly published radar DG literature further block generic domain-generalization/augmentation claims. A distinct hand-gesture mechanism still needs prior-art clearance and evidence.

## Decision

This is a stronger public-cohort lead than the agreement-gated MCD archive because its record exposes a downloadable file and frame-level labels. It is not yet an experiment candidate:
- authoritative reuse license is not confirmed;
- common class semantics/instructions and orientation mapping are not verified;
- the 36.5 GB archive is large, and the cloud runtime is CPU-only;
- no radar-specific transfer method has passed the novelty gate.

Therefore the archive was not downloaded and no project labels, radar arrays, or checkpoints were read. No fitting or training ran. Existing X4 restore remains available; no active job was found. Next action is metadata/license and paper-method review, followed by an explicit feasibility/protocol gate before any download.

## Scope

Bounded source and prior-art screen on 2026-10-11. This is not an exhaustive search or a full-text review of every cited paper.


## Current cloud execution feasibility check

Observed 2026-10-11 13:51:34 UTC: managed cloud executor is running (spec revision 616), but only CPU capabilities are exposed; HTTP policy is restricted/enforced with no user egress hosts or outbound identities. `/workspace` has 712 MB free on a 32 GB filesystem. The restored X4 directory is present. Process inspection found only historical defunct Python entries and no active training process. The 36.5 GB Zenodo archive cannot be staged on the current volume. No alternative persistent volume, external storage binding, or larger execution environment is attached. This is an exact resource/access blocker, not a failed experiment. No labels, radar array values, or checkpoints were read, and no training was launched.


## Additional radar gesture domain-generalization prior art

A primary university research portal record for Sukianto et al., *Unsupervised Federated Learning With Harmonized Foundation Models for FMCW Radar-Based Hand Gesture Recognition*, IEEE Transactions on Radar Systems 4 (2026), DOI [10.1109/TRS.2026.3654128](https://doi.org/10.1109/TRS.2026.3654128), was reviewed at abstract/metadata level. The described method aligns FMCW range/Doppler dimensions across device configurations, simulates missing antenna channels, pretrains a transformer, fine-tunes with 20 labeled examples per class, distills a lightweight model, then uses unlabeled federated refinement. This closes generic claims based only on FMCW device harmonization, missing-channel augmentation, few-shot tuning, or federated adaptation. It is not evidence that the specific IR-UWB-to-FMCW hand-gesture setting has been solved: the described cross-device experiments use FMCW devices, while the 2023 UWB↔FMCW transfer precedent concerns air-written digits. The full 2026 article was not reviewed, so these scope statements follow only the institutional portal abstract and must not be treated as an exhaustive novelty review.

The method gate remains unresolved. A future candidate must specify a gesture-specific, radar-signal mechanism beyond generic harmonization/augmentation/adaptation and be supported by full-text prior-art review before any download or training.


## Public implementation precedent on the same FMCW cohort

The dataset repository's public implementation `uctopus/hand_gesture_recognition` documents a frame pipeline that extracts range bin, Doppler bin, azimuth, elevation, and magnitude; it detects a gesture window by a signal-thresholded minimum-range frame, uses a fixed 20-frame window, forms background windows, and classifies the six gesture/background classes with a small two-layer LSTM. The code is available at [repository](https://github.com/uctopus/hand_gesture_recognition), [preprocessing implementation](https://raw.githubusercontent.com/uctopus/hand_gesture_recognition/main/src/data_preprocessing.py), and [LSTM implementation](https://raw.githubusercontent.com/uctopus/hand_gesture_recognition/main/src/model.py). Its README-reported metrics are self-reported and were not independently reproduced. This is implementation-level prior art for peak-based temporal localization plus lightweight sequence classification on the FMCW corpus; it is not an X4 result and the five engineered features do not directly match the restored X4 input.

This further narrows any future event-localization candidate: it must go beyond a minimum-range anchor, fixed-length window, background windows, and a generic LSTM. No external data were downloaded and no project fitting ran.
