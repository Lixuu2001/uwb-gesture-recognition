# New-capture protocol: synchronized hand-motion truth for IR-UWB gestures

**Status:** design only; no capture hardware was available in this cloud session, and no labels, radar arrays, fitting, or training were used to create this protocol. This is not a validated method or a claim of novelty.

## Why a new capture is the next useful step

The restored UWB-Gestures tensor contains three 90-frame range-time views per gesture instance, but the delivered sample representation does not preserve a common per-frame clock or external hand-motion truth. Existing fixed-cache experiments have exhausted the currently justified recognition/readout candidates, and a 2026 paper already reports three-view fusion on this same dataset. Another architecture or fusion sweep would therefore not answer a new physical question.

Direct literature establishes the acquisition requirements, while also setting a high novelty bar:

1. Ahmed et al. describe three independently operated monostatic X4 radars, each view saved separately, with 90 slow-time frames per 4.5-second gesture sample at 20 frames/s. This supports single- and three-radar capture, but the paper does not establish that the current cloud arrays have frame-level synchronization metadata. [Dataset paper](https://doi.org/10.1038/s41597-021-00876-0)
2. Kern et al. demonstrate synchronized radar and stereo-camera pose ground truth, a shared trigger, and per-radar camera-to-radar calibration using a target visible to both systems; their setup uses four 79-GHz MIMO radars and explicitly extends to radar networks. This is precedent for the measurement design, not an IR-UWB method contribution. [2022 paper PDF](https://mwt-www.e-technik.uni-ulm.de/downloads/papers/2022/2022_GeMiC_Kern_Ground_Truth_System_for_Radar_Measurements_of_Humans.pdf)
3. Engel et al. use 12-camera optical motion capture, a hardware input trigger at 15 Hz, and shared corner-reflector/optical-marker registration for mmWave pose estimation; they report 12 participants and held-out participant evaluation. Thus “radar plus synchronized optical pose” and pose-supervised radar learning are already established ideas. The paper uses FMCW mmWave, not X4 IR-UWB. [2025 paper PDF](https://push-zb.helmholtz-munich.de/deliver.php?id=41193)
4. Xue et al. report three-view global/local and adaptive fusion on the same public UWB-Gestures data. Generic three-view fusion is therefore not an eligible novelty claim here. [2026 same-dataset paper](https://doi.org/10.3389/fnins.2026.1864448)

## Candidate research question (unvalidated)

**Physical-observability question:** with externally measured hand motion registered to radar coordinates, when does one X4 view fail to distinguish gestures whose hand trajectories have similar line-of-sight components, and does adding the other two views resolve that ambiguity on participants, sessions, and placements held out from development?

This is a candidate measurement study, not yet a proposed novel algorithm. Before any model is implemented, perform a scoped literature review of IR-UWB hand trajectory/pose estimation, range-only multi-radar tracking, and radar-to-optical supervision. If the observation and evaluation are already covered, stop or formulate a materially different hypothesis before collecting/modeling. Do not present synchronization, camera calibration, multi-view fusion, or pose supervision by themselves as original contributions.

## Capture plan

### 1. Hardware and synchronization preflight

- Start with the smallest viable setup: one X4 radar plus an external hand-motion reference. Add the left/top/right three-view setup only if the research question requires cross-view geometry and all three X4 units can be synchronized and calibrated.
- Before recruiting participants, verify from the exact X4 model, firmware, and acquisition interface whether a hardware frame trigger or shared clock is supported. Do not assume the mmWave trigger interfaces in the cited papers are available on X4.
- Record a monotonic acquisition timestamp and frame counter for every radar frame and every reference frame. For three radars, retain a common trigger/counter if supported. Otherwise, record the measured offset and drift for every device and sequence; reject sequences whose independently measured residual alignment error exceeds half of the radar frame period (25 ms at 20 Hz). This threshold follows the temporal sampling interval; it is not a result from the cited studies.
- Save the untouched radar capture, exact sensor configuration/firmware, per-frame timestamps, dropped-frame flags, and all deterministic preprocessing outputs. Never infer synchronization from gesture labels or from a classifier’s predictions.

### 2. External truth and spatial registration

- Use calibrated stereo/optical motion capture to measure a hand or wrist reference trajectory. If the intended claim concerns finger articulation, wrist/hand-centroid truth is insufficient; use a validated finger-level marker/vision arrangement or narrow the claim to gross hand motion.
- Calibrate camera/reference coordinates to each radar coordinate frame with a rigid target detectable by both modalities. Keep the calibration samples and fit separate from the gesture evaluation split. Report held-out calibration residuals per radar.
- Record gesture class, repetition ID, participant, session, placement, orientation, range, start/stop cue timestamps, and quality/occlusion flags. Randomize gesture order; include rest intervals and failed/ambiguous trials rather than silently discarding them.
- Restrict access to identifiable video; publish derived pose tracks and consented radar records where licensing/consent permits.

### 3. Evaluation roles and measurements

- Separate engineering pilot data from the final evaluation. Determine participant count from a prospective power calculation using participant-level variability; do not treat frames or repeated gestures as independent participants.
- Lock development/validation participants and a final, untouched participant cohort before fitting. Within development, tune only with participant-disjoint validation. Report participant-held-out performance separately from session/placement-held-out performance; do not mix them into one headline number.
- Freeze preprocessing, sensor selection, calibration, model selection, and thresholds before opening final test labels. Keep the final cohort and any new placement/session condition unavailable to development.
- For gesture recognition, report macro-F1 and macro accuracy by participant, class, radar view, session, and placement, with participant-level intervals. For a trajectory task, report 3D hand/wrist position error, radial-velocity agreement per radar, and synchronization/calibration residuals. Do not substitute a recognition score for physical trajectory accuracy.
- Compare one-view inputs against the predeclared three-view system using identical participant/session splits and training budget. Include a simple per-view baseline and the strongest reproducible baseline available; do not repackage the 2026 fusion architecture as novelty.

## Pre-training stop gates

1. **Hardware gate:** exact sensor models and firmware support a timestamp/trigger path whose measured residual meets the frozen tolerance.
2. **Truth gate:** calibration and tracking quality are quantified before any training; hand/finger truth matches the intended scientific claim.
3. **Novelty gate:** primary-paper review shows the exact physical question/metric is not already answered for IR-UWB hand gestures; generic pose supervision and view fusion are presumed known.
4. **Data gate:** the participant/session/placement split, primary endpoint, model-selection rule, and stopping criteria are preregistered before fitting.
5. **Pilot gate:** a label-isolated, development-only pilot demonstrates measurable signal-to-truth alignment above a fixed null/control. If it fails, stop before a large training queue.

## Current blocker

The cloud contains the previously verified static UWB-Gestures restore, but it has no attached X4 acquisition interface, synchronized optical-motion-capture stream, or new participants/session/placement data. The existing arrays cannot be retrofitted with external pose ground truth. Therefore this protocol can be archived now, but collection, the physical-observability test, and any new training cannot honestly be marked started. Continue literature checking and protocol refinement until a new acquisition package is available; do not restart the fixed-cache sweep.
