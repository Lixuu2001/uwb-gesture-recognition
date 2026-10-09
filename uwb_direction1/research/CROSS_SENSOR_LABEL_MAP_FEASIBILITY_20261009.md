# Cross-sensor gesture label-map feasibility (2026-10-09)

## Metadata-only family map

A label-level two-family endpoint can be specified from the published dictionaries, correcting the earlier statement that no coarse common-label map had been established:

| UWB-Gestures labels | Pantomime labels | Frozen family |
|---|---|---|
| Eight directional swipe classes (LR, RL, UD, DU and four diagonals) | Swipe left; swipe right | swipe |
| Clockwise rotation; counterclockwise rotation | Circle clockwise; circle counter-clockwise | circular |

Exclude UWB inward-push and empty/no-gesture, plus Pantomime push, pull, and every other action, from this first candidate. In particular, do not map UWB “inward push” to Pantomime “push”: the current label wording does not verify equivalent hand path, body-relative direction, or execution. The proposed endpoint is therefore a closed-set two-family task, not the native 12- or 21-class task and not online gesture/background detection.

This mapping is a semantic aggregation of names/descriptions, not proof of sample-level kinematic equivalence. UWB uses 90 slow-time rows over 4.5 seconds and independently stored monostatic X4 views; Pantomime uses sparse 3D FMCW point-cloud sequences and has a different participant pool. Sensor geometry, framing, handedness/orientation, sequence duration, preprocessing, and class counts still need to be audited from dataset metadata before any score. Any evaluation must split participants, freeze source/target direction and any calibration budget before opening target labels, and report both directions separately. A target model cannot simply consume one modality's raw representation without a defined cross-modal representation bridge.

## Novelty boundary

The label map itself is not a method or novelty claim. The UWB dataset paper explicitly motivates gestures as common motion families including swiping and cyclic rotation. Generic cross-modality gesture sensing/adaptation is also prior art: CM-ASAP (2024) adapts RGB/depth sensing on EgoGesture, while Sukianto et al. (2026) covers cross-device FMCW harmonization/adaptation and Jung et al. (2023) covers UWB↔FMCW transfer for digit recognition. A bounded search did not identify a direct IR-UWB/X4↔FMCW point-cloud dynamic hand-gesture transfer study, but that is not an exhaustive novelty clearance.

The only defensible next research question is whether a predeclared, modality-appropriate representation or adaptation protocol can transfer the two action families across these genuinely different radar domains under participant-disjoint evaluation, and where/why it fails. Even that endpoint is a benchmark/design lead until direct prior-art review, licensing/access, sample availability, and a validated protocol are complete. It cannot be presented as an original effective method based on metadata alone.

## Access and execution gate

Zenodo publicly lists the Pantomime archive (dataset ZIP about 107.6 MB plus source ZIP), but this cloud runtime currently has restricted egress and no configured capabilities, secrets, or outbound identities. No external archive was downloaded; no external sample metadata, UWB labels, or radar arrays were read in this pass. The exact archive license and internal class-count/file manifest must be confirmed before use. No fitting/training was run. There is no active research process or acquisition device; the UWB restore and 789-file original-source snapshot remain present and untouched.

## Sources

- UWB-Gestures dataset article (gesture vocabulary, X4 views, sample timing): https://www.nature.com/articles/s41597-021-00876-0
- Pantomime primary paper PDF (Figure 7 gesture labels and point-cloud representation): https://orbilu.uni.lu/bitstream/10993/46834/1/pantomime.pdf
- Pantomime author-institution record: https://research.aalto.fi/en/publications/pantomime-mid-air-gesture-recognition-with-sparse-millimeter-wave/
- Pantomime public archive record: https://zenodo.org/records/4459969
- CM-ASAP (RGB/depth adaptive cross-modality gesture sensing): https://arbabianlab.stanford.edu/cross-modality-adaptive-sensing-and-perception
- Sukianto et al. (2026), DOI 10.1109/TRS.2026.3654128: https://doi.org/10.1109/TRS.2026.3654128
- Jung et al. (2023), DOI 10.1109/JSEN.2023.3289232: https://doi.org/10.1109/JSEN.2023.3289232
- DI-Gesture authors' repository: https://github.com/leeyadong/cross_domain_gesture_dataset


## 2026-10-09T07:31Z Additional cross-modal prior art

A primary author-institution record for Hazra et al. (SAM 2022, DOI 10.1109/SAM53842.2022.9827785) describes gesture cross-learning from camera point clouds to 60-GHz FMCW radar point clouds using DGCNN/Bi-LSTM. This is closer than generic RGB/depth sensing and preempts a broad claim that cross-modal point-cloud gesture learning is new. It still does not use IR-UWB/X4 CIR or test transfer between independently collected UWB and FMCW radar cohorts. Treat the exact UWB-to-FMCW radar-to-radar endpoint as unvalidated and continue the bounded direct-prior-art review; do not infer novelty from the modality pair alone.
