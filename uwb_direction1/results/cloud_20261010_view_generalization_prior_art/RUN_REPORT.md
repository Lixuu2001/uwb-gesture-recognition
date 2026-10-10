# Held-view generalization literature gate

**Date:** 2026-10-10  
**Branch parent:** d5e8fc3cb5451095f0ac141292d15fe11f4459d5  
**Decision:** do not fit/train a new model or claim a method contribution from leave-one-radar-view-out evaluation alone.

## Primary-source evidence reviewed

1. The UWB-Gestures dataset descriptor states that the corpus contains three independently operated monostatic X4 radars at left, top, and right positions, with separate view files and 90 slow-time rows per 4.5-second gesture record. It does not document external motion ground truth or a synchronization/calibration protocol. This supports treating the view as a discrete acquisition domain for a classification study, but does not support coherent signal alignment or a calibrated hand trajectory. [Dataset descriptor, Scientific Data (2021)](https://www.nature.com/articles/s41597-021-00876-0)

2. Xue et al., *Neuromorphic-inspired multi-view global-local fusion for IR-UWB radar dynamic gesture recognition* (Frontiers in Neuroscience, 19 June 2026), is an exact-corpus prior. Its full text reports a subject-independent 6-subject train / 2-subject test protocol and evaluates left, right, top, all dual-view combinations, and all three views as input configurations (Table 6). It reports three-view fusion and claims view complementarity. The reported protocol varies the views presented to each model; the paper does not describe training on two views and testing on an unseen third view. Its numbers are not comparable to this project's eight-fold held-user protocol. [Full paper](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1864448/full)

3. Yang et al., *Sparsity aware dynamic gesture recognition using radar sensors with angular diversity* (IET Radar, Sonar & Navigation, 2018), directly studies dual-radar angular diversity, feature fusion, observation aspect, and robustness when one sensor is poor. The sensor/task differ from the X4 corpus, but generic angular-diversity fusion or viewpoint robustness is established prior art. [Full paper](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-rsn.2018.5206)

4. A bibliographic search surfaced Chen et al., *Domain-Generalized Gesture Recognition via mmWave Radar Signal Multiview Learning*, DOI 10.1109/TIM.2025.3637962. The publisher full text was not accessible in this review; available metadata/abstract are not treated as a verified primary-source review and do not support an exact-protocol comparison. This remains a required literature check before any view-generalization method is considered novelty-cleared.

## What remains possible, and what it does not establish

A strict **held-subject + held-view** evaluation appears technically testable with the current three-view cache: hold out one of eight subjects and one of the three fixed radar views, train using only the other six subjects and the other two views, and keep all target-view data outside model/threshold/epoch selection. The 24 outer cells would test transfer to one of the three recorded placements, not arbitrary radar placement, new radar hardware, or physical 3-D motion recovery. Before using it, the sample-index correspondence across the separately stored view files must be confirmed from metadata/provenance; equal row counts alone are insufficient.

This protocol is an evaluation question, not an original method. Multi-view fusion, radar angular diversity, and generic domain generalization have prior art. No candidate algorithm has cleared novelty. Current corpus limits also preclude a ground-truth-based physical-invariance claim: the restored files contain clutter-removed maps but not synchronized external hand-motion labels, raw echoes, or sensor calibration.

## Gate outcome and execution record

- No labels or radar array values were read in this review.
- No model was fit, trained, or tuned; no test-fold information was used.
- No active research process was found; CUDA is unavailable; the verified restored directory remains present.
- No novelty-cleared method is identified. Do not launch a 24-cell experiment yet.
- Next: (a) review the full primary source for the 2025 mmWave multiview domain-generalization paper and exact radar-view OOD protocol, and (b) verify view/sample provenance using metadata only. If those gates leave a distinct mechanism, pre-register it and its source-only model-selection rules before any fit. Otherwise wait for calibrated independent radar systems and synchronized motion truth.
