# Gesture-onset / early-recognition feasibility — 2026-10-08

## Decision

Do not treat online gesture-onset detection as a validated next experiment on the restored UWB-Gestures package. The original dataset is stored as gesture-aligned samples: the public descriptor states that every group of 90 slow-time values is one gesture sample. The restored package does not contain a continuous radar stream with pre-gesture background or independently annotated onset times. Prefix classification on these isolated windows would not measure real-time onset detection or false alarms in background.

Generic radar-based onset detection is also already published. Seifi et al. (2025) extend MIRA with frame-energy onset detection, background-class handling, and post-onset classification; their data are 60-GHz FMCW recordings with approximately 100 frames and gesture/background intervals. This is a direct prior for the broad “detect onset, then classify” claim, although it is not the same UWB sensor or dataset. A UWB-specific method could still be distinct, but this package cannot establish it without continuous UWB streams and onset/background truth.

## Primary sources reviewed

- Seifi, Ott, Carbonelli, Servadei & Wille, “Interpretable hand gesture recognition: a rule-based framework for radar-based gesture onset detection and classification,” *International Journal of Microwave and Wireless Technologies* 17(9), 1629–1641 (2025), DOI [10.1017/S1759078725102535](https://doi.org/10.1017/S1759078725102535). Full publisher PDF reviewed: [Cambridge Core PDF](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/16D84F4CD2AE152FEFAC7ED84CF06B9F/S1759078725102535a.pdf/interpretable_hand_gesture_recognition_a_rulebased_framework_for_radarbased_gesture_onset_detection_and_classification.pdf).
- Ahmed et al., “UWB-gestures, a public dataset of dynamic hand gestures acquired using impulse radar sensors,” *Scientific Data* 8, 102 (2021), DOI [10.1038/s41597-021-00876-0](https://doi.org/10.1038/s41597-021-00876-0). The article describes 9,600 gesture samples from eight participants and says each group of 90 slow-time values is one sample: [publisher article](https://www.nature.com/articles/s41597-021-00876-0).

## Project boundary

- The current 90-frame sample windows permit offline classification experiments; they do not contain the preceding continuous background needed to estimate onset latency, false activations, or real stream segmentation.
- Existing temporal-extent screening already rejected its fixed energy-preserving crop premise and explicitly warns that its proxies do not equal onset labels. Do not relabel that result as an onset experiment.
- The prior MIRA results use a different 60-GHz FMCW system and data; no score comparison is valid. The prior establishes that the generic onset idea is not new, not that its method transfers to UWB.
- Feasible next measurement requires continuously captured raw UWB streams spanning background and gestures, synchronized onset annotations (or a defensible external annotation procedure), participant-separated evaluation, and an exact acquisition setup. Cloud runtime has no X4 acquisition or synchronized annotation interface, and no new such package is present.

No gesture labels or radar arrays were read for this literature/feasibility review. No fitting, model selection, or training was run. The research contribution goal remains unvalidated.
