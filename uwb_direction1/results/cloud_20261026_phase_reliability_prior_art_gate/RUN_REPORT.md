# Phase-reliability weighting candidate: bounded prior-art gate

**Date:** 2026-10-26 21:19:04 UTC  
**Parent before update:** 5fc5ee479cc278e8b3a1cefe1a707e24e8755c1f  
**Decision:** do not start a model experiment for generic local phase-coherence/reliability weighting. This closes only the generic candidate as currently specified; it does not show that all reliability-aware methods fail.

## Candidate and direct relation to this corpus

Candidate: locally weight the existing phase-increment channel by a short-window phase-coherence statistic, then compare amplitude+phase against amplitude+phase+coherence-weight on the same held-user protocol.

The current historical feature already builds an analytic signal along fast time, computes adjacent-frame complex products, and weights their principal phase increments by echo magnitude before concatenating with amplitude. A second confidence weight could therefore mostly duplicate weak-echo attenuation. A low local phase concentration can also occur during a real gesture reversal or multipath-scatterer transition, so suppressing it could erase gesture structure rather than only remove noise. The available processed maps do not include independent synchronized kinematic truth to adjudicate that distinction.

## Bounded primary-source review

- Peng & Li, *The Journal of Engineering* (2019), DOI [10.1049/joe.2019.0434](https://doi.org/10.1049/joe.2019.0434), study phase noise effects on micro-Doppler extraction and define a signal-to-clutter prominence measure for motion-feature extraction. This is direct prior art for phase-noise affecting motion features and signal-quality-dependent feature reliability, though not this exact CIR feature or gesture benchmark.
- *Phase Noise Effects on Micro-Doppler Extraction for Radar Target Classification Using Machine Learning*, IEEE APMC 2024, DOI [10.1109/APMC60911.2024.10867536](https://doi.org/10.1109/APMC60911.2024.10867536), tests phase-noise effects on extracted micro-Doppler signatures and classification. The source available here is the publisher abstract/metadata, not full-text review.
- Qorvo's 2026 UWB sensing white paper describes phase coherence across successive CIR measurements as a way to detect subtle motion. It is an industry source, not an independent peer-reviewed validation.
- Circular-variance/coherence measures are established radar signal-quality and clutter/noise tools outside gesture recognition. Their reuse as a reliability weight is not itself a new method.

The search is targeted and not exhaustive. It finds no cleared narrow contribution from simply appending a local coherence weight. The current signal's existing amplitude weighting further weakens the case for a generic extra weight.

## Experimental gate and non-claims

No array values or labels were read; no statistic was computed; no fitting, training, synthetic perturbation or model evaluation ran. This is a prior-art and observability gate only. Environment remains Python/NumPy/SciPy/scikit-learn on five CPUs, without PyTorch; the restored clutter-removed tensor and original 789-file source snapshot are present; no UWB job or serialized checkpoint was found.

Do not claim local coherence weighting as novel, effective, or physically validated. A future materially distinct candidate would first need to specify a reliability quantity not reducible to the existing echo-amplitude weight, a way to distinguish true gesture reversals from unreliable phase using currently available inputs, and a frozen same-protocol control. Without that, keep the historical phase-motion result as exploratory empirical evidence only and leave the contribution goal unresolved.

## Relevant literature links

- [Peng & Li (2019), IET Journal of Engineering](https://doi.org/10.1049/joe.2019.0434)
- [IEEE APMC 2024 phase-noise study](https://doi.org/10.1109/APMC60911.2024.10867536)
- [Qorvo UWB radar sensing white paper (2026)](https://www.qorvo.com/-/media/files/qorvopublic/white-papers/uwb-radar-sensing-white-paper.pdf)
