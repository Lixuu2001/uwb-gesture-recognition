# Bounded prior-art review: compact and speed-robust UWB gesture models — 2026-10-09

## Scope and execution decision

This was a bounded literature review of methods using the same UWB-Gestures corpus, focused on whether the existing small-CNN result or a speed/multi-scale variant could support a distinct method claim. No local labels, radar arrays, checkpoints, or predictions were read; no fitting or training was run. The restored data receipt remains verified and the runtime has no active training process, no GPU, and enforced restricted shell egress.

## Directly relevant published work

1. Qiao et al., “Gesture-ProxylessNAS: A Lightweight Network for Mid-Air Gesture Recognition Based on UWB Radar,” IEEE JSTARS (2023), DOI [10.1109/JSTARS.2023.3274830](https://doi.org/10.1109/JSTARS.2023.3274830). The accessible author manuscript describes LBP texture encoding, a ShuffleNet-based backbone, and a ProxylessNAS-style search/compression method on UWB-Gestures, reporting 96.52%. Its experiment section describes a random 20% validation sample from the image training set; the reviewed text does not establish a participant-disjoint split. Thus its number is not directly comparable with the project's eight held-user folds, but “a lightweight CNN for UWB gestures” is already prior art.

2. Khan and Kwon, “Radar-Based Hand Gesture Recognition with Feature Fusion Using Robust CNN-LSTM and Attention Architecture,” IEEE Access (2025), DOI [10.1109/ACCESS.2025.3558293](https://doi.org/10.1109/ACCESS.2025.3558293). The authors' institutional publication record says it evaluates the same 12-class, eight-participant, three-X4 UWB-Gestures corpus and reports 98.33% with early feature fusion; the abstract also lists Gaussian noise, signal inversion, and multipath tests. Only the institutional abstract/metadata was reviewed here, not the full paper or its split details. This closes generic early feature fusion/noise-robustness claims, not a protocol-matched score comparison.

3. Kim et al., “Robust UWB Radar Gesture Recognition Addressing Speed-Induced Scale Variations via Multi-Scale Feature Extraction,” IEEE Access (2026), DOI [10.1109/ACCESS.2026.3671243](https://doi.org/10.1109/ACCESS.2026.3671243). The author manuscript and DGIST institutional record describe a multi-scale first-layer range-Doppler block for speed-induced time-frequency scale variation, evaluated on UWB-Gestures at 98.12% with runtime analysis. The project already records its speed-scale novelty gate as failed; this review confirms that another generic multi-scale/speed-robust variant should not be launched.

## Decision

Do not present the existing 265,836-parameter CNN, a generic lightweight architecture, ordinary feature fusion, synthetic noise robustness, or a generic speed/multi-scale block as a new method. The cited papers use incompletely specified or non-matching evaluation protocols in the sources reviewed, so their headline accuracy values must not be ranked against the project's eight user-held-out folds. The prior art does not prove that the existing compact CNN is ineffective; it means its size/accuracy result alone is not a novelty-cleared contribution.

No distinct candidate clears both novelty and execution gates in this review. Do not start another training grid. Continue only if a materially distinct, literature-reviewed question is testable under a frozen participant-disjoint protocol, or if a new synchronized cohort and required capture hardware become available. Literature search was bounded, not exhaustive.
