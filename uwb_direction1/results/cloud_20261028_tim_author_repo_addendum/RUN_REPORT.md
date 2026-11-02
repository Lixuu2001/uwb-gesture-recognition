# Addendum: author-repository evidence for the 2025 mmWave multiview DG paper

**Date:** 2026-10-28 UTC  
**Scope:** bounded literature-source check only. No project data, labels, logits, or weights accessed; no training or fitting.

## New source evidence

The authors' public code repository, Tkwer/MVRADAR, is available at https://github.com/Tkwer/MVRADAR. Its README identifies the paper *Domain-Generalized Gesture Recognition via mmWave Radar Signal Multi-View Learning* (Chen et al., IEEE Transactions on Instrumentation and Measurement, 2025, DOI 10.1109/TIM.2025.3637962) and documents multimodal input/configuration support: simple concatenation/addition and several attention/evidence fusion modes including Dempster-Shafer combination. The repository describes a 4D FMCW/mmWave gesture dataset and cross-domain evaluation.

An indexed author abstract states that the paper's method uses domain-knowledge-guided, evidence-theory-fused multiview learning, Dempster-Shafer rules, and causal feature/fusion components for OOD generalization across users/scenarios. The publisher full text was not obtained: IEEE access was previously blocked, the author repository has code/readme but no manuscript, and ResearchGate offers only a request-full-text page. No access control was bypassed and no author was contacted.

## Research impact

This evidence further closes generic multi-view fusion/evidence aggregation as a novelty route. It does not show that the exact algorithm is identical to this project's UWB-X4 task, and the paper uses FMCW/mmWave rather than the project's IR-UWB corpus. The available abstract and README are insufficient to compare equations or protocol details. The independent prior same-corpus tri-view paper (Xue et al. 2026) already preempts a generic multi-view performance claim on the same UWB-Gestures corpus.

This check does not identify a new single-view physically grounded mechanism. The current method goal remains unresolved. Continue only with a distinct mechanism observable in the existing clutter-removed maps; do not launch a multi-view experiment or claim novelty from a radar-modality port of the public code.

## Sources

- Author code repository / README: https://github.com/Tkwer/MVRADAR
- DOI record: https://doi.org/10.1109/TIM.2025.3637962
- Indexed author abstract and access status: https://www.researchgate.net/publication/398060763_Domain-Generalized_Gesture_Recognition_via_mmWave_Radar_Signal_Multi-View_Learning
- Same-corpus tri-view prior art: https://doi.org/10.3389/fnins.2026.1864448
