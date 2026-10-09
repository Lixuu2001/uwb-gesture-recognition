# Cross-sensor radar gesture prior-art update (2026-10-09)

## Finding

A closer prior-art check found a 2026 peer-reviewed IEEE Transactions on Radar Systems paper by Sukianto et al., “Unsupervised Federated Learning With Harmonized Foundation Models for FMCW Radar-Based Hand Gesture Recognition,” DOI [10.1109/TRS.2026.3654128](https://doi.org/10.1109/TRS.2026.3654128), published 2026-01-14. The author institution's publication record describes signal-level range/Doppler harmonization across radar configurations, radar-specific antenna-channel/gesture augmentation, transformer pretraining, few-label target fine-tuning, distillation, and unlabeled federated refinement across Infineon and TI FMCW data. This makes generic cross-device FMCW harmonization, adaptation, foundation-model transfer, and federated refinement unavailable as standalone novelty claims. The source record is an author-institution abstract, not a full-text review.

This paper does **not** establish IR-UWB/X4 ↔ FMCW gesture transfer: its stated experiments use FMCW devices. A distinct cross-modality dynamic-hand-gesture endpoint is not ruled out by this source alone. But it remains a design question, not a contribution: the existing prior-art record includes DI-Gesture's cross-user/environment/location mmWave study and Jung et al.'s 2023 UWB↔FMCW transfer for digit recognition. No frozen common-class endpoint or validated, directly comparable dataset pair has yet been established.

## Data and protocol implications

UWB-Gestures is 12 dynamic gesture classes from eight participants, with three independently placed monostatic X4 radars saved separately. Pantomime is a different FMCW representation (sparse 3D point-cloud sequences), with about 22,000 instances, 41 participants and 21 classes; the original paper studies environmental, speed, angle and distance variation. The datasets may contain semantically related directional gestures, but that is not yet a verified sample-level label mapping. Any common-label transfer benchmark would need a metadata-only label map, fixed source/target direction, participant-disjoint splits, and access/license verification before downloading or scoring. Its result would be a new cross-dataset endpoint, not the native 12-class UWB score.

## Decision

No training, data download, label/array read, or fitting was performed. The current cloud remains CPU-only with no configured capture capability, secrets, or outbound identities. Local process inspection found no active research process or acquisition device; previous runner PIDs were defunct. The restored UWB package and 789-file original source snapshot remain present and untouched.

Do not start transfer training or request restricted data until the class mapping, split, direct-prior-art review, and data-access path are frozen. The overall research goal remains unvalidated.

## Primary sources checked

- [JKU author-institution record and abstract for Sukianto et al. (2026)](https://research.jku.at/en/publications/unsupervised-federated-learning-with-harmonized-foundation-models/)
- [UWB-Gestures dataset paper](https://www.nature.com/articles/s41597-021-00876-0)
- [Pantomime paper record](https://research.aalto.fi/en/publications/pantomime-mid-air-gesture-recognition-with-sparse-millimeter-wave/)
- [Pantomime dataset record](https://zenodo.org/records/4459969)
- [DI-Gesture authors' dataset/code repository](https://github.com/leeyadong/cross_domain_gesture_dataset)
- [Jung et al. (2023), DOI 10.1109/JSEN.2023.3289232](https://doi.org/10.1109/JSEN.2023.3289232)
