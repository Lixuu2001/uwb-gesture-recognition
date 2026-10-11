# Physically interpretable FMCW gesture augmentation prior-art review

**Reviewed:** 2026-10-11  
**Repository branch:** `codex/uwb-direction1-cloud-20260930`  
**Research decision:** no experiment authorized from this lead; generic physical augmentation is prior art and the current restored cache lacks the measurements needed to reproduce its condition transformations credibly.  
**Data use:** no UWB labels, array values, or checkpoints were read; no fitting or training was run.

## Primary paper reviewed

A. Fusco, Z. A. Zaman, S. Hazra, L. Servadei, and R. Wille, “Enhancing FMCW Radar Gesture Classification With Physically Interpretable Data Augmentation,” *IEEE Access*, vol. 13, pp. 60556–60569, 2025, DOI [10.1109/ACCESS.2025.3556565](https://doi.org/10.1109/ACCESS.2025.3556565). The public author-uploaded full text is marked CC BY 4.0 on [ResearchGate](https://www.researchgate.net/publication/390361151_Enhancing_FMCW_Radar_Gesture_Classification_with_Physically_Interpretable_Data_Augmentation). The publication record is also listed by [TUM](https://portal.fis.tum.de/en/publications/enhancing-fmcw-radar-gesture-classification-with-physically-inter-2/).

This is direct radar gesture-classification prior art, although it uses a different sensor and corpus: a 60 GHz Infineon BGT60TR13C (one TX, three RX), its own six-participant recordings, and measured hand-gesture time series. The five inputs are range, radial velocity, azimuth, elevation, and magnitude.

## Method-level overlap

The paper specifies and evaluates:
- time scaling by interpolation to vary gesture duration/speed;
- amplitude variation and Gaussian feature noise;
- feature-offset shifting;
- range transformation using observed target-distance statistics, including range offsets/troughs, magnitude attenuation, and non-gesture noise;
- angle transformation using target-angle azimuth-offset and magnitude statistics;
- combinations of time scaling, noise, and range transformation.

Its reported range experiment trains on recordings through 1 m and evaluates on real recordings through 1.2 m. The combined augmentation improves accuracy from 51.25% to 67.92% in that extended-range scenario. These are results on its own sensor, dataset, split, and task conditions, not a result comparable to the project's eight held-user folds.

A closely related earlier paper is Y. Hassab, T. Stadelmayer, and F. Lurz, “Physically-interpretable data augmentation for multi-range hand gesture recognition using FMCW radar time series,” *IEEE Transactions on Radar Systems*, vol. 1, pp. 571–582, 2023, DOI [10.1109/TRS.2023.3320869](https://doi.org/10.1109/TRS.2023.3320869). Its institutional record is available from [TUHH](https://tore.tuhh.de/entities/publication/f52147d1-54e9-4ca7-9573-81251c484cf9/). The 2025 paper itself cites that work for physically interpretable radar gesture augmentation.

## Consequence for this project

Do not claim novelty for generic radar-feature time scaling, Gaussian noise injection, range/angle feature shifts, attenuation scaling, target-condition-statistics augmentation, or their simple combination. The 2025 paper is close enough that a new claim would need to specify a materially different measurement mechanism and validation question, not just substitute the UWB corpus or classifier.

The restored UWB input is a clutter-removed (90\times183\times3) range-time representation. The recorded inventory does not contain raw echoes, per-radar empty-scene captures, calibrated source/antenna transfer functions, or independent range/pose labels. Consequently:
- speed/noise manipulation on cached map values would be a generic augmentation replication, not a new physical model;
- range or angle synthesis cannot be grounded in project-specific target-condition measurements from this cache;
- the 2025 paper’s target-range/target-angle statistics and 60 GHz hardware effects cannot be transferred as if calibrated for X4 IR-UWB.

No labels or tensor values were inspected to test these assumptions. This is a bounded prior-art and artifact-availability review, not a claim that every radar-augmentation paper has been exhaustively searched.

## Gate and next action

Status remains **no validated research contribution**. Keep augmentation experiments closed. Continue screening for a distinct mechanism that is both prior-art-clear and identifiable in the existing inputs. If a candidate requires calibrated range/pose or synchronized movement truth, defer it until those measurements exist; do not infer them from class labels or view order. Preserve the held-user protocol and fit any eventual model only within each training fold.

## Sources

- Fusco et al. 2025 full-text author upload: [ResearchGate publication page](https://www.researchgate.net/publication/390361151_Enhancing_FMCW_Radar_Gesture_Classification_with_Physically_Interpretable_Data_Augmentation).
- Fusco et al. 2025 DOI: [10.1109/ACCESS.2025.3556565](https://doi.org/10.1109/ACCESS.2025.3556565).
- Fusco et al. 2025 institutional record: [Technical University of Munich](https://portal.fis.tum.de/en/publications/enhancing-fmcw-radar-gesture-classification-with-physically-inter-2/).
- Hassab et al. 2023 DOI: [10.1109/TRS.2023.3320869](https://doi.org/10.1109/TRS.2023.3320869).
- Hassab et al. 2023 institutional record: [TUHH](https://tore.tuhh.de/entities/publication/f52147d1-54e9-4ca7-9573-81251c484cf9/).
