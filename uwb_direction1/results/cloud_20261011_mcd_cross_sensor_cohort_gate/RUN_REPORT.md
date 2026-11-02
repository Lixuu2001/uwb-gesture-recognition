# MCD-Gesture cross-sensor cohort and method gate

Date: 2026-10-11 13:27 UTC

## New cohort lead

The public MCD-Gesture repository describes an independent 77 GHz TI AWR1843/DCA1000 dataset: 24,050 sequences from 25 people, six predefined gestures (push, pull, left/right slide, clockwise/counterclockwise turn), seven negative actions, six rooms, and five radar positions. It includes raw ADC and preprocessed dynamic range-angle-image (DRAI) sequences. The repo says dataset and code access require signing its agreement and emailing the signed copy; the archive was not fetched. Primary sources: [dataset and access terms](https://github.com/leeyadong/cross_domain_gesture_dataset), [DI-Gesture preprint](https://arxiv.org/abs/2111.06195), [published TMC paper](https://doi.org/10.1109/TMC.2022.3207570), and [authors' code-access page](https://github.com/leeyadong/Radar-Gesture).

The MCD gesture names have possible semantic overlap with X4 classes (directional swipes, rotations, and push-like motion), but there is no verified shared label ontology, acquisition instruction equivalence, or event correspondence. Treat it as a candidate independent cross-sensor cohort, not a matched external test yet. Its single 77 GHz MIMO FMCW sensor and DRAI representation differ from X4's three-view IR-UWB, clutter-removed range-time maps.

## Prior-art gate

DI-Gesture explicitly targets domain-independent, real-time mmWave gesture recognition across new users, environments, and locations. Its reported design derives spatial-temporal signal variation, uses signal/gesture-correlation-based augmentation, and segments gestures for real-time inference. Therefore, generic cross-user/environment/location generalization, radar-signal augmentation, and spatial-temporal segmentation are not cleared as standalone novelty for this project. This paper does not establish that the exact X4-to-AWR1843 cross-radar-modality transfer task has been solved.

A bounded search surfaced a 2026 MCD-based multi-scale/contrastive domain-generalization paper at abstract/metadata level, but its full text was not reviewed; no detailed claim is made from it. A cross-radar-modality transfer paper exists for radar heart-rate monitoring, a different task, so cross-modality transfer as a broad concept is not itself new. The exact hand-gesture IR-UWB-to-FMCW transfer task and a distinct method remain unresolved.

## Decision and required gates

MCD is a promising *independent cohort lead* that corrects the previous search's conclusion that no separate gesture cohort was identified. It is not freely downloadable under the repository's stated process. Do not sign the agreement, contact the authors, download data, or train in this continuation: none of those external access actions has been authorized, and the method/label/protocol gates are still open.

Before any experiment, establish (1) authorized dataset access, (2) a verified common-class map from gesture instructions rather than name similarity, (3) a frozen cross-user and cross-sensor evaluation split that excludes overlapping users/events, and (4) a radar-specific method that survives method-level prior-art review. Generic DG/augmentation, classifier swaps, and evaluation splits alone will not meet the contribution goal. No labels, arrays, or checkpoints were read. No MCD data were downloaded. No project fitting/training ran. The restored X4 cache remains present; no active training process was found. The contribution goal remains unvalidated.

Scope: public repository README and public DI-Gesture preprint/publisher metadata were reviewed on 2026-10-11. This is a bounded screen, not an exhaustive literature search or a completed full-text review of every cited work.
