# IR-UWB gesture representation prior-art gate

Date: 2026-10-11 (UTC)

## Scope and source review

This was a bounded primary-source review of IR-UWB gesture representations, not an exhaustive novelty search. It used accessible primary papers and publisher metadata/abstract where publisher full text was blocked.

- Ren et al. (2017), *Hand-Based Gesture Recognition for Vehicular Applications Using IR-UWB Radar*, uses magnitude-histogram statistics, time-of-arrival (TOA) variation and reflected-signal frequency; its TOA estimate uses a waveform center-of-mass method. This means TOA/range variation or centroid tracking alone is not a cleared contribution. [Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5422194/)
- Ahmed et al. (2020), *Hand Gesture Recognition Using an IR-UWB Radar with an Inception Module-Based Classifier*, represents signals as 3-D image patterns and applies an Inception-style classifier. It reports XeThru X4 hardware, eight gestures and three participants, but uses its own collection and a sample-level train/test split, not this project's eight held-user folds. Thus generic range-time image conversion/CNN/Inception is prior art; its accuracy is not comparable to the current protocol. [Primary paper](https://www.mdpi.com/1424-8220/20/2/564)
- Wang et al. (2022), *Hand Gesture Recognition Using IR-UWB Radar with Spiking Neural Networks*, describes level-crossing sampling/time-event coding followed by an SNN on an IR-UWB gesture dataset. The accessible IEEE record is metadata/abstract only; the exact cohort relation and full protocol were not verified. Generic event encoding or SNN use is therefore not cleared as novel, and its reported score is not used for comparison. [IEEE record](https://ieeexplore.ieee.org/document/9870013/)
- The UWB-Gestures dataset paper defines fast time as range and slow time as gesture duration. [Dataset paper](https://www.nature.com/articles/s41597-021-00876-0)

## Decision

This review closes standalone novelty claims based only on TOA/centroid variation, generic range-time image classification, or threshold/event-coded SNN classification. It does not establish that every possible signal-level IR-UWB method is known.

The current restored cache is processed, clutter-removed range-time data and does not provide the raw complex echoes needed to validate phase/coherent waveform mechanisms. No independent calibrated motion labels are available for physical kinematics. Consequently, no candidate from this screen has both a distinct mechanism and a valid evaluation path. Do not start a classifier grid or repackage a known representation. Continue only after identifying a more specific radar-signal mechanism with primary-source novelty review and a validation protocol supportable by the current data; otherwise the missing evidence requires calibrated acquisition/annotation.

## Execution and data handling

The cloud workspace remains CPU-only with restricted egress and about 712 MB free; the restored dataset exists and process inspection found no active training job. This literature-only review downloaded no data and read no labels, radar arrays, or checkpoints. No fitting/training ran; no checkpoint or source snapshot changed.

This is a scoped prior-art gate, not a validated research contribution.
