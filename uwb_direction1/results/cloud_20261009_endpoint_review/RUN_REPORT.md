# Pantomime endpoint prior-art review — 2026-10-09

## Scope and execution

This was a read-only literature/metadata review following the latest branch state. It did not open UWB labels or arrays, download/extract Pantomime archives, fit a model, or start training. The current executor remains CPU-only with restricted shell egress; no experiment process was found in the current continuation check. The full UWB source snapshot and restored data are already present.

## Dataset DOI and license evidence

- Aalto's institutional Pantomime dataset entry lists CC-BY-4.0 and DOI 10.5281/zenodo.3772659.
- Aalto's related dataset listings pair that DOI with the Zenodo URL/record 4459969; Research.fi's institutional catalog also lists both identifiers and marks the dataset CC BY 4.0. The current Zenodo 4459969 record has the same named creators and dataset/source description, but its visible license field is blank.
- Zenodo record 3772660 is a separate earlier “Pantomime dataset” record and its “newer version” route resolves to 4459969. Do not confuse 3772660 with Aalto's 3772659 DOI.
- This is substantially better institutional evidence that the Pantomime dataset is CC BY 4.0 and that 4459969 is the intended current dataset/source record. The exact file-level manifest/license association still has not been independently checked: Zenodo archive preview did not yield the member list, and no archive was downloaded or extracted. The license claim should retain this provenance caveat; no source-code license is inferred from the dataset license.

## Prior-art findings

1. Feng, Cheng & Duan, “mmWave Radar-Based Unsupervised Gesture Recognition via Image-Aligned Heterogeneous Domain Transfer,” IEEE TMC 25(3), 2026, DOI 10.1109/TMC.2025.3614353. The full author-hosted paper was reviewed. It transfers open image gesture data to unlabeled FMCW/mmWave gesture signals through trajectory-image construction and adversarial/contrastive/self-training alignment; it evaluates digit/alphabet gestures and includes cross-user and cross-environment protocols. This preempts a broad “heterogeneous gesture transfer” or generic image-to-radar adaptation claim. It does not evaluate X4 IR-UWB CIR against Pantomime point clouds or the proposed two-family endpoint.
2. Xu et al., “GesturePrint: Enabling User Identification for mmWave-Based Gesture Recognition Systems,” ICDCS 2024, DOI 10.1109/ICDCS60910.2024.00103. Its paper evaluates Pantomime, mHomeGes, and mTransSee as separate gesture-recognition benchmarks with 8:2, five-fold evaluation. It studies gesture-based identity too. It does not report training on UWB-Gestures and transferring to Pantomime; its use of Pantomime alone is not a direct match for that cross-dataset endpoint.
3. Pantomime's paper describes 21 self-defined gestures with 41 participants and studies radar point-cloud recognition. UWB-Gestures describes 12 dynamic gestures from eight participants with X4 IR-UWB. A map from UWB “swipe” / circular gestures to Pantomime semantic families is therefore a metadata-level label relation only. It is not evidence that the datasets contain kinematically equivalent executions or that family aggregation is scientifically meaningful.
4. A bounded targeted search for UWB-Gestures/X4-to-Pantomime-point-cloud family transfer returned no exact paper. This is not exhaustive novelty clearance. Generic transfer, adaptation, self-training, and sensor heterogeneity are already well covered.

## Decision

The exact cross-dataset endpoint is still unevaluated, but the proposed coarse swipe/circular mapping by itself is too generic to count as a method or contribution. No distinct, prior-art-reviewed algorithm and no fair frozen protocol have cleared the gates. Do not start transfer training, and do not present the mapping or a baseline result as novelty.

A future candidate must state a substantive research question beyond “can these broad labels transfer?”, establish semantic validity without using target test labels, and have a frozen participant-disjoint evaluation with an attribution-controlled baseline. If no meaningful target-label correspondence can be justified from dataset documentation, abandon this endpoint rather than infer a match from class names.

## Primary sources

- Aalto Pantomime dataset record: https://research.aalto.fi/en/datasets/pantomime-dataset-and-source/
- Research.fi institutional metadata: https://research.fi/en/results/dataset/c4cf1d4e-4527-4919-921e-9bd8fbf3eb2d
- Current Zenodo dataset/source record: https://zenodo.org/records/4459969
- Earlier Zenodo Pantomime dataset record: https://zenodo.org/records/3772660
- Palipana et al., Pantomime paper: https://orbilu.uni.lu/bitstream/10993/46834/1/pantomime.pdf
- Ahmed et al., UWB-Gestures data descriptor: https://www.nature.com/articles/s41597-021-00876-0
- Feng et al., full paper: https://duanch.github.io/papers/2026-mmwave-radar-based-unsupervised-gesture-recognition-via-image-aligned-heterogeneous-domain-transfer.pdf
- Xu et al., GesturePrint full paper: https://chaojiegu.github.io/assets/pdf/GesturePrint_ICDCS24.pdf
