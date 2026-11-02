# Pantomime endpoint prior-art review — 2026-10-09

## Dataset identity and rights

The Aalto institutional dataset record names Pantomime, lists CC-BY-4.0, and cites DOI 10.5281/zenodo.3772659. Aalto's related listings and Research.fi metadata pair that DOI with Zenodo record/URL 4459969; Research.fi also labels the dataset CC BY 4.0. Zenodo 4459969 has the same creators and dataset/source description, though its visible Rights → License field is blank. The old Zenodo 3772660 record is a separate earlier “Pantomime dataset” record whose newer-version link resolves to 4459969; it should not be conflated with Aalto's 3772659 DOI.

This strengthens the institutional license/release linkage, but the archive's file manifest and license-to-file association have not been independently checked. The Zenodo preview did not expose the member list; no dataset/source archive was downloaded or extracted. The dataset license does not determine the source-code license.

## Endpoint-specific prior art

- Feng, Cheng & Duan (IEEE TMC 2026, DOI 10.1109/TMC.2025.3614353) transfer open image gestures to unlabeled FMCW/mmWave gesture signals by representing radar motion as trajectory images and using heterogeneous adversarial/contrastive alignment plus target self-training. Their full author-hosted paper reports digit/alphabet gesture tasks and in-domain, cross-environment, and cross-user tests. This preempts broad heterogeneous gesture transfer, image-to-radar transfer, and generic self-training claims. It does not report X4 IR-UWB CIR → Pantomime point-cloud transfer.
- Xu et al. (ICDCS 2024, DOI 10.1109/ICDCS60910.2024.00103) evaluates GesturePrint on Pantomime, mHomeGes and mTransSee as separate benchmark datasets with an 8:2 split and five-fold CV. It studies gesture recognition and user identification, not transfer from UWB-Gestures into Pantomime.
- Pantomime is documented as 21 self-defined gestures with 41 participants; UWB-Gestures is 12 dynamic gestures from eight participants using XeThru X4 IR-UWB. Name-level mapping of “swipe” and circular motions to broad families does not establish matching kinematics. The map is a possible label aggregation rule only.
- A bounded targeted search did not identify a paper reporting the exact X4/UWB-Gestures-to-Pantomime point-cloud family-transfer endpoint. This is not exhaustive novelty clearance.

## Decision

Cross-dataset transfer remains an unevaluated research endpoint, not a cleared method. Coarse family-name mapping alone is too generic to claim as a contribution. Do not train until a question beyond broad-label transfer is justified, semantic mapping is established independently of test labels, exact release/manifest is verified, and a participant-disjoint protocol plus attribution-controlled baseline are frozen. If documentation cannot justify the class correspondence, abandon this endpoint rather than assume equivalence.

## Sources

- Aalto Pantomime dataset portal: https://research.aalto.fi/en/datasets/pantomime-dataset-and-source/
- Research.fi metadata: https://research.fi/en/results/dataset/c4cf1d4e-4527-4919-921e-9bd8fbf3eb2d
- Zenodo 4459969: https://zenodo.org/records/4459969
- Zenodo older 3772660 record: https://zenodo.org/records/3772660
- Pantomime paper: https://orbilu.uni.lu/bitstream/10993/46834/1/pantomime.pdf
- UWB-Gestures descriptor: https://www.nature.com/articles/s41597-021-00876-0
- Feng et al. full paper: https://duanch.github.io/papers/2026-mmwave-radar-based-unsupervised-gesture-recognition-via-image-aligned-heterogeneous-domain.pdf
- GesturePrint full paper: https://chaojiegu.github.io/assets/pdf/GesturePrint_ICDCS24.pdf
