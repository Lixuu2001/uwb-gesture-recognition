# Pantomime data-rights and semantic-overlap follow-up — 2026-10-09

## Scope
Read-only follow-up on the gated X4-UWB → Pantomime cross-sensor candidate. We checked institutional dataset metadata, the linked Zenodo record, the Pantomime paper summary and its public class-vocabulary descriptions, plus the UWB-Gestures data descriptor. No archive was downloaded, no source ZIP extracted, and no local radar labels/arrays were read. No model was trained.

## Rights evidence
Aalto University's official research portal lists “Pantomime dataset and source” under DOI 10.5281/zenodo.3772659, states Dataset Licence CC-BY-4.0, and links its “Access Dataset” button to Zenodo record 4459969. Thus there is positive institutional-catalog evidence that the data archive is reusable under CC-BY-4.0. The Zenodo record itself marks the item Open and lists the dataset and source ZIP files, but its visible Rights/License field is blank. We will use no source-code archive unless its separate software rights are independently clear. No file was fetched, so no checksum or archive-member manifest was inspected.

## Semantic scope
The Pantomime publication describes sparse mmWave point-cloud gesture recognition and reports 21 gestures from 41 participants (~22,000 instances), with conditions spanning environments, distance, angle and articulation speed. The UWB-Gestures descriptor defines eight directional swipes, clockwise/counterclockwise rotations, inward push, and empty gesture, with three independent X4 views. Public descriptions expose plausible name-level overlap for directional swipes and circular motions, but they do not establish matched kinematic instructions, handedness, axes, amplitude, or execution timing across the two collection protocols. Therefore no target-label mapping is approved from names alone.

## Novelty and execution decision
The exact X4-UWB → Pantomime endpoint has not appeared in the bounded targeted search recorded previously, but broad heterogeneous radar/gesture transfer is already prior art (see `research/PANTOMIME_ENDPOINT_PRIOR_ART_20261009.md`). This is not novelty clearance. A defensible next step would require (1) a published or independently auditable movement-level mapping for the overlapping classes, (2) a narrowly stated sensor-physics question distinct from generic transfer, (3) a frozen participant-disjoint source/target protocol, and (4) a primary-source prior-art review of that exact mechanism.

The current managed executor reports restricted HTTP egress with no Zenodo host allowlisted and no attached device/GPU capability; the shell policy lists no custom allowed hosts. Browser access to metadata does not provide a materialized dataset file. Do not bypass the proxy or ask for a manual upload. Until the supported environment policy grants Zenodo access and the scientific gates above pass, do not download or train.

## Sources
- Aalto institutional catalog: https://research.aalto.fi/en/datasets/pantomime-dataset-and-source/
- Linked Zenodo record (the Aalto access link redirects here): https://zenodo.org/records/4459969
- Pantomime author overview/paper reference: https://dariush-salami.github.io/bibliography/palipana2021pantomime.html
- UWB-Gestures data descriptor: https://www.nature.com/articles/s41597-021-00876-0
- Prior endpoint review: `results/cloud_20261009_endpoint_review/RUN_REPORT.md`
