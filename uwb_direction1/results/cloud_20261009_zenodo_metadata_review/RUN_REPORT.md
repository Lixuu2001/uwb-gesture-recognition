# Zenodo metadata follow-up — 2026-10-09

## Scope
Read-only metadata/prior-art follow-up after the 2026-10-09 endpoint review. No UWB labels/arrays were opened, no archive was downloaded or extracted, and no model/training was started. Existing restored dataset and the full 789-file original project snapshot remain untouched.

## New source observation
The browser-readable current Zenodo record 4459969 identifies “Pantomime dataset and source,” marks the dataset Open, names the same four creators, describes the two archives as the dataset and source files for the Pantomime paper, and lists:
- `dataset_5_11_2020.zip`, 107.6 MB (MD5 shown by record: `766f9b3bb4838589ad5a6f918833d005`)
- `source_files.zip`, 937.5 kB (MD5 shown by record: `4bf7eb78bdaa68771ef26a94cbfc353b`)
- DOI `10.5281/zenodo.4459969`

The same record's visible Rights → License field is blank. Aalto University's research portal lists CC-BY-4.0 for the Pantomime dataset and DOI `10.5281/zenodo.3772659`, and links to an open Zenodo record. These records strongly support that the named dataset is intended for open reuse, but do not establish a file-by-file license mapping from the 3772659 catalog record to both ZIPs on 4459969. In particular, the dataset license must not be imputed to the separate source archive. The Zenodo REST API and file-member manifest were not accessible in this read-only browser check.

## Decision
This resolves the visible record's identity and archive names, but not the exact archive manifest/license association or the scientific validity of cross-dataset gesture-family labels. The earlier decision stands: no training until a specific question, target-label-independent semantic justification, license/manifest evidence, and a frozen participant-disjoint protocol with matched baselines are all clear. This review does not validate a research contribution.

## Environment
The current shell still has restricted egress, CPU-only execution, no acquisition hardware, and no active training process. The restored dataset is already present at `/workspace/uwb-restored-20260930`; do not repeat migration or ask the user to transfer those files.

## Sources
- Zenodo record 4459969: https://zenodo.org/records/4459969
- Aalto Pantomime dataset record: https://research.aalto.fi/en/datasets/pantomime-dataset-and-source/
- Prior-art and endpoint analysis: `research/PANTOMIME_ENDPOINT_PRIOR_ART_20261009.md`
- Previous report: `results/cloud_20261009_endpoint_review/RUN_REPORT.md`
