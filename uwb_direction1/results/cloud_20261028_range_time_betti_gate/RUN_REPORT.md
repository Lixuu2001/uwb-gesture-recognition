# 2026-10-28 — Range-time Betti Stage A execution receipt

## Outcome

**INCOMPLETE — stopped before features or fitting.** The frozen preflight count assertion detected a mismatch. This cannot be interpreted as a mechanism failure or scientific negative.

| Source participant | Retained count observed | Required by preregistration |
|---|---:|---:|
| 1 | 1200 | 1200 |
| 2 | 1000 | 1000 |
| 3 | 1200 | 1200 |
| 4 | 1200 | 1200 |
| 5 | 1200 | 1200 |
| 6 | 1100 | 1107 |
| **Total** | **6900** | **6907** |

## Access and compute receipt

- One attempt, frozen Stage A protocol SHA-256: `04668d3cf2a16514132e88a60dd43c0a1be031acb6da9dd5a1085e2af6ac6290`.
- Runner SHA-256: `ddd58ac798cba14a01d458882c5ec9e30714c25e27c1f5313c8fdf5a6d3350ae`.
- The runner opened array headers/shape metadata, selected participants 1–6 by participant IDs, and read only their gesture labels to apply the preregistered quality mask.
- It read **no map pixel values**, extracted 0 feature records, fit 0 models, and emitted no predictions or scores.
- Participants 7–8 gesture labels were not indexed or read. No test/audit fitting, threshold selection, parameter search, or CNN occurred.
- The assertion failed before the fixed 900 CPU-second and 900 wall-second limits were relevant; no timeout occurred.
- Local logs and failure receipt are in the cloud workspace at `/workspace/uwb-range-time-topology-20261028/results/`. Raw arrays and model artifacts were not uploaded or committed.

## Interpretation and next gate

The registered run is incomplete because its source data/quality-mask counts do not match the preregistration. It neither supports nor rejects the proposed range-time topology descriptor. Do not alter the mask, expected counts, features, or gate after observing this mismatch and do not rerun this preregistration. First reconcile the discrepancy from source-only provenance; any later analysis requires a new frozen protocol. The method-contribution goal remains unvalidated.


# Amendment — run02 count-field correction (2026-10-28 UTC)

The first registered runner stopped before feature extraction or fitting because its preflight expected 1,107 retained rows for participant 6, while the frozen quality mask returned 1,100. Source-only review of the archived original `experiments/p1_20260905/AUDIT.md` confirms HV6 has 1,100 retained records after excluding its 93 quality-flagged G4 records; the archive also states HV2 has 1,000 and the other participants have 1,200. Thus the source cache's 1,100 is consistent with the original audit; 1,107 was a preregistration transcription error.

The prior attempt remains preserved as incomplete (0 feature records, 0 fits, no map pixels or scores; users 7–8 gesture labels unread). Under the user's earlier authorization for one repair-and-retry when a run stops before scores, the separate run02 preregistration changes only participant 6's expected retained count from 1,107 to 1,100 and the source total from 6,907 to 6,900. The quality exclusions, participant split, features, controls, classifier, weights, decision thresholds, views, and user-label boundary are unchanged. Run02 has a new protocol/code hash and output directory; it has not yet started.


# Run03 engineering amendment — 2026-10-28 UTC

Run02 stopped before pixel/feature/model access because the corrected per-participant assertion still contained a stale literal total of 6,907. The archived source audit supports 6,900 retained source rows (HV6=1,100). Run01 and run02 logs are retained, with 0 feature records, 0 fits, and no scores; no audit-user labels were read. User-authorized pre-score engineering repairs permit this third invocation. Run03 changes only the remaining total-count guard. To keep aggregate work within 900 seconds, the process has an 885-second CPU soft stop, 895-second CPU hard ceiling, and independent 895-second wall limit; the two earlier preflight exits were each under two seconds wall. New run03 protocol/code hashes are recorded in `PREREGISTRATION_RUN03.md`. The descriptor, splits, fixed quality mask, classifiers, controls, decision thresholds, and all science remain unchanged.


## 2026-10-28 run03 result

The source-only range-time Betti feasibility screen completed but failed its preregistered GO criteria: Betti − amplitude overall −2.716 pp; Betti − fixed spatial shuffle +5.605 pp; 0/6 positive Left folds, 3/6 Top, 3/6 Right; Left/Top also failed the per-view non-inferiority threshold. All 54 fitted model artifacts and the feature artifact were hash-verified. Only participants 1–6 labels were used; users 7–8 gesture labels were not read. The terminal SystemExit(0) wrapper bug was corrected in metadata from saved outputs only, without rerunning. Exact candidate is closed, no Stage B; overall method-contribution goal remains unresolved. See results/cloud_20261028_range_time_betti_gate/RUN03_REPORT.md.
