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
