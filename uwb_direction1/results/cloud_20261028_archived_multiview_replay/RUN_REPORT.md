# Archived eight-fold multi-view prediction replay audit

**Date:** 2026-10-28 UTC  
**State:** complete as an artifact-integrity/metric replay; no novel method validated.  
**Scope:** existing `p1_20260928/multiview_s2026` saved outputs; no model fit, tuning, checkpoint read, or feature-matrix access.

## Result

- Replayed all 8 held-user folds and all 7 saved sensor-subset heads. Structural and metric checks passed: **136/136**.
- Every fold’s saved test-accuracy vector matched a fresh argmax replay from saved logits and the archived test labels (absolute tolerance `1e-12`). Validation logits matched the corresponding rows bitwise. Split IDs, role codes, subject assignments, and sample uniqueness were checked.
- No head was selected after inspecting outcomes; all seven are reported below. Test labels were accessed only for fixed metric calculation, never for fitting or tuning.

| Fixed view head | Mean held-user class-macro accuracy |
|---|---:|
| L | 68.1534% |
| T | 58.1085% |
| R | 71.6765% |
| L+T | 87.4932% |
| L+R | 84.1066% |
| T+R | 90.5833% |
| L+T+R | 93.4417% |

### Full three-view head by held user

| Test subject | Class-macro accuracy |
|---:|---:|
| 1 | 96.2500% |
| 2 | 98.7000% |
| 3 | 91.2500% |
| 4 | 96.7500% |
| 5 | 98.1667% |
| 6 | 75.0000% |
| 7 | 96.4167% |
| 8 | 95.0000% |

The fixed full-view head averages **93.4417%**, reproducing the historical saved-prediction figure at the metric level. This is a replay of one existing archive, not fresh training, an independent cohort, or independent validation; these users have prior project-development exposure.

## Engineering attempts and reproducibility artifacts

- `RUN01_AUDIT.json` preserves an initial incomplete audit caused by an erroneous fixed per-split row-count assumption; quality masking makes counts vary by held user. The same archive was replayed after correcting the check to compare against each saved split, not after changing data or metrics.
- `RUN02_AUDIT.json` preserves the 120-check pass before making shape/unique-ID checks explicit in the counter.
- Final `AUDIT.json` records 136/136 checks, aggregate and per-user metrics, split sample counts, and SHA-256 values for all 24 input files (8 folds × split/result/prediction archive).
- Final replay wall time: 0.277 s. No training process was launched.
- `PROTOCOL.json`, `audit_archived_outputs.py`, and `PRE_RUN_AND_INPUT_MANIFEST.json` preserve the fixed audit scope and code/input hashes. No row-level IDs or arrays were copied into Git.

## Interpretation and limits

This audit establishes consistency of the archived predictions, split metadata, and reported metrics. It does not validate an original research method, establish independent-user generalization, compare against a newly fitted baseline, or resolve literature novelty. It does not satisfy the remaining method-contribution goal. No new training is justified by this replay alone.
