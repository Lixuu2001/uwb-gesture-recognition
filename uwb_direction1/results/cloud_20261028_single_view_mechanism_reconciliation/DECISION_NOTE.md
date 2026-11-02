# Current single-view signal-mechanism gate reconciliation

**Date:** 2026-10-28 UTC  
**Purpose:** reconcile prior candidate gates before considering another implementation. This is a read-only research decision note, not a new experiment.

## Current observable signal boundary

The restored cache contains real-valued clutter-removed range-time maps, metadata reports shape `[9593, 3, 90, 189]` and `float32`. The restore and source inventory report no raw complex echo tensor or per-view background/calibration files; the dataset does not provide synchronized hand trajectory truth. Thus a method claimed from this cache must be a physically defensible operation on those maps and cannot claim to recover calibrated 3-D hand motion or raw-I/Q phase.

## Relevant routes already screened

1. **Local echo/patch correspondence.** The frozen 2026-10-04 diagnostic found 0/6 source users passing the joint gate for each of Left, Top, and Right. It did not fit a classifier or measure accuracy. This closes only the registered local-envelope matching configuration; it does not show that every motion representation is ineffective. See `results/cloud_20261004_local_echo/RUN_REPORT.md` and `AUDIT.json`.
2. **Range-ridge direction/displacement plus echo energy or apparent hand area.** A prior IR-UWB hand-gesture paper already uses direction, distance/target-location change, and frontal-area/echo-energy change. The generic mechanism is not novel; no project experiment was run. See `results/cloud_20261012_range_ridge_mechanism_gate/RUN_REPORT.md`.
3. **Range-time trajectory extraction.** Same-corpus range-time/energy trajectory prior art and absent synchronized motion truth prevent a new physical-trajectory claim from generic ridge/centroid features. See `results/cloud_20261011_range_trajectory_prior_art/RUN_REPORT.md`.
4. **Range-resolved recurrence/RQA.** Radar self-similarity matrices already cover HRRP, micro-Doppler, and range-Doppler sequences; generic RQA is also prior art. No candidate classifier was trained. See `results/cloud_20261010_range_resolved_recurrence_gate/RUN_REPORT.md`.
5. **Temporal reversal with inverse gesture labels.** Gesture-recognition prior art already reverses frame order and maps left-swipe to right-swipe. Closed before experiment on 2026-10-28. See `results/cloud_20261028_temporal_reversal_label_gate/DECISION_NOTE.md`.
6. **Generic multi-view fusion or evidence fusion.** Same-corpus tri-view fusion prior art and the 2025 mmWave multiview domain-generalization paper preempt generic fusion claims; author-repository evidence is in `results/cloud_20261028_tim_author_repo_addendum/RUN_REPORT.md`.

## Decision

No candidate in this reconciliation clears both physical observability and a sufficiently distinct, literature-supported mechanism. Do not train a model or run another parameter grid. This is not evidence that all methods on the current cache are impossible or ineffective. It establishes that the obvious range, echo, recurrence, temporal-order, and generic multiview routes should not be repackaged.

Continue method-first only if a concrete, materially different mechanism can be stated from the current map values, separated from these prior routes, and paired with a frozen, leakage-safe user-held-out test. If no such mechanism appears, keep the existing saved-prediction replay as a reproducibility result and leave the original method-contribution goal explicitly unresolved. No new data request is made here.
