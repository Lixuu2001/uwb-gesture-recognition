# 2026-10-08 — online onset candidate is not testable on current UWB package

## Result

A targeted literature/full-text review found Seifi et al. (2025), who add frame-energy onset detection and background-class classification to MIRA. This preempts generic radar gesture-onset detection as a novelty claim. The work uses 60-GHz FMCW recordings; its result is not a direct UWB benchmark.

The UWB-Gestures publisher description states that each 90 slow-time values form a single gesture sample. The restored cache therefore supports segmented-window classification, not onset latency or false-positive evaluation in a continuous background stream. The existing event-extent study is not onset ground truth and its fixed crop screen failed. No rerun is warranted.

## Runtime and data checks

- Remote branch at start: `25d80225fe60abe91b7a72e9a0de3cb12e6b6b3a`.
- Restored data remain at `/workspace/uwb-restored-20260930`; no data were downloaded or read for this review.
- Current cloud status reports no attached device capabilities; direct process/device inspection found no active UWB training process and no X4/optical capture device nodes.
- No labels, radar arrays, or scores were accessed; no model was fitted or trained.

## Disposition

Do not use prefix classification on the isolated 90-frame clips to claim online onset detection. Reconsider a UWB onset study only after a new continuous stream package includes background intervals and synchronized onset labels, and after a UWB-specific prior-art review. The overall contribution goal remains unmet.
