# Continuation — Stage A engineering/exchangeability review (2026-10-25)

No real diagnostic, model fit, or CNN training was run. Added a parent supervisor with Linux `RLIMIT_CPU` soft=hard, independent monotonic wall timeout, immediate process-group SIGKILL at deadline, single-thread BLAS environment, periodic memmap/null flushing, terminal-state preservation, model/runtime/hash manifests, and 12 passing synthetic unit tests.

Metadata review resolves the exact duplicates: 100 three-view-identical rows occur only in participant 2, across separately named G11/G12 MAT arrays with contradictory labels. Source provenance cannot distinguish acquisition repetition from upstream copy. Existing quality handling excludes both conflicting classes; these rows are outside the Stage-A source split, and no exact duplicate is listed for users 3–8.

However, the dataset metadata do not record trial order, timestamp, session, or repeat-block IDs within the 100 participant/class samples. The current record-level conditional permutation is therefore not justified under possible within-session dependence. `diagnostic.py` now blocks before reading arrays while `protocol.json` says `run_authorized: false`; this is inconclusive, not a scientific negative. Before launch, resolve using trustworthy capture metadata plus a frozen block-level null, authorize a pre-result descriptive-only design without p-values, or stop the candidate. Do not infer blocks from cache row indices.

No data-derived local artifacts were produced or uploaded. Tests were synthetic only. Outer users HV1/HV2 remain sealed; audit labels do not enter fitting, and group identity stays fit-anchored under audit-row permutation.

Interpretation remains narrow: Right is represented by one binary K=2 control, so any incremental CMI would be relative to that partition, not all Right-view information. HV7/HV8 labels are class-conditional audit variables, not deployment inputs or independent validation labels.
