# H-Inception full audit engineering repair; no scientific result yet

Original audit55822 exited1 after9,453.333773313905 wall seconds and
9,452.627575 child CPU seconds, without timeout. Its traceback reaches the
final_prediction_state comparison after the full training/prediction replay.
This is evidence of the reached program location, not a completed audit
receipt or an accuracy result. Training100epochs/7400updates was not rerun,
and the diagnostic score SUMMARY has not been parsed for this repair.

Read-only diagnosis found saved last and final checkpoints have identical
optimizer/metrics/all RNGs/callbacks/modes/identity/cursor; selected best
weights are their intended model-weight difference. Mirroring native model
saving changes no recorded state. The auditor's subsequent separate native
checkpoint loads consume only Python random state, while preserving the
audited model/optimizer/metrics and other RNGs. Those constructor calls are
absent from producer inference. Restoring the inspection RNG and applying
the selected weights matches the entire saved final checkpoint even without
opening real inputs/labels or doing a new real forward/update.

The repaired auditor wraps only native checkpoint inspection in a context
that preserves Python RNG. It then demands the entire before/after snapshot
be exact. It does not restore model variables or other RNGs to mask changes.
All original assertions and state-comparison calls, including the final
state check, remain present and unchanged. No tolerance, gate, epoch, seed,
model, preprocessing, sample selection or producer code changes. Original
1423 scientific/dependency hashes and19 trained artifact hashes remain exact.

The synthetic native reload regression reproduces the Python RNG side effect,
passes full-state equality with isolation and still rejects an injected
optimizer mutation. Four driver/static guard tests pass on temporary metadata:
exact registration required before moving files; changed evidence refused;
existing completion/archive/live worker prevents a duplicate; and every
original assertion/compare call is retained. No actual UWB fitting or score
selection occurs in these checks.

After exact remote readback/force=false publication of this repair, a separate
registration marker authorizes one manual re-audit of the original data under
the same14,400-wall/14,390-soft/14,400-hard CPU bounds. The driver preserves the
failed canonical supervisor directory as audit_attempt01, verifies its log
hash, then creates a fresh canonical audit_supervisor01. Original audit code,
logs and failure receipts are retained; the old postprocess-incomplete state
describes attempt01. The new driver's STATE.json is the current repair state.
There are no automatic retries, training restarts, new cost policies or next
method configurations. A second independently bounded audit is additional
engineering compute, explicitly separate from the failed first attempt.

If every full numerical check passes and the new canonical TERM is exit0,
the driver builds the complete local model archive. The adapted builder
retains all original backup checks and adds failed-attempt logs plus repair
code/evidence to the payload. Canonical study/COMPLETE_AUDIT.json identifies
the repaired auditor/helper and original failed termination. The complete
archive remains compatible with the existing all-member raw-readback verifier
and future canonical prior-closure guards. Original first failure is not
silently turned into success or discarded.

The driver cannot upload Drive or publish Git. Agent continuation must verify
owner-only metadata, upload/read back at most3 chunks concurrently using
download_raw_file=true/include_base64=false, immediately materialize/check
size/SHA, reassemble and verify every member. Only then release audited scores
and apply the unchanged original gate. Pass requires separate source
confirmation; failure closes only the exact registered CPU100 transfer.

No claimed accuracy improvement, new contribution or independent test result.
Historically developed all8 evidence stays exploratory; users7–8 are excluded
from new fitting/selection. Prepared Disjoint/MLSTM methods stay unselected
until this repair/audit/private-preservation phase closes. Goal unresolved;
existing continuation stays enabled. Original789/arrays/maps/actual ID values/
models/runtime binaries are excluded from this repair's Git changes.
