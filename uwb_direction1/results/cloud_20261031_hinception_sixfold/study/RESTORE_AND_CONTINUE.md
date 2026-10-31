# Fixed source confirmation recovery

Restore this `study/` directory to
`/workspace/uwb-hinception-sixfold-20261031`. The frozen manifest uses exact
absolute paths for existing dependencies. Recover the audited H pilot and its
dependencies from the already committed pilot study and its private complete
Drive index under `results/cloud_20261031_hinception_pilot/complete_backup/`.
That seven-chunk archive includes every selected/last/initial/first/final model,
native checkpoints, sample IDs, prediction archives and the preserved original
audit failure/repair. Verify the archive and every member. Original input data
remain private and separately indexed; prefer existing SHA-verified files.
Never upload original arrays or actual sample-ID values to Git.

This directory reuses pinned source/runtime/launch helpers; their bytes are
bound by FROZEN_HASHES.json. The native dependencies are Keras3.11.3/TorchCPU
at the previously registered paths. Wheels/runtime binaries are excluded from
Git. The original pilot's dependency restoration records describe that runtime.
Check all frozen hashes before any model forward or source value/label access.

After remote byte-exact preregistration readback, REMOTE_PREREGISTRATION.json
records the registration commit, protocol/frozen hashes and exact blob checks.
It must have state `remote_source_confirmation_exact_readback_verified`.
Only then launch `python -u controller.py --run` once in this directory.
First inspect STATE.json, jobs/, run01/, locks and OS processes: existing work
must never trigger duplicate launches. Controller refuses an existing jobs/run01.
Two isolated supervisors execute held2..6, each train then full audit. Every
subprocess has a wall/CPU limit. No automatic retry/resume. On failure retain all
evidence, diagnose and separately register any necessary engineering repair.
Do not change scientific settings or use intermediate diagnostic scores.

The producer/auditor never index that fold's diagnostic labels. The aggregate
scores only after allfive independent full replay audits pass and both archives
per fold freeze. All logged sample IDs and checkpoints stay local/private.
After completion, build and privately raw-readback-verify the entire new
modelset including per-job termination logs/audits and the held1 recovery
reference before publishing final results. STATE alone is not proof of private
preservation. A partial insurance archive is not complete-modelset recovery.
No current fold claim is independent validation; allsource evidence exploratory.
