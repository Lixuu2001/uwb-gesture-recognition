# Completed fixed source pilot: full fusion fails, local baseline eligible

Both24epoch CNN arms completed in16603.925wall/47173.445CPU seconds, within12hwall/36CPU-hour allocation.93independent checks pass, including full selected validation/diagnostic forward replay, optimizer/RNG/BN state, roles, frozen hashes and checkpoint selection. Full selected epoch17; local selected epoch23.

Train users3–6(4700), validation2(1000), diagnostic1(1200), fixed seed20261029. All source users have previous development history. Macro recall averages over the held user's present classes.

| Method | Held-source-user macro recall |
|---|---:|
| Paper-informed full MV-GLFNet-style network |96.0000%|
| Local ConvMixer CNN component |96.8333%|
| Existing MiniRocket with original fixed LSQR readout |85.3333%|

Full-minus-local=-0.8333pp; the full>=local+0.5pp criterion fails. Close this exact24epoch full-fusion configuration, without seed/LR/normalization/epoch rescue grids. This does not reject the untested100epoch paper setting or all multiview learning.

Local-minus-MiniRocket=+11.5000pp; the separately preregistered local baseline>=MiniRocket+1pp criterion passes. This licenses a separately registered source-sixfold baseline study only. It does not establish superiority to the old eightfold93.4417%, which has different training roles/protocol, or an original contribution.

See [SUMMARY.json](SUMMARY.json), [COMPLETE_AUDIT.json](COMPLETE_AUDIT.json), [LOCAL_ARM_COMPLETE.json](LOCAL_ARM_COMPLETE.json) and [LOCAL_ARM_EPOCHS.jsonl](LOCAL_ARM_EPOCHS.jsonl). No users7–8 labels used. No test-label fitting/checkpoint tuning. Diagnostic labels were indexed only after both predictions froze. The exact learned source-only models, IDs, optimizer states, predictions and logs are preserved privately with authenticated remote readback and all hashes verified; recovery index: [FINAL_QUEUE_DRIVE_BACKUP_RECEIPT.json](FINAL_QUEUE_DRIVE_BACKUP_RECEIPT.json).

Next: unchanged local baseline only, reuse audited held1 and train the five missing source folds under a new frozen protocol. Full fusion and fixed MultiRocket are not expanded after their failed gates. Accuracy-improvement and original-contribution goals remain unresolved.
