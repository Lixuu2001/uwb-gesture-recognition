# Fixed radar-exchange inference: source gate failed

Registered before real diagnostic inference at `0eb4e8088e658afd72be3466dad8c7099f8a52aa`, protocol SHA256 `dc906aba5a19f6a7ead4d8e2ce1dd57282d7a36e1c692997e93c45a5665f2ebb`. Six source folds reuse users3–8 (four training, next source user validation, held source user diagnostic); all users have historical development exposure. This is not a new independent test or an eightfold result. Inference completed October4; archival finalization resumed October7 after execution-session handles were lost. Existing completed artifacts were recovered, not regenerated; no duplicate training was started.

| Frozen inference arm | Mean user macro accuracy |
|---|---:|
| Original onehot ridge |93.063131%|
| Semantic equivariant average (primary) |92.352273%|
| Label-invariant average (negative control) |76.327020%|
| Label-mapped swapped branch alone |84.839646%|

Primary vs ridge: **−0.710859 percentage points**, two of six wins; worst paired drop1.416667pp. This fails both the required+.5pp mean gain/four wins and maximum1pp paired decline. Comparisons to invariant and mapped-swap controls pass, but the joint gate fails. No eightfold expansion, mixing-weight/permutation/seed/normalization sweep or replacement of the primary by a control. Failure concerns this fixed inference projection, not all possible equivariant training.

The L/R radar slots are exchanged, Top retained, and labels0↔1,4↔6,5↔7,8↔9 exchanged (2/3/10/11 fixed). Scores are averaged with exactly.5 weight, with no softmax, calibration or new fit. The baseline uses the existing validation-selected onehot ridge on each frozen source epoch24 averaged encoder; the same weights and selection are used for every arm. Source validation and diagnostic outputs are fully saved before evaluation labels are opened. No query labels, query graph, class-balance constraint or new parameter selection enter prediction. Original quality exclusions remain.

Cohen & Welling, ICML2016 *Group Equivariant Convolutional Networks*, sections2–3 support the distinction between input and output group actions and acknowledge prior group averaging. This is a known finite-group projection, not a new G-CNN or validated original contribution. Detailed reading scope, publisher PDF commit/SHA and geometric assumptions are in `research/RADAR_EXCHANGE_REVIEW.md`. The algebraic equivariance of averaged scores does not establish that radar/hand scattering actually obeys the assumed mirror symmetry. The weaker swapped branch could reflect hardware/data asymmetry and/or a model not trained for that action; this experiment cannot distinguish those causes. The strong failure of the label-invariant control is consistent with the need for a nontrivial output action, but does not rescue the accuracy claim.

CPU timing used64 source-training clips only:0.212489s, estimated143.430s including3x margin, under600s budget. Sixfold inference took23.178422s, without encoder training. Independent functional convolution/GN/GELU replay on both actual original and exchanged inputs checked14,200 clip appearances (28,400 forward evaluations), all128dim hidden values and output scores exactly;39 artifact hashes, split IDs, raw/prepared hashes, inherited selection/readout hashes, semantic action and gate were checked. Existing archived original hidden features differed by at most3.814697e−6 for a changed batching context, within preregistered atol/rtol2e−5 and with all ridge argmax unchanged. Audit-only extra readout SHA checks were added after initial run; scientific code/protocol were unchanged. Final `AUDIT.json` identifies the verified outcome; `AUDIT_IDENTITY.json` pins audit code and its log.

Per-user values and machine-readable gate are in `PER_USER.csv` / `SUMMARY.json`. Large hidden arrays, complete scores and sample IDs remain in private backup, not Git. The source checkpoint and raw/prepared recovery chains are preserved in prior backup receipts. The previously established **eightfold** stronger ridge94.897727% remains the relevant comparator for future same-protocol eightfold studies; it must not be compared as though it were this six-source93.063131% protocol.

Backup verification is recorded separately in `DRIVE_BACKUP_RECEIPT.json`. The archived report is captured before that final receipt; latest Git report/receipt is authoritative. Research goal remains unmet. No background training or next experiment queue is running.

Final backup verified:3 private parts, 21,727,798 compressed bytes, 92 members; metadata owner-only, authenticated redownload, part/bundle/member SHA256 all passed. No permission changes.
