# GAP source-sixfold completion audit preparation — 2026-10-30

The existing fixed experiment continues at `/workspace/uwb-gap-sixfold-20261030`: worker45683/supervisor45678, held2/validation3. No duplicate launch, scientific edit, new seed, held-score observation or result-gate change. All35original frozen entries still match. This update implements result-verification code; it does not claim a completed experiment or a real completed-result audit.

The independent `audit_complete.py` imports no training driver and refuses array/checkpoint decoding until STATE says complete/pidnull, allsixfolds and allfive new prediction freezes exist, every completion/hash agrees, and the independent training supervisor exited0 within43200wall/CPU seconds. The actual active run was presented to `audit_launch.py --run`: it rejected with `not_ready: GAP training not complete` before creating an audit supervisor/output. This is the expected guard check, not a failed real analysis.

When the guard passes, the auditor will verify all35original frozen hashes, code/input/reference hashes and all completed run artifacts; exact private IDs/roles/counts; all6900source records' signed-map RMS/reshape independently in NumPy; each GAP initial/first/best/last model, AdamW fast state/Lookaheadslow/count/reload, saved Torch/NumPy/Python RNG structures, BN counters; validation-only macro/BCE/earlier-epoch selection; all6900selected-validation forwards plus6900diagnostic logits across allsix folds, exact diagnostic argmax and fixed tolerances; allpaired scores, user means/wins/worst and the original gate. Reused held1 uses upstream `validation_bce` history schema, normalized in read-only memory. CNN reference hashes/metrics and membership in the prior independent full-forward audit are checked; CNN forwards are not repeated. No historical nonselected GAP epoch retraining or forward replay is claimed.

Numerics fixed before new diagnostic outcomes: preprocessing rtol2e-5/atol2e-6, logits/BCE rtol1e-4/atol1e-5, exact diagnostic argmax, macro/summary tolerance1e-12. Ten pinned reused input files and four audit-code/input-index hashes are recorded separately; original training FROZEN_HASHES remains unchanged.

Ten synthetic test methods pass with budget/failure/freeze/fold/tamper subcases, six disjoint role splits and overlap rejection, independent signed/zero preprocessing vsTorch, held1log-schema and selection ties, missing-class macro/gates, and fast/slow optimizer reload plus corrupt-state rejection. They use synthetic arrays/models and no UWB values or new diagnostic outcomes. Log `AUDIT_TESTS.log`; actual guard receipt `AUDIT_READINESS_CHECK.log` and metadata `AUDIT_PREPARATION.json`. These component and guard checks do not claim an end-to-end real sixfold audit has already passed.

After alltraining/predictions/supervisor requirements succeed, execute once:

```bash
python /workspace/uwb-gap-sixfold-20261030/audit_launch.py --run
```

The reused independently tested supervisor enforces one CPU thread,890seconds wall,880CPU soft/890hard. Inspect `audit_supervisor01/TERMINATION.json` and `COMPLETE_AUDIT.json`; failures are incomplete engineering, not an accuracy-negative result. Preserve attempts and repair only within authorized same-calculation accounting; no new fit/seed/resampling/diagnostic selection. No automatic controller was added. While training is active, do not repeatedly launch the completion audit.

After actual completion/audit, privately preserve allstates/IDs/logs and only then apply the unchanged mean>=1pp/atleast4wins/worst>=-2pp gate versus existing24epochlocalCNN under the same source-user roles. GAP64versusCNN24/inputpipeline differences and repeated historical development remain explicit. No users7–8labels, legacy8fold gain, independent validation or originality claim. Overall user goal remains unresolved.
