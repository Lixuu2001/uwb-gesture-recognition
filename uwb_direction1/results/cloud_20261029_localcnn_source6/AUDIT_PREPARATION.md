# Completed-run verification prepared while source-sixfold training continues

The original worker 43382 and supervisor 43381 remain alive. At this observation the first new trajectory, held user 2 / validation user 3, is at epoch 13 of 24. Held user 1 remains reused; no new diagnostic score is reported. Original scientific code, protocol and all 12 preregistered hashes are unchanged. No new training process, candidate, seed or tuning grid was started.

Added `audit_complete.py`, `audit_launch.py` and nine synthetic tests. The verifier imports no training runner and refuses array/checkpoint decoding until training is complete, the supervisor exits successfully, and every new fold's prediction and best-checkpoint freeze verifies. The real running-state guard was invoked once and correctly rejected the premature audit before array access or audit output-directory creation. This expected rejection is not a training failure. Synthetic tests also reject missing freezes, corrupt predictions, failed supervisors and role contamination; they check validation-only tie selection, present-class macro averaging and all three frozen gate conditions.

After completion, the read-only audit checks all input/source/manifest/code hashes; exact six-fold IDs and roles; initial model equality; first-update training IDs; finite model and optimizer states; optimizer step counts and reload compatibility; RNG and BN counters; validation-only epoch selection; selected-validation predictions and all diagnostic predictions by forward replay; fixed MiniRocket predictions; every reported metric and the original descriptive gate. It does not retrain historical nonselected epochs. Numerical replay uses fixed rtol 1e-4 / atol 1e-5 plus exact argmax equality. No fitted weights or choices change.

The audit has its own independent 1,790-second wall deadline and process CPU soft/hard limits of 1,780/1,790 seconds, one BLAS/PyTorch thread and a terminal receipt even after a hard kill, using the previously tested immutable supervisor. These are verification resources, separate from the unchanged 12-hour wall / 36-CPU-hour training limit. Audit errors or resource stops are incomplete engineering outcomes, never scientific negatives. No automatic retry is installed.

Preparation artifacts: `AUDIT_HASHES.json`, `AUDIT_PREPARATION.json`, `AUDIT_TESTS.log`, `AUDIT_READINESS_CHECK.log`. The nine tests passed. No real sample pixel/label values or model tensors were read by this preparation; no users 7–8 labels were used. The full real completion audit has not run and no success claim about it is made.

After the actual training supervisor finishes, use:

```bash
python /workspace/uwb-localcnn-sixfold-20261029/audit_launch.py --run
```

Then inspect `COMPLETE_AUDIT.json` and `audit_supervisor01/TERMINATION.json`, preserve the new model/optimizer/RNG/ID/prediction artifacts privately, and report the frozen gate. If training is dead or incomplete, inspect its own `supervisor01/TERMINATION.json` and retained state first; do not launch the audit or duplicate/restart training. TimeMIL remains prepared but unlaunched while this confirmation is active. All source results remain exploratory and do not establish old-eightfold improvement, independent validation or an original contribution.
