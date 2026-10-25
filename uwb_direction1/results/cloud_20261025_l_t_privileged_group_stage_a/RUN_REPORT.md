# L/T-derived within-class strata — run02 reauthorized after pre-data error

**Current state:** run01 ended in an engineering error before `np.load`; it produced no scientific statistic and is preserved. The owner has now explicitly authorized one same-protocol retry. This does not add a seed, a model, a threshold, or a second statistical analysis. The second attempt uses a new immutable output directory. Cumulative real-analysis resource use is capped below the original 900 CPU-second / 900 wall-second budget.

## Duplicate triplet scope

The prior exact-duplicate audit and quality audit map all 100 identical L/T/R triplets to participant ID 2 = sealed validation HV2, in separately named `HV_02_G11` and `HV_02_G12` MAT inputs. Those gesture labels conflict and the pre-existing quality rule excludes them. Since participant 2/HV2 is outside source users HV3-HV8, these known pairs do **not directly contaminate** the current source inputs. Their provenance remains unknown: repeat acquisition versus upstream copy. Metadata review lists no exact source-user duplicates among HV3-HV8. Before model fitting, run02 hashes each retained source row's complete three-view triplet and blocks on any alias; no post-hoc removal/relabeling is allowed.

## Frozen descriptive protocol and boundaries

Trial/session exchangeability remains unresolved. The 1,999 seeded count-preserving row shuffles are descriptive references only. No p-value, significance claim, inferential null, or p-value-based gate is computed or used. The source split, quality mask, normalization/block pooling, within-class PCA/K=2 clusters, Right Ridge probe, error/CMI definitions, coverage and directional thresholds, and seed remain unchanged.

A positive result is only `descriptive_screen_supports_further_design`. It is not a scientific GO, mechanism validation, contribution, or independent generalization evidence. A non-supportive screen is limited to this frozen representation/probe and is not a general mechanism failure. HV7/HV8 class labels are post-fit diagnostic conditioning variables, not deployment inputs; these users have prior project-development use and are not independent validation. The binary Right K=2 control is finite and does not represent all Right-view information. No CNN, grid, extra seed, or data acquisition is authorized.

## run01 receipt and correction

The preserved run01 traceback says `restore_provenance()` required transfer-manifest state `verified`; actual verified transfer manifest state is `complete`. The companion restore receipt was independently verified and its manifest SHA matched. The exception occurred before `np.load`, before source aliases, and before any labels or radar values were accessed. run01 used 0.966 CPU seconds / 1.003 wall seconds, exit code 1; it is engineering-incomplete, not a scientific negative.

The code now accepts `complete` and `verified` transfer-manifest terminal states, still requiring a `verified` receipt and matching manifest SHA. A synthetic test exercises the actual `complete` state. run01 is retained unchanged under `/workspace/uwb-stage-a-descriptive-20261025/run01/` and in the repository's run01 receipts.

## run02 cumulative budget and command

Prior usage: 0.9661069999999999 CPU seconds and 1.0033835080103017 wall seconds. run02 hard limits: **899 process CPU seconds** and **898.9 seconds wall time**, keeping cumulative run01 + run02 under 900 seconds. Numerical pools remain at one thread. Any abort/exception is incomplete and is not a scientific negative. Only one retry has been explicitly authorized. If run02 fails before source arrays are read, code-only repair may be considered within the remaining cumulative budget; once any source data are read, do not restart or resample, only recover/complete the same computation.

```bash
python diagnostic.py --confirm-stage-a \
  --cpu-seconds 899 --wall-seconds 898.9 \
  --output-dir /workspace/uwb-stage-a-descriptive-20261025/run02
```

## run02 pre-run hashes

The following hashes identify the retry's exact source version; the execution manifest will repeat them.

| File | SHA-256 |
|---|---|
| `diagnostic.py` | `984743fe8e2de4526fa78df8c79c49b95515130f9aed81a24ed46ad90e2896d0` |
| `budget_exec.py` | `2585b65d905d08d47d398f449d05f7261ee0b2c2622daef7a378fea1ed355566` |
| `protocol.json` (run02 authorization) | `303d9322fca6d7a51c61fbbf5abff17ef09ec35b243bf907fe0619ff237220bd` |
| `tests/test_budget_and_protocol.py` | `a90289c04a9c0116ec24881b28ca44d8933b05a440145a24a93fa34c13c8f5a8` |

The current fixed runner passes 15/15 synthetic tests; compile and JSON validation pass. run01 hashes and logs remain separately recorded and unchanged. No source arrays or labels are stored in Git.

## run02 execution and results

The reauthorized attempt is ready to launch once in `run02`; the exact protocol snapshot is `/workspace/uwb-stage-a-descriptive-20261025/run02_prelaunch_protocol.json`. Cumulative cap is below 900/900 after charging the first pre-array failure. Do not launch any third attempt, CNN, or follow-on experiment from a completed descriptive screen.
