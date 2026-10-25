# L/T-derived within-class strata — one-shot Stage A attempt incomplete

**Final state:** `failed_exception` before source-array access. The single user-authorized run was attempted once and stopped during restore-metadata preflight. This is an engineering error, not a scientific negative. No descriptive statistic was produced. The one-shot authorization is consumed; there is no rerun or CNN authorization in this report.

## Duplicate triplet scope

The prior exact-duplicate audit and quality audit map all 100 identical L/T/R triplets to participant ID 2 = sealed validation HV2, in the separately named `HV_02_G11` and `HV_02_G12` files. Those labels conflict and the pre-existing quality rule excludes them. Since participant 2/HV2 is outside source users HV3-HV8, these known pairs do **not directly contaminate** the current source inputs. They remain aliases across gesture files with unresolved provenance (repeated acquisition versus upstream copy). The frozen run code additionally contains a pre-fit byte-exact source-triplet scan for HV3-HV8; it was not reached in this attempt. Metadata review lists no exact source-user duplicates among HV3-HV8. No post-hoc exclusion was performed.

## Registered descriptive protocol

Before launch, the owner selected the descriptive-only path. Trial/session independence remains unknown, so 1,999 seeded count-preserving row shuffles are only a finite-randomization descriptive reference. The amended pre-run protocol computes no p-value, makes no significance claim, and uses no inferential threshold. The source split, quality rule, fixed normalization/block pooling, within-class PCA/K=2 clustering, Right Ridge probe, CMI definitions, coverage/error-direction thresholds, and random seed were otherwise frozen. A completed positive screen would only mean “may merit a separate validation design”; it could not establish mechanism, contribution, or independent generalization.

The exact protocol used at launch is preserved as `registered_protocol.json`; SHA-256 `2f3a8d613f44da59916cbc85cb9656c6bed57de8442732fe0b8269fafc9b69e0`. It was committed before execution at branch head `1999b8d2a8cafd893917df112cc3f8e3f7f7094a`. After the attempt, the checked-in protocol was marked consumed (`run_authorized=false`) to prevent an accidental second run. No scientific threshold was changed.

## Run and budget receipt

Command used once:

```bash
python diagnostic.py --confirm-stage-a \
  --output-dir /workspace/uwb-stage-a-descriptive-20261025/run01
```

The supervisor launched the worker with process CPU limit 900 s, independent wall timeout 900 s, and one-thread numerical pools. It stopped after 1.003 s wall / 0.966 CPU s with exit code 1. `run_status.json` says `failed_exception`, phase `worker_start`, no partial artifacts, `scientific_result_recorded=false`, and `is_scientific_negative=false`.

The traceback identifies the mismatch: `restore_provenance()` required `TRANSFER_MANIFEST.json` to have `state == "verified"`, but the actual manifest's valid terminal state is `"complete"`. Its companion `RESTORE_VERIFIED.json` was `verified`, its manifest SHA matched, and all three expected cache file sizes matched. The state assertion failed before the code reached `np.load`; no radar-array values or gesture labels were read, no source-alias scan ran, and no feature/model fit, shuffle, CMI, coverage, or Right-error result exists.

Postmortem fix accepts both `complete` and `verified` for the transfer manifest while still requiring a verified receipt and matching manifest SHA. A synthetic test covers the actual `complete` state. This fix is not followed by a second real run.

## Hashes and environment

Pre-run code hashes from the committed version:

| File | SHA-256 |
|---|---|
| `diagnostic.py` | `661fd578b966d672627a0f3a8cc6949650870533a746be7932abceb4f75696dc` |
| `budget_exec.py` | `2585b65d905d08d47d398f449d05f7261ee0b2c2622daef7a378fea1ed355566` |
| `protocol.json` used at launch | `2f3a8d613f44da59916cbc85cb9656c6bed57de8442732fe0b8269fafc9b69e0` |
| `tests/test_budget_and_protocol.py` | `27cc043e3131b7ac42b505750bd106f52d62c045fe2062317af943378507b104` |

Postmortem code fix (not used by the failed attempt): `diagnostic.py` SHA-256 `4bdfae4e8a17dd1a13bd61cf51bbba190fc51ba6b5a58b75b7377884c8ebb3bb`; consumed protocol SHA-256 `691543462cb19fa374aa602d4aa3ee1363925dafede1d6d1330f67437e7c33f9`; updated synthetic test SHA-256 `860b34154de8bc9aeace02a3fb32d186469fc57a46ff1a77f8ccc53138d580a9`. The fix was tested synthetically only.

Restoration metadata SHA-256: `10516baf0b49a84e6a39169ef447fffcdcb29defa2a6cc79e301dc938ddceb4d`. Expected cache hashes (from that verified transfer manifest): X `3004e7a2aa44cb4ffd763a6ef3bf30c85d2b7dc0499140c195c48fa3ed6ce519`; y `d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd`; subjects `5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480`. Those are manifest provenance values; the failed worker did not hash or load arrays.

Runtime at the worker's preflight: Python 3.12.14; NumPy 2.3.5; SciPy 1.17.0; scikit-learn 1.8.0; threadpoolctl 3.6.0; Linux x86_64; 5 CPUs. BLAS/OpenMP pools were verified at one thread. Full runtime details are in the outside-Git run artifacts.

## Tests and reproducibility artifacts

The amended/fixed runner passes **15/15 synthetic tests**; compile and JSON validation pass. Tests cover sealed HV1/HV2 sentinels, fit/audit separation, audit-order identity, CMI toys, shuffle-count preservation, exact-alias detection and source-only row access, restore-receipt/hash/size metadata, one-shot consumed state, and hard CPU/wall limit behavior. These tests use only synthetic arrays.

Outside-Git immutable run artifacts: `/workspace/uwb-stage-a-descriptive-20261025/run01/`. They include the actual `worker.stderr.log`, `worker.stdout.log`, terminal `run_status.json`, `BUDGET_RECEIPT.json`, `RUN_ATTEMPT.json`, `runtime_environment.json`, the registered protocol snapshot, and `artifact_hashes.json`. No array, label, prediction, or per-sample data was copied into Git.

The same non-data run receipts are committed in the private branch under `uwb_direction1/results/cloud_20261025_l_t_privileged_group_stage_a/`; the final postmortem artifact commit is `9af565272009781b68b90aa05c796016c933be66` (ordinary content commits; no force update).

## Strongest support and counterevidence

This attempt supplies **neither support nor refutation** for the L/T subgroup hypothesis: it ended before reading source samples. The only newly verified fact is an engineering schema mismatch, already fixed and covered synthetically. The known HV2 aliases do not directly contaminate HV3-HV8, while unresolved trial dependence still forbids inference. Prior-art limits remain: subclass discovery, privileged-view learning, and group-robust learning are established; even a future positive descriptive screen would not by itself be novel or validated.

The user authorized one run. It has been used and no retry is performed. Any future real run requires a new explicit instruction; no additional seed, parameter sweep, CNN, or background training is authorized.
