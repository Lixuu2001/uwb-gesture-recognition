# L/T-derived within-class strata — pre-run engineering and exchangeability review

**State:** no real diagnostic run. This update adds a bounded process supervisor, synthetic correctness tests, and a source-metadata exchangeability block. It records no scientific result.

## Verified work

- Re-read current private branch `codex/uwb-direction1-cloud-20260930`; current head at review start was `264bf5b24bfb1b4fd0c88dcd9a96412d19537d33`.
- Did not access Drive, replay migration, read radar array values, fit the Stage-A models, or start a training job.
- Reviewed the saved paired-time report, exact-duplicate subject audit, source `quality_audit.json`, and source `data_manifest.json` metadata.
- Added process-level CPU enforcement (`RLIMIT_CPU`), a separate parent wall deadline, process-group termination, single-thread numerical pools, incremental progress/partial artifacts, terminal state classification, and model/runtime/hash manifests.
- Added 12 synthetic unit tests covering sealed-user label and map sentinels, audit-only-after-fit behavior, fit-user membership, frozen group identity under audit-order permutation, CMI toy cases, conditional-permutation counts, exchangeability preflight blocking, CPU limit, wall timeout, and normal completion.

## Source duplicate and exchangeability finding

The “100 duplicate triplets” are not merely duplicate indices in a manifest. The preserved quality audit identifies separate HV2 G11/G12 MAT inputs with equal full arrays at float64 precision for all three views and contradictory gesture labels; the paired-time diagnostic independently compared and verified all 100 cached triplets. The source manifest maps each G12 repetition to its G11 counterpart. Provenance cannot establish whether this came from repeated acquisition or upstream copying, so it is conservatively classified as an exact data alias across distinct gesture files. The already-existing quality rule excludes the HV2/class-10/11 conflict, HV2 is outside Stage A source users 3–8, and the manifest has no other exact duplicate rows in those source users.

The current metadata does **not** establish independence among the 100 recordings in each participant/class cell: it supplies counts/indices, but not trial timestamps, acquisition order, session IDs, or repeat-level block identifiers. Thus the frozen per-record permutation within `(class, audit user, Right group)` is not statistically cleared. The code now reads `protocol.json` and exits as `blocked_exchangeability_review` before calling `np.load()` while that review is unresolved. This is an inference-validity block, not a failed Stage-A result.

Before any data run, the owner must select one pre-result path: recover trustworthy session/order metadata and preregister a block-level null; revise Stage A before data access into a descriptive-only screen with no record-level p-value; or stop this candidate. Cache row order will not be treated as acquisition order. Effect, coverage, split, representation, and model thresholds remain frozen.

## Budget and artifact behavior

The supervisor launches one worker in its own session/process group. The kernel CPU ceiling is `RLIMIT_CPU` soft=hard=900 seconds. An independent monotonic parent deadline is 900 seconds wall and SIGKILLs the worker group without an extra grace interval. The worker inherits one-thread settings for OpenBLAS, OMP, MKL, BLIS, NumExpr, and vecLib; `threadpoolctl` also limits loaded pools to one thread.

The worker flushes source-only feature rows every 32 records, the permutation null every 50 draws, and atomically records progress after each chunk. Fitted classwise scaler/PCA/KMeans centers/remaps and the Right Ridge state are serialized locally as classes finish. Parent finalization preserves partial files, last worker phase/progress, logs, CPU use, elapsed wall time, hashes, and one of `aborted_cpu_limit`, `aborted_wall_timeout`, `failed_exception`, or `blocked_exchangeability_review`. None of these states is a scientific negative. A complete run records Python/platform/numerical-library versions, observed thread pools, initial code/protocol hashes, fitted model state, null statistics, and per-file artifact hashes. Data-derived artifacts stay outside Git.

## Synthetic test evidence

Command:

```bash
python -m py_compile diagnostic.py tests/test_budget_and_protocol.py
python -m unittest discover -s tests -v
```

Result: **12/12 passed** in 2.83 seconds. The CPU-limit test used a synthetic CPU-burning subprocess with a 1-second limit; the wall test used a synthetic sleeping subprocess with a 0.3-second deadline. A separate synthetic runtime-manifest test confirmed all loaded BLAS/thread pools report one thread. Both left their synthetic partial file, emitted the right inconclusive terminal status, and created no scientific result file. The protocol preflight test patched `numpy.load` to raise if called; it observed `blocked_exchangeability_review` without any array access. Other tests used small synthetic arrays only.

No real cache values, labels, logits, or weights were read during this review. No Stage-A diagnostic or CNN was run.

## Requested next decision

The implementation is ready for review, but a real run should not be authorized under the current record-level permutation until exchangeability is resolved. Choose the pre-run path above or stop the candidate. After that, a separately bounded Stage-A run can be considered; no CNN is authorized by this update.

## Interpretation constraints

The matched R-only control is a single fixed binary K=2 view-derived partition. Incremental CMI, if ever estimated, would mean only “incremental to this control,” not incremental to all information available in Right. HV7/HV8 class labels are used as source-audit diagnostic strata after fit; they are not deployment-time inputs. Those users have prior project-development use and are not untouched independent validation.
