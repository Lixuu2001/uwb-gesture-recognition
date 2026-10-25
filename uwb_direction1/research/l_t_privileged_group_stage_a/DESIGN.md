# Stage A — L/T privileged within-class subgroup feasibility screen

**Status: frozen design and runnable diagnostic prepared; not run.** Date: 2026-10-25 UTC.

## Question

Within each known gesture class, can L/T-only signal strata fitted on source users 3–6 transfer by frozen centroids to source users 7–8 and add information about errors of a Right-only diagnostic probe beyond gesture class, audit-user identity, and a matched Right-derived stratum?

This is a narrow feasibility question for later training-only supervision. It does not assert that clusters are physical or kinematic subgroups, that a linear-probe error matches the historical CNN's failures, or that any method contribution has been validated.

## Existing experiment audit

The exact test was not found in the latest branch records or preserved project results:

| Existing result | What it actually measured | Why it is not this test |
|---|---|---|
| `results/cloud_20261003_group_gap/RUN_REPORT.md` | User-level source groups; four train users and one validation user per trajectory; the source-group continuation gate passed 0/6 at epochs 12/24. | It did not infer class-conditional groups from L/T views, transfer group centers to two inner source users, or test within-user Right error association. |
| `results/cloud_20261003_regularized_groups/RUN_REPORT.md` | Regularized GroupDRO on known user groups, matched against user-balanced CE and a historical tail baseline; primary comparisons failed. | It neither discovered within-class groups from L/T nor treated L/T as training-only privileged supervision. |
| `original/project/experiments/p1_20260905/OOF_PRIVILEGED_KD_PLAN.md` and `confusion_kd_results/RESULTS.md` | Full multi-view teacher or multi-view OOF soft-target distillation into a Right-only student. OOF-shape student was −0.23 pp vs the single-view OOF control and −2.87 pp vs global smoothing; the earlier teacher-uniform result was 74.61% vs 74.65% for label-uniform. | This used teacher probabilities / label smoothing, not L/T-derived binary within-class stratum labels. It is a relevant negative result, not an equivalent experiment. |

The restored cache contains the 1.96-GB real-valued clutter-removed map and historical P1 artifacts. A metadata-only NPZ-header check found `features (9300,3,256)` and `logits (9300,7,12)` in the P1 multi-view fold files. These are seven view-subset outputs, not Right-only predictions from a model trained on users 3–6 and evaluated on 7–8. No NPZ payload values were read. No suitable historical source-only Right-CNN predictions/checkpoint were found in the current executor. Therefore the prepared diagnostic uses one fixed, non-neural Right-only Ridge probe; a Stage-A pass could only justify a new instruction to run a matched CNN confirmation.

## Frozen split and leakage boundary

- Existing quality rule only: exclude subject 6 / class 3 among the source rows. The conflicting subject 2 labels and outer held subject 1 are not opened.
- Outer test user 1 and validation user 2 are sealed: no radar rows or gesture labels for them are loaded by the script.
- Source users 3–8 are split once: fit all transforms, KMeans centers, and the Right probe on users 3–6; apply them unchanged to inner source audit users 7–8.
- The inner audit labels are used only after all transforms/probe are frozen, to calculate error association. Subjects 3–8 have prior project development use, so this is not independent validation or an untouched user test.
- For each class, L/T groups use fixed magnitude normalization, 10×21 average pooling, train-only StandardScaler, train-only 16-component PCA, then KMeans K=2 (one fixed seed and n_init=10). The R-only control uses the same transform and clustering procedure on R. The Right-error probe is a single train-only StandardScaler + RidgeClassifier (`alpha=1`, `class_weight=balanced`), with no parameter selection.
- Cluster IDs are sorted by the sign-fixed first PCA coordinate of the fit centroids. Audit rows are assigned to those frozen centers. No re-clustering, audit-label alignment, Hungarian matching, or group identity comparison across folds. If Stage B is later authorized, each outer fold gets fold-local IDs from that fold's training users; no cross-fold identity is claimed.
- Audit labels are used only after model/group fitting, to assign each audit row to its known class's already-fitted centers and calculate the fixed association statistics. They never enter scaler/PCA/KMeans/Ridge `.fit()` calls. Permuting audit row order preserves fitted state and per-row group identity.

## Source exchangeability preflight — unresolved; no run authorized

The preserved paired-alignment report and independent source audits resolve the earlier “100 duplicate triplets” note. The manifest lists 100 exact duplicate row pairs, all in participant 2 and across separately named `HV_02_G11` / `HV_02_G12` inputs. The original quality audit reports equal full matrices at float64 precision for L, T, and R and contradictory labels; the label-free alignment run independently verified each duplicate across all three cached views. This is a data-level alias across separately named gesture arrays, not merely a repeated manifest index. The available provenance cannot tell whether it came from repeated acquisition or upstream copying. Existing quality handling excludes the participant-2/class-10/11 contradiction; participant 2 is outside Stage A's source users, and none of the known exact duplicate rows enters users 3–8.

The source manifests and loader establish 100 rows per participant/class cell but do not preserve per-trial timestamps, acquisition order, session IDs, or repeat-level block identifiers. Thus the planned row-level conditional permutation within `(class, audit user, Right group)` is not cleared: it could overstate evidence if clips share a capture session or temporal dependence. This is a pre-result design limitation, not a negative result. The runner reads the protocol and writes `blocked_exchangeability_review` before any `np.load()` while this status remains unresolved.

Before a real run, either (a) recover trustworthy per-trial/session/order metadata and preregister a block-level permutation preserving those acquisition clusters, (b) revise Stage A before data access to a descriptive-only two-audit-user screen without a record-level inferential p-value, or (c) stop this candidate. Do not invent contiguous blocks from cache row order. The frozen effect/coverage thresholds remain unchanged, but the current permutation gate is uninterpretable until this issue is resolved.

## Primary measure and controls

Primary: `I(Right-probe error; L/T group | class, audit user, R-only group)` in bits per audit sample. A fixed conditional permutation test shuffles L/T group IDs within `(class, audit user, R group)`, preserving those counts, using 1,999 draws and seed 20261025.

The Right-derived control is one fixed binary K=2 partition. A positive conditional MI would show incremental information only relative to this finite R-group control; it would not show that L/T adds information beyond every possible Right-only representation. HV7/HV8 gesture labels are used only for class-conditional source-audit diagnostics, not as deployment-time inputs. HV7/HV8 are not new independent validation users.

Report the matched R-only group association `I(error; R group | class,user)`, user-only control `I(error; user | class)`, raw L/T association `I(error; L/T group | class,user)`, per-user/class subgroup coverage, and whether the same frozen L/T group has the higher Right-probe error in both audit users. These are diagnostics; no classwise selection is allowed.

## Frozen decision rule

All conditions must hold to mark Stage A eligible for follow-up:

1. In at least 8/12 classes, each group has at least 10 examples in each audit user, and both groups have at least 10 examples in at least 3/4 fit users.
2. In at least 8/12 classes, both audit users show the same frozen group ID with at least a 5 percentage-point higher Right-probe error, with at least 10 examples per group/user.
3. Primary conditional MI is at least 0.005 bits/sample and its single global permutation p-value is at most 0.05.

Any failed condition closes this candidate. Do not change K, representation, split, probe, effect threshold, or permutation count after seeing results. A pass is not a contribution; it only supports asking separately for Stage B: matched Right-only CNN CE versus the same CNN with an auxiliary L/T subgroup head, plus R-group, user-group, and within-user/class count-preserving random-group controls. L/T must be absent at inference. The Stage B primary comparison and held-user rule must be frozen before fitting.

## Prior-art boundary

The following component ideas are established and are not claimed as new: hidden-subclass clustering/measurement and robust mitigation (GEORGE); cross-modal error-aware slice discovery (Domino); partition inference for invariant learning (EIIL); adversarial error-prone group discovery with robust optimization (AGRO); and training-only privileged information / teacher-to-student transfer (LUPI). The same UWB-Gestures corpus already has single/dual/tri-view X4 comparisons and fusion (Xue et al. 2026); the project already tested multi-view teacher-to-Right student distillation and did not find teacher increment over its controls.

The narrow, not-yet-tested empirical distinction is only: **whether L/T-derived class-conditional groups transfer across source users and add Right-error information beyond a matched R-derived group and user identity on this exact corpus.** This is a bounded search, not novelty clearance. Any Stage B result would require another primary-source review; no group discovery algorithm or combination of known components may be described as the contribution.

Primary references:

- GEORGE, NeurIPS 2020: <https://papers.neurips.cc/paper_files/paper/2020/hash/e0688d13958a19e087e123148555e4b4-Abstract.html>
- Domino, ICLR 2022: <https://openreview.net/pdf?id=FPCMqjI0jXN>
- EIIL, ICML/PMLR 2021: <https://proceedings.mlr.press/v139/creager21a.html>
- AGRO, ICLR 2023: <https://openreview.net/pdf?id=IrzkT99fDJH>
- LUPI, Vapnik & Vashist 2009: <https://doi.org/10.1016/j.neunet.2009.06.042>
- Privileged information / knowledge transfer, JMLR 2015: <https://jmlr.org/papers/v16/vapnik15b.html>
- Same-corpus multi-view Xue et al. 2026: <https://doi.org/10.3389/fnins.2026.1864448>

## Runtime and reproducibility

The restored 86-file dataset was already SHA-verified at `/workspace/uwb-restored-20261020`; no restore is repeated. Current workspace has no active training process. The executor has 5 CPUs and NumPy/SciPy/scikit-learn; no PyTorch/GPU. The parent launches one worker in a new process group through `budget_exec.py`, which sets the CPU limit and `exec`s the worker in the same PID (avoiding a multithreaded-parent `preexec_fn`). Linux `RLIMIT_CPU` soft=hard enforces the 900 process-CPU-second ceiling; an independent monotonic parent timer SIGKILLs the group at 900 wall seconds. BLAS/OpenMP/BLIS/NumExpr are limited to one thread before worker import and through `threadpoolctl`. Feature rows flush every 32 records; per-class scaler/PCA/centroids/remaps and the Right probe state are atomically checkpointed; null statistics flush every 50 draws. Terminal status records last phase/progress and partial artifact paths. CPU/wall timeout, exception, and exchangeability block are inconclusive statuses, never scientific failures.

The implementation is `diagnostic.py`; it requires `--confirm-stage-a`, but that flag does not bypass the exchangeability preflight. An authorized run stores Python/platform/numerical-library versions, observed thread pools, initial code/protocol hashes, source-only IDs/labels/users, fitted model state, null statistics, and artifact hashes locally. Value-derived artifacts are not committed. **Only synthetic tests are permitted in this review; no real maps, labels, logits, or weights were read and no Stage-A fit was run.**

The previous suggested command is withdrawn until exchangeability is resolved. Currently even `--confirm-stage-a` stops before opening cache arrays and records `blocked_exchangeability_review`, not a scientific negative.

## 2026-10-25 descriptive-only amendment (preregistered before this run)

The owner selected the descriptive-only path. Trial/session exchangeability remains unresolved; therefore the 1,999 count-preserving row shuffles are retained solely as a descriptive finite-randomization reference. The runner computes no permutation p-value, reports no significance, and uses no inferential criterion or p-value for any decision. The original split, feature transform, per-class K=2 fit, fixed Right Ridge probe, CMI calculation, thresholds, shuffle count, and seed are unchanged. A screen that meets the descriptive thresholds is labeled only `descriptive_screen_supports_further_design`; it is not a scientific GO, validated mechanism, or contribution.

The prior exact-alias audit identifies 100 identical L/T/R triplets only for participant ID 2, which the source quality audit maps to sealed HV2 (`HV_02_G11/G12`). These pairs do not directly enter the present HV3-HV8 source split. Their provenance remains unknown (repeat acquisition versus upstream copy). Before any model fitting, the amended runner hashes every retained source sample's complete three-view input and blocks if any exact alias occurs; it neither removes nor relabels a source row after seeing results. The scan accesses only rows and labels selected under the frozen source split and quality rule; HV1/HV2 radar rows and labels remain sealed.

All other interpretation boundaries remain: the binary Right K=2 control is finite and does not stand for all Right information; HV7/HV8 labels are post-fit class-conditional diagnostic strata, unavailable at deployment; HV7/HV8 have prior project-development use and are not independent validation. This screen can only guide whether a separately designed validation study is worthwhile.
