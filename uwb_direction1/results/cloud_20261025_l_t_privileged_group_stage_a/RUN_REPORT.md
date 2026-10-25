# L/T privileged within-class strata — Stage A run02 outcome

**State: completed descriptive feasibility screen; no contribution validated.** Run02 was the explicitly authorized same-protocol retry after run01 failed in metadata preflight before array access. Run01 remains preserved as an engineering-incomplete attempt and is not a scientific negative. Run02 completed all 1,999 fixed-seed descriptive shuffles. No third run, CNN, parameter search, extra seed, or follow-up experiment is authorized by the consumed run02 instruction.

## Frozen data boundary and source-integrity check

- Source cache: 7,100 retained records after the existing quality mask; 4,700 fit records from users 3–6 and 2,400 inner audit records from users 7–8. All 12 classes appear in fit and audit. HV1/HV2 remained sealed; HV7/HV8 labels were used only for post-fit class-conditional diagnostics and are not deployment inputs or independent external validation.
- The 100 known exact L/T/R triplets map to participant ID 2 = sealed HV2, in separate `HV_02_G11/G12` inputs. They are outside the source split, so they do not directly contaminate HV3–HV8. Provenance remains unresolved (repeat acquisition versus upstream copying).
- The preregistered pre-fit exact-triplet audit checked all 7,100 retained source rows: 0 exact alias pairs and 0 digest-collision pairs. The scan passed. This does not resolve possible within-session/trial dependence: no timestamps, capture order, session IDs, or repeat-block metadata are available.

## Frozen descriptive results

The single fixed Right-only Ridge probe had audit errors of 673/1,200 (56.08%) for HV7 and 551/1,200 (45.92%) for HV8. These are diagnostic linear-probe results, not results for the historical CNN.

| Quantity | run02 result |
|---|---:|
| `I(error; L/T group | class,user)` | 0.009603 bits/sample |
| `I(error; Right group | class,user)` | 0.022300 bits/sample |
| Increment `I(error; L/T group | class,user,Right K=2 group)` | 0.011062 bits/sample |
| `I(error; user | class)` | 0.181763 bits/sample |
| Classes meeting frozen group coverage | 1/12 (class 4 only) |
| Classes with same higher-error L/T group in both audit users | 1/12 (class 4 only) |
| Descriptive incremental-effect floor 0.005 | passed |
| Frozen screen as a whole | did not support further design |

The fixed-seed, 1,999-draw shuffle reference had median 0.008454 bits/sample (5th–95th quantiles 0.005498–0.012368; observed-minus-median 0.002608). Because trial/session exchangeability is unresolved, this is only a finite-randomization description: **no p-value, significance, inferential null, or decision gate was computed or used.**

Class 4 is the only positive descriptive slice. Its frozen L/T group 0 had higher Right-probe error in both audit users: HV7 1.000 vs 0.881 (n=16 vs 84), and HV8 0.697 vs 0.455 (n=89 vs 11). However, HV8’s smaller group is only just above the support threshold, and the class fails the frozen coverage rule because fit user 4 has 95/5 group counts. The other 11 classes fail coverage and same-direction support; several are degenerate or severely imbalanced. Per-class/per-user counts, errors, and descriptive CMI are in `DESCRIPTIVE_RESULTS_SAFE.json`.

## Interpretation: strongest support and counterevidence

The strongest support is that the aggregate incremental CMI exceeds the predeclared descriptive floor, and class 4 shows a consistent error direction across the two inner audit users. This only shows an association for one fixed representation and one fixed binary Right-cluster control; it does not establish that L/T adds information beyond every possible Right-only representation.

The dominant counterevidence is poor subgroup coverage and transfer stability: only 1/12 classes meets either criterion, against a frozen 8/12 requirement. User-conditioned error information (0.181763 bits/sample) is much larger than the L/T or Right-group summaries, indicating substantial user structure. The clusters are statistical strata, not verified motion or kinematic subgroups. Existing work (GEORGE, Domino, EIIL, AGRO, privileged-information learning, and same-corpus multi-view recognition) already establishes the component concepts; this screen did not establish a novel algorithm or effective training method.

Therefore the result is **not** a general scientific negative for all privileged-supervision mechanisms, but it fails this narrow frozen feasibility screen and does not justify proceeding to Stage B. No method contribution, independent-user generalization, or performance improvement was validated.

## Runtime, budget, and reproducibility

- run01: 0.966107 CPU seconds / 1.003384 wall seconds; failed before `np.load`, no data or labels read.
- run02: 25.742206 process CPU seconds / 26.273591 wall seconds; all 1,999 shuffles completed.
- Cumulative: 26.708313 CPU seconds / 27.276974 wall seconds, below the 900/900 limits. BLAS/OpenMP were verified at one thread. Runtime was Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, threadpoolctl 3.6.0, Linux x86_64, five CPUs.
- The 15/15 synthetic budget/protocol tests passed before the run. No training/CNN ran.

Pre-run hashes: protocol `303d9322fca6d7a51c61fbbf5abff17ef09ec35b243bf907fe0619ff237220bd`; diagnostic `984743fe8e2de4526fa78df8c79c49b95515130f9aed81a24ed46ad90e2896d0`; budget wrapper `2585b65d905d08d47d398f449d05f7261ee0b2c2622daef7a378fea1ed355566`; tests `a90289c04a9c0116ec24881b28ca44d8933b05a440145a24a93fa34c13c8f5a8`. run02 execution-manifest SHA `299b118657d4e699f982de252d7c893380c7e9335cfad6e67d6b65e52a320aa2`; safe aggregate SHA is recorded in the branch artifact manifest.

Full, value-derived local artifacts remain outside Git under `/workspace/uwb-stage-a-descriptive-20261025/run02/`, including model states, features, row-level audit, row-level hashes and shuffle array. Only aggregate counts/statistics and hashes are committed. Protocol snapshot used: `/workspace/uwb-stage-a-descriptive-20261025/run02_prelaunch_protocol.json`.
