# Independent conditional-policy result audit

**PASS**: 56301 elementary consistency/numerical checks; 0 errors.

Run: `D:/D_conda/envs/pytorch/python.exe D:/uwb_gestures/experiments/p1_20260928/value_policy_s2026/independent_audit.py`

This is a CPU-only scientific-result audit. It does not import the original analysis module and writes only its script and the JSON/Markdown audit reports.

## Recomputed outcome

**STOP_VALUE_POLICY_MECHANISM**.

| Method | User-mean macro accuracy | User-mean actual views |
|---|---:|---:|
| extra_trees | 90.863257576% | 2.007873106061 |
| class_gain | 87.563825758% | 1.786047348485 |
| frozen_budget2_cascade | 91.920833333% | 2.004615530303 |
| fixed_mixture_within_candidate_cost | 88.679935092% | 2.007873106061 |
| three_views | 93.441666667% | 3.000000000000 |

| ExtraTrees minus comparator | Difference | Positive users | User bootstrap 95% interval |
|---|---:|---:|---:|
| frozen_budget2_cascade | -1.057575758 pp | 0/8 | [-1.645833333, -0.515909091] pp |
| class_gain | 3.299431818 pp | 8/8 | [1.721742424, 5.339015152] pp |
| fixed_mixture_within_candidate_cost | 2.183322483 pp | 6/8 | [0.656602083, 3.895114924] pp |

ExtraTrees minus LTR: -2.578409091 pp.

## Gate reconstruction

| Gate | Pass |
|---|---|
| mean_actual_views_at_most_2 | False |
| within_1pp_of_three_views | False |
| gain_at_least_2pp_vs_frozen_budget2_cascade | False |
| positive_at_least_6users_vs_frozen_budget2_cascade | False |
| population_cost_no_greater_than_frozen_budget2_cascade | False |
| gain_at_least_2pp_vs_class_gain | True |
| positive_at_least_6users_vs_class_gain | True |
| population_cost_no_greater_than_class_gain | False |
| gain_at_least_2pp_vs_fixed_mixture_within_candidate_cost | True |
| positive_at_least_6users_vs_fixed_mixture_within_candidate_cost | True |

Every comparator is mandatory. Better accuracy than class-gain or the fixed mixture does not override the cascade accuracy failure, excess cost, adaptive-cost fairness failures, or the LTR gap.

## Evidence coverage

- Eight complete folds; live and frozen source/plan hashes, model pickle hashes, result identities, validation IDs, raw labels/subjects and role alignment checked.
- Per-class 60/40 split independently regenerated with seed 33000 + fold*100 + class. Source-gradient, validation and test users remain disjoint; checkpoint-selection reuse of validation is explicitly retained as a limitation.
- All class-gain tables reconstructed with 20-example shrinkage. All 24 ExtraTrees models refit in memory with 200 trees, depth 4, minimum leaf 20, seed 2026 and n_jobs=1. All 4800 tree node/value states have complete comparison records and canonical hashes in the JSON evidence.
- All signed negative, zero and positive gain targets reconstructed. All 208 price-grid points and tie/budget selection rules independently replayed using only price-validation rows.
- Loaded-pickle causal routes reproduced every test prediction, state path, added-view count, requested mask and saved gain. Poisoning all unvisited subsets with NaNs leaves outputs unchanged. Direct add-both costs two extra views; all final costs equal unique requested physical views.
- Fixed mixtures checked against an independent linear program with a cost ceiling, including a synthetic case where the optimal solution is strictly cheaper. All result fields and final summary/bootstraps/gates recomputed from raw arrays.
- The previous labeled-oracle opportunity trigger is independently recomputed and recorded as conditional/exploratory, not external confirmation.

## Per-fold decisions

| Test user | Validation user | Fit/tune/test rows | Class-gain price | ExtraTrees price | ExtraTrees accuracy | ExtraTrees views |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 2 | 600/400/1200 | 0.05 | 0.01 | 96.000000% | 2.323333333333 |
| 2 | 3 | 720/480/1000 | 0.1 | 0.05 | 98.800000% | 1.927000000000 |
| 3 | 4 | 720/480/1200 | 0.15 | 0.1 | 88.500000% | 1.898333333333 |
| 4 | 5 | 720/480/1200 | 0.05 | 0.05 | 89.916667% | 1.945000000000 |
| 5 | 6 | 660/440/1200 | 0.05 | 0.05 | 92.750000% | 2.055000000000 |
| 6 | 7 | 720/480/1100 | 0.05 | 0.05 | 72.272727% | 2.151818181818 |
| 7 | 8 | 720/480/1200 | 0.05 | 0.02 | 95.000000% | 2.003333333333 |
| 8 | 1 | 720/480/1200 | 0.2 | 0.1 | 93.666667% | 1.759166666667 |

## Limits

- The validation user received no encoder gradient updates but was used to choose the encoder checkpoint, the frozen cascade and now the policy fit/price partition. This is internal reuse, not a fresh independent policy-development user.
- The predeclared oracle trigger uses outer-test labels. Conditional policy results from these same eight users are exploratory and cannot be called new external confirmatory evidence.
- Within run_arrays, choices are serialized before target_loader is invoked, and independently reconstructed fit/price inputs contain only validation IDs. Earlier budget/oracle analysis and policy preflight already read target arrays; this audit does NOT claim that all target I/O occurred after selection.
- Every saved sklearn ExtraTrees model was independently refit in memory with the frozen 14 observed-state features and signed -1/0/+1 targets; all 4800 trees are compared field by field. sklearn fitting/prediction is reused, while routing, features, costs, selection, mixtures, metrics and gates are independently implemented.
- Pickle files are fingerprint-checked and original policy classes are mapped to inert audit classes; original value_policy.py, checked_existing, fit_policy, route_policy and summarize are never imported or called.
- Hashes and saved lineage prove current internal consistency. They do not provide independent historical timestamp attestation. The exact in-memory refits support fit-data/recipe provenance, not reconstruction of historical process timing.
- Processing cost counts unique physical views including the initial view and any direct add-both action. This audit measures no latency, speedup, energy use, hardware saving or acquisition saving.
- The fixed random mixture is optimized under the candidate cost CEILING and can retain a cheaper better subset. This is not an equality-constrained cost match.
- The candidate must beat EACH frozen comparator by at least 2 pp with at least 6 positive users, cost no more than BOTH adaptive comparators and 2 views, and remain within 1 pp of LTR. Passing one comparison cannot repair another failed gate.
- The conclusion concerns one fixed ExtraTrees value predictor, one model seed and eight users under this data/model protocol. It does not show that all active sensing/value policies fail.
- Whole-user bootstrap is exploratory. Outer-fold training sets overlap, users are few, and quality filtering removes specified subject/class combinations.
- B model-training/encoder replay was audited separately. This audit checks the linked raw prediction artifacts and current identities without retraining or rerunning the neural encoder. B audit files remain unchanged.
- All elapsed/refit times in this evidence file are audit-work durations only, not inference-runtime measurements. Latency is explicitly outside scope.

Audit script SHA-256: `b1d8156adc88d43996dc3de364d70a2c1a81cd184ae4b86d4435b4d838a4449a`.
Frozen policy source SHA-256: `0222130b12461521f5bd45fc628caec6da67d0d448796c5ff43cb83f76cc25df`; test source SHA-256: `fe138d320bc0c9d85a046fa810d821b323f90f9b4c6f089e7a9f4934f263c8a1`.
Audit duration: 6.279 s. This is not a latency benchmark.
