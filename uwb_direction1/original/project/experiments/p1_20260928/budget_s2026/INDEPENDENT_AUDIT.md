# Independent audit of B: fixed multiradar processing budget

**PASS** — 168963 checks; 0 failures; CPU only.

Run: `D:/D_conda/envs/pytorch/python.exe D:/uwb_gestures/experiments/p1_20260928/budget_s2026/independent_audit.py`

The standalone script imports no original analysis/training module. It reads the formal eight-fold artifacts and writes only this report and its JSON evidence companion.

## Independently recomputed results

| Quantity | Result |
|---|---:|
| Fixed L macro accuracy | 68.153409091% |
| Fixed T macro accuracy | 58.108522727% |
| Fixed R macro accuracy | 71.676515152% |
| Fixed LT macro accuracy | 87.493181818% |
| Fixed LR macro accuracy | 84.106628788% |
| Fixed TR macro accuracy | 90.583333333% |
| Fixed LTR macro accuracy | 93.441666667% |
| Primary cascade accuracy | 91.920833333% |
| Primary cascade actual mean views | **2.00461553030303 (>2)** |
| Cascade minus validation-selected mixture within its actual cost ceiling | 2.745349759 pp |
| Paired user bootstrap 95% interval | [1.279450955, 4.322051148] pp |
| Positive users | 7/8 |
| Oracle at ceiling 2 minus mixture at ceiling 2 | 5.508522727 pp |
| Oracle at cascade actual cost ceiling minus cascade | 3.124621212 pp |
| Conditional opportunity decision | **POLICY_PILOT_WARRANTED** |

The cascade fails the strict <=2 average-view deployment budget and the <=1 pp gap-to-LTR condition. Passing the oracle opportunity test only authorizes a conditional policy pilot under the frozen plan; it is not a successful learned policy.

## Fold evidence

| Test user | Validation user | Train/val/test rows | Baseline/MV selected epoch | First→pair | Cascade accuracy | Actual views | Joint-only rescues |
|---:|---:|---|---|---|---:|---:|---:|
| 1 | 2 | 7100/1000/1200 | 11/12 | R→TR | 96.416667% | 2.127500000000 | 5 |
| 2 | 3 | 7100/1200/1000 | 6/2 | R→TR | 99.200000% | 1.747000000000 | 2 |
| 3 | 4 | 6900/1200/1200 | 2/14 | L→LT | 90.750000% | 2.005833333333 | 2 |
| 4 | 5 | 6900/1200/1200 | 8/5 | L→LT | 92.250000% | 2.100833333333 | 35 |
| 5 | 6 | 7000/1100/1200 | 37/2 | L→LR | 92.833333% | 1.757500000000 | 21 |
| 6 | 7 | 7000/1200/1100 | 2/3 | R→TR | 74.000000% | 2.289090909091 | 30 |
| 7 | 8 | 6900/1200/1200 | 20/2 | R→TR | 95.583333% | 2.052500000000 | 27 |
| 8 | 1 | 6900/1200/1200 | 1/2 | R→TR | 94.333333% | 1.956666666667 | 16 |

## Verification coverage

- Raw labels/subjects: 9593 records, 293 excluded by the frozen quality rule, 9300 retained; exact 6/1/1 whole-user split and all IDs/roles/labels checked.
- Live source/data hashes, frozen source snapshots, manifests, same-fold Right checkpoint lineage, actual inherited encoder/projection tensors and seeded fusion-head tensors checked.
- All listed artifacts hashed; each epoch history and epoch JSON checked; shuffle/subset RNG traces and optimizer step counts reconstructed; initial candidate, earliest mean-seven-CE selection, best state and embedded progress checkpoint checked.
- All seven masks, all nine validation threshold selections, every saved cascade prediction/choice, fixed/mixture/oracle result leaf and summary field recomputed. CPU-only full fusion-head replay additionally checks every exported MV logit against the saved features, with cross-device numerical differences retained as diagnostic evidence.
- Mixture vertices independently cross-checked by a seven-variable LP. Oracle bounds independently cross-checked by a full sample-action LP at all 72 actual cascade costs and eight primary cost-2 points. Harmful additions, joint-only rescue, fractional allocation and unequal-class weighting are covered.
- Unobserved data are replaced with NaNs in every saved cascade route replay; predictions/decisions remain identical. Unique requested views exactly reproduce reported sample costs. This is causal replay of saved logits, not a latency measurement.
- 338 file fingerprints and per-array shape/dtype/content hashes are in INDEPENDENT_AUDIT.json.

## Scope and limitations

- Opportunity only: the oracle reads outer-test labels, uses fractional per-row selection, and is not deployable.
- The predeclared test-label oracle decides whether to run a conditional policy pilot. These test users are not new external confirmatory data for that later pilot.
- Cost means number of processed physical radar views per completed synchronized clip, including the first view. It is not measured elapsed time, acquisition power, or hardware reduction.
- The primary cascade has exact user-mean test cost 2.0046155303030306 (>2); its gap to fixed LTR is 1.52083333333333 pp (>1). Neither failure may be rounded away.
- Fixed mixture selection obeys a cost CEILING; the chosen mixture can be cheaper. The oracle is also an upper bound under the same ceiling and may spend less. Labels do not choose the fixed mixture.
- The audit independently replays every formal budget metric and saved route from raw exported logits and checks source/weight/partition hashes. It does not retrain or rerun the complete radar encoder.
- Full CPU BF16 fusion replay from saved features is an additional cross-device diagnostic; small CPU/GPU BF16 differences and any argmax differences are explicitly recorded, and original exported logits remain the audited metric input.
- Only one model seed and eight users. Whole-user bootstrap resamples paired user results; overlapping cross-validation training sets and class omissions limit independence and generalization.
- The shared subset model is one architecture, not seven separately optimized models; the fixed names are descriptive, while validation-selected fixed subsets and mixtures are the fair selection controls.
- Current hashes and internal lineage establish present consistency, not independent historical timestamp attestation. Intermediate epoch logits/weights are not all retained; earliest selection is verified against the complete saved history and embedded progress transaction.
- Source-only smoothing is verified from its saved OOF probabilities, fit/predict ID evidence, matrix reconstruction and fingerprints; the proxy logistic regressions were not refit.
- value_policy and real CPU/GPU latency results are outside this audit; this report makes no claim about their completion status or outcomes.

## CPU fusion replay numerical diagnostic

| Fold | Logit elements | Exact elements | Max absolute difference | Argmax differences |
|---:|---:|---:|---:|---:|
| 1 | 781200 | 779540 | 0.25 | 0 |
| 2 | 781200 | 779451 | 0.125 | 0 |
| 3 | 781200 | 779725 | 0.25 | 0 |
| 4 | 781200 | 779493 | 0.125 | 0 |
| 5 | 781200 | 779417 | 0.0625 | 0 |
| 6 | 781200 | 779695 | 0.125 | 0 |
| 7 | 781200 | 779301 | 0.125 | 0 |
| 8 | 781200 | 779578 | 0.125 | 0 |

Audit script SHA-256: `58309eef9d8bb12e37bf0401f25e0f96ba897416c277415dc87e07c5d97d7a1d`.
Elapsed: 46.906 seconds. CUDA initialized: False.
