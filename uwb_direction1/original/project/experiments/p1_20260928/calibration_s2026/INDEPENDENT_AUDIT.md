# A pilot independent scientific audit

Audit status: **PASS_WITH_SCOPE_LIMITATIONS**. Frozen decision independently reproduced: **STOP_LOW_RANK_MECHANISM**.

3532 checks; 0 failed. CPU only; no implementation, results, parameters, or model weights modified.

## Evidence and counts

- Prepared labels/subjects: 9593 raw rows; 293 protocol exclusions; 9300 retained unique IDs.
- Eight complete folds; six whole users for parameter fitting, one for validation, one for test. HV2 excludes classes 10/11; HV6 excludes class 3 (zero-based).
- Aggregate held-out population: 7440 unique fixed queries + 1860 reserved support candidates. Every observed test class has 80 query rows and 20 candidates.
- Same-budget methods share exactly the same 9 support IDs in each user/repeat: three selected classes, three examples per class, seeds 92000–92019. No support/query overlap.
- Recomputed eight source-only PCA32/variance/masked-rank2 models, all 672 source class-triple scores, all 960 validation grid scores (19200 support-repeat scores), 800 target adaptation deltas and 1190400 saved prediction cells.
- Maximum validation-grid score discrepancy: 0; maximum target-delta absolute discrepancy: 5.14e-13.
- Baseline checkpoint/result/features and calibration model/selection/predictions hashes agree. Original validation/test logits exactly match exported feature-archive logits.

## Independently recomputed outcomes

Accuracy values below are percentages: average the 20 supports within each user, then equally weight the eight users.

| Method | Uncalibrated known classes | All classes | Calibrated classes | Worst user, uncalibrated |
|---|---:|---:|---:|---:|
| frozen | 72.788628 | 71.340909 | 67.187500 | 57.031250 |
| source_prototype | 73.312252 | 71.258996 | 65.364583 | 58.125000 |
| supported_only | 73.171286 | 72.620893 | 70.880208 | 57.484375 |
| translation | 72.368490 | 71.295336 | 68.265625 | 57.031250 |
| ridge_affine | 73.107329 | 72.018643 | 68.830729 | 57.031250 |
| lowrank | 68.951373 | 68.460571 | 67.111979 | 57.132813 |
| validation_selected_simple | 72.751147 | 72.575320 | 71.958333 | 57.484375 |
| allclass_lowrank | 71.441902 | 70.956274 | 70.956274 | 57.031250 |

Lowrank − validation-selected simple: **-3.799774 pp**, 1/8 users improve, paired user bootstrap 95% interval **[-9.879193, -0.208059] pp**.
Lowrank − frozen: **-3.837255 pp**, 2/8 users improve, 95% interval **[-10.129799, 0.465710] pp**.

All five implemented GO conditions fail: mean advantage >=2 pp, >=6/8 positive users, positive mean advantage over frozen, and no user losing >2 pp against either comparator.

## Per-user harm and numerical diagnostics

| Test user | Validation user | Source-selected classes | Query rows | Selected simple | Lowrank − simple, pp | Lowrank − frozen, pp | Imputation iterations |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | [2, 4, 6] | 960 | supported_only | 0.000000 | 0.000000 | 37 |
| 2 | 3 | [4, 5, 6] | 800 | supported_only | -1.428571 | -1.348214 | 45 |
| 3 | 4 | [0, 6, 7] | 960 | supported_only | 0.166667 | 2.694444 | 36 |
| 4 | 5 | [0, 6, 7] | 960 | supported_only | -0.319444 | -0.319444 | 26 |
| 5 | 6 | [2, 4, 6] | 960 | supported_only | -24.493056 | -24.493056 | 25 |
| 6 | 7 | [2, 4, 6] | 880 | supported_only | -0.351562 | 0.101563 | 24 |
| 7 | 8 | [4, 6, 9] | 960 | supported_only | -0.097222 | -0.097222 | 87 |
| 8 | 1 | [4, 7, 8] | 960 | translation | -3.875000 | -7.236111 | 39 |

All eight masked-SVD fits converged within the preregistered 100-iteration ceiling and 1e-8 relative-change tolerance. Missing user/class cells remained masked; no missing cell was treated as a zero observation. The independent source-fit reconstruction, D-optimal class selection, grid maximum and tie rules agree with saved artifacts.

The HV5 lowrank mean loss is -24.493056 pp, and its support-repeat standard deviation is 12.554080 pp. This is observed across 20 alternative support draws for the same user, not 20 independent users.

The uncalibrated-class raw center MSE is 0.965121657 for the frozen centers and 0.982289031 after lowrank adaptation, a reduction of -0.017167375 (negative means worse). This post hoc diagnostic also does not support the proposed center-shift mechanism in this pilot.

The interval versus the frozen classifier includes zero; do not describe that comparison as statistically conclusive harm. The preregistered engineering GO conditions fail regardless.

## Errors and limitations

No numerical, ID/label, support-leakage, model-selection, saved-artifact hash or report-reconciliation errors were found.

- No raw radar-to-feature GPU inference or retraining was rerun; SHA bindings, role IDs, source snapshots, baseline saved logits and exact logit export replay were checked.
- The trusted local source pickle is mapped to an inert attribute holder. Its PCA, source centering, variance, masked SVD and support deltas were independently recomputed.
- Local file hashes prove current consistency, not trusted external timestamps. Amendment mtimes precede the first A selection; cryptographic proof of preregistration chronology is unavailable. AMENDMENT_001/002 hashes are recorded here but not separately bound in the original A identity.
- Eight users and one encoder/model seed. Twenty supports are repeated measurements within a user. User bootstrap does not include retraining uncertainty or correct dependence from overlapping source-user folds.
- Uncalibrated means known output classes omitted from the nine support labels; this is not novel-class or open-set recognition.
- The higher-budget allclass reference uses 30/33/36 labels and a different validation objective. Its calibrated metric is all available classes; its uncalibrated column uses the original nine-label route subset only for reference.
- The raw center-MSE diagnostic uses query labels after selection and before tau scaling. It is not a deployment score and may change even when tau=0 leaves predictions unchanged.
- The support/query split is a retrospective class-stratified evaluation: 20 candidates are reserved from every available class, including nonselected classes. Results apply to the resulting fixed query population.
- Source label-smoothing OOF IDs, saved probabilities, matrix construction and hashes were checked; the historical proxy classifiers were not refit.
- Reported adaptation timing is not an end-to-end inference latency benchmark and was not independently remeasured here.
- The STOP conclusion is specific to this dataset, protocol, single seed and implemented rank-2 mechanism; it does not establish that partial-action calibration is universally ineffective.

## Reproduction

Run the independent auditor with:

```powershell
& 'D:\D_conda\envs\pytorch\python.exe' 'D:\uwb_gestures\experiments\p1_20260928\calibration_s2026\independent_audit_a_20260928.py'
```

The accompanying JSON records every check, per-user metrics, all validated artifact SHA-256 values and independent source-fit diagnostics. The audit writes only its two report files; the script itself is the third allowed artifact.

Generated UTC: 2026-09-28T15:45:10.390498+00:00. CPU audit elapsed: 13.43 seconds.
