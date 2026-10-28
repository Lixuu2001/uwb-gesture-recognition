# Run02 amendment — range-time echo Betti source feasibility

**Registered before the repaired run.** This is the single repair-and-retry authorized after the first invocation stopped at the pre-fit retained-count assertion. It does not alter the scientific candidate or gate.

## Engineering correction

The first preregistration expected participant 6 to retain 1,107 rows. The archived source audit at `experiments/p1_20260905/AUDIT.md` says HV6 G4 has 93 rows excluded and HV6 retains 1,100 rows (11 classes); the original corrected dataset has 9,300 rows total. The observed source counts 1–6 were 1,200 / 1,000 / 1,200 / 1,200 / 1,200 / 1,100, totaling 6,900. This agrees with the archive. The incorrect 1,107 / 6,907 figures were transcription errors in the preregistration's preflight expectation.

The first run is preserved at `results/failure.json` locally and its incomplete report remains in Git. It produced 0 features, 0 models, no scores, and read no map pixels or participants 7–8 gesture labels. The only protocol data-field correction for run02 is expected user 6 count 1,100 and total source count 6,900. No observed model result informed this amendment.

## Frozen method and decision rule

The method, six source users (participants 1–6), leave-one-source-user-out folds, fixed quality exclusions, three separately reported views, 3×3 max pool, per-map normalization, eight fixed quantile levels, beta0/beta1 descriptor, amplitude and fixed spatial-shuffle controls, train-only standardization, sample/user balanced RidgeClassifier(alpha=1.0), and the preregistered GO/STOP thresholds are unchanged from `research/range_time_betti_gate/STAGE_A.json`. Participants 7–8 gesture labels are not read. No view selection, fusion, tuning, p-values, CNN, or contribution claim.

The runner also now records an explicit incomplete terminal state on CPU soft-limit or external wall-time signals. CPU soft-stop is 890 s, hard CPU ceiling 900 s, and independent wall timeout 900 s; one process and one BLAS thread. This is an execution safeguard only.

- Protocol SHA-256: `ab8cffd5efaf5be05a9de373cbae88c1a5a254cc1d60cc00cd7b7d641083cfd9`
- Runner SHA-256: `46d08787c130bfbbe4d546eef6826fa045e855068e517711bec7e439bec8896a`
- Synthetic descriptor and timeout-receipt checks: passed without opening dataset files.
- Run status at registration: not started.
- GO would only permit a separate frozen validation design; any incomplete run is not a scientific negative.
