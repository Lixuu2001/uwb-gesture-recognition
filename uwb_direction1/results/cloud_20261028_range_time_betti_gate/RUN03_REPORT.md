# 2026-10-28 — Range-time Betti Stage A result

## Outcome

The amended source-only screen completed and **failed its frozen GO gate** for this exact descriptor. This closes this representation under the preregistered configuration; it does not reject every topological or signal-based method.

| Measure | Result | Required | Gate |
|---|---:|---:|---|
| Mean macro-recall delta: Betti − amplitude | −0.027164 (−2.716 pp) | ≥ +0.010 | Fail |
| Mean delta: Betti − spatial-shuffle null | +0.056046 (+5.605 pp) | ≥ +0.010 | Pass |
| Source users improved over amplitude, Left | 0/6 | ≥4/6 in ≥2 views | Fail |
| Source users improved over amplitude, Top | 3/6 | ≥4/6 in ≥2 views | Fail |
| Source users improved over amplitude, Right | 3/6 | ≥4/6 in ≥2 views | Fail |
| View mean delta lower bound, Left | −0.053765 | ≥ −0.010 | Fail |
| View mean delta lower bound, Top | −0.018260 | ≥ −0.010 | Fail |
| View mean delta lower bound, Right | −0.009467 | ≥ −0.010 | Pass |

Betti features beat the fixed spatial shuffle descriptively, suggesting this representation retains local arrangement beyond the intensity multiset. They nevertheless lose to the spatial-layout-free amplitude quantiles overall, and no view reaches the preregistered user-consistency threshold. No p-values were planned or computed. The result does not justify Stage B or a contribution claim. Do not change thresholds, masks, connectivity, quantiles, readout, or seed to rescue this candidate.

## Run and integrity receipt

- Corrected protocol SHA-256: `aae558f7c50c94993f9d383709ba5fdf0111fb1b382cf913efe63bf9a471a4d8`.
- Executed runner SHA-256: `d78df4577005c4339612deabec5aea8d9302fb48c6ab984c537c6c7f604ba00c`.
- Input records: 6,900 source rows from participants 1–6; retained counts 1,200 / 1,000 / 1,200 / 1,200 / 1,200 / 1,100.
- Feature records: 6,900; fitted models: 54 (three fixed views × three methods × six held-source users).
- CPU 68.922 s; wall 68.932 s. Earlier pre-score count-guard attempts each ended in under 2 s and produced zero features/models; total use stayed below the authorized 900 s budget.
- Participants 7–8 gesture labels were never read. No fitting/evaluation used them.
- All 54 model artifact hashes and the source-feature artifact hash were verified against the saved state.
- Environment receipt: `environment.json`; per-fold predictions, source IDs, fitted parameters, and feature matrices are in the private cloud workspace under this run directory.
- No raw arrays or bulky model artifacts were committed to Git. No process remains running.

## Terminal-state recovery

The analysis completed all 54 fits, wrote the complete state and exited with code 0. The wrapper then caught `SystemExit(0)` because it used `except BaseException`, incorrectly overwriting only the terminal status as incomplete. We preserved that original wrapper receipt, verified every saved model and feature hash, and restored the terminal metadata to complete without recomputing any data or scores. The exact executed runner remains recoverable from the registered commit and hash above.

Runs 01 and 02 remain preserved as pre-score engineering failures: first an incorrect expected HV6 count, then a stale total-count literal. The archived source audit explains the correct 1,100 HV6 retained count. Their logs did not expose any score and did not change the gate.

## Research status

This result is exploratory: source participants 1–6 have prior project-development exposure and are not an independent cohort. It is not a validated contribution or external generalization result. The exact tested descriptor is closed with `STOP_CANDIDATE`; the overall method-contribution goal remains unresolved. Next work must identify a materially different, physically justified observable with a distinct literature basis. No Stage B, classifier tuning, or extra view selection is authorized by this GO/STOP screen.
