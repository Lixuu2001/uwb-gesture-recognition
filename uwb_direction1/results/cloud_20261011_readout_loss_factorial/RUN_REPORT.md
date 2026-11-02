# Frozen-feature 2×2 readout baseline: eight-fold result

**Date:** 2026-10-11  
**Scope:** fixed, known baseline comparison only; no novel-method claim. No encoder retraining.

## Frozen protocol and execution

Used the remotely registered `research/READOUT_LOSS_FACTORIAL_PROTOCOL.json`, protocol SHA-256 `821a16048eb018fe1e8aa9d4d1db4c2f31954127dfb3f1ac707613642c2b0010`. It fixes eight leave-one-user diagnostic folds, six source users for fit, the next user for validation, the held user for final scoring, the existing quality exclusions (9,300 eligible rows), 128-dimensional frozen full8 features, and a 2×2 readout comparison: balanced/uniform record weights × one-hot ridge/multinomial logistic regression. Each arm selected one of three lambdas solely on its validation user. There was no held-user fit, tuning, or selection. All eight-fold models, selections, sample IDs, and predictions were persisted before held-user labels were used for scoring.

The synthetic-only cost screen used one thread, did not read real data, and passed: 0.382869 s for one 7,200×128 fit; preregistered 48-fit ×2-headroom estimate 36.755432 s (<3,600 s budget). The actual fixed run completed 96 fits and saved all eight folds. At final aggregation the runner raised `KeyError: 'mean_gain'`: the code expected a gate key absent from the registered protocol. The runner had already written all fold predictions and evaluation files. No fitting was repeated. The summary below was reconstructed only from those existing predictions and `evaluation.npz` files, applying the protocol's actual gate keys; this was post-hoc scoring, not model selection. No source file or protocol was changed.

Cloud artifacts: `/workspace/uwb-readout-loss-factorial-runs-20261011`; log: `/workspace/uwb-readout-loss-factorial-runs-20261011.log`; cost record: `/workspace/uwb-readout-loss-factorial-cost-20261011/COST.json`. The output has eight fold directories with predictions, models, selections, splits, and evaluation labels written after all predictions were frozen. Reconstructed artifact manifest covers 43 files; manifest SHA-256 `aab21fd03a20d3be89b23e6a42db9a6567e2cbf16f8ea6413561e03b8f826534`; summary SHA-256 `7191cb521c3054842b3a8a4719f03a87d1523b8e21e9d6c85a3f9e616168f2ce`.

## Results

| Readout | Mean held-user macro accuracy |
|---|---:|
| Balanced one-hot ridge | 94.897727% |
| Uniform one-hot ridge | 94.840909% |
| Balanced multinomial logistic | 94.884659% |
| Uniform multinomial logistic | 94.883712% |
| Frozen CNN decisions | 94.184280% |

Primary effect (balanced logistic minus balanced ridge): **−0.013068 percentage points**, 4/8 held users improved, worst fold −1.400000 pp. The preregistered gates required mean gain ≥+0.5 pp, ≥6/8 wins, and no fold below −1 pp. All three failed. Balanced logistic is +0.700379 pp over frozen CNN, meeting that one control threshold, but the comparison is not a new contribution and does not rescue the failed ridge-controlled gate.

Uniform-weight loss effect (logistic minus ridge): +0.042803 pp, 4/8 wins, worst −1.400000 pp. Weighting effects were small: balanced versus uniform ridge +0.056818 pp (1/8 win); balanced versus uniform logistic +0.000947 pp (1/8 win). The fixed factorial does not support a useful loss or weighting effect.

For reproducibility, per-fold values and selections are in the cloud summary artifact. Balanced-logistic per-user gain over balanced ridge (folds 1–8): +0.333333, −1.400000, +0.500000, +0.083333, 0, +0.545455, −0.083333, −0.083333 pp.

## Decision

This completes the preregistered known baseline comparison; the primary gate failed. Do not expand the lambda set, reweight, replace the head, or promote the control arm. It does not validate a novel method and does not satisfy the research goal. Preserve the result as a negative/neutral baseline and continue only with a distinct candidate that clears prior-art and feasibility gates.

No active training remains. The cloud is CPU-only; no acquisition hardware is exposed. No raw data were uploaded to GitHub; only this report and project memory/status are updated.
