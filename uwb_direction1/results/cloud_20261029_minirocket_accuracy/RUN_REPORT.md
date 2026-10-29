# Full author MiniRocket accuracy experiment — 2026-10-29

The latest user instruction authorizes ongoing paper-backed accuracy experiments even for established methods, while assessing original contribution separately. The prior stop-before-experiment novelty rule is superseded for such experiments. No failed historical configuration was rerun: the 20261004 experiment independently modified a small504 response bank; this run uses unchanged author multivariate MiniRocket with the default actual9996 features.

## Completed result

| Held development user | MiniRocket | Blockmean raw Ridge |
|---|---:|---:|
| 1 | 85.3333% | 14.2500% |
| 2 | 84.8000% | 26.7000% |
| 3 | 80.3333% | 16.7500% |
| 4 | 78.3333% | 21.0000% |
| 5 | 83.5833% | 27.7500% |
| 6 | 58.8182% | 19.6364% |
| User mean | 78.5336% | 21.0144% |

Metric: recall macro averaged over ground-truth classes present for each held user, then equally over six users. CPU 318.045 seconds; wall 316.974 seconds, within1800 CPU/wall bounds. All six folds and twelve classifiers completed. Audit passed93 checks: all saved hashes, config/code identity, disjoint train/validation/diagnostic IDs, all6900 role rows each fold, prediction ID alignment, all twelve metric replays, transform and coefficient shapes, all six held users, and run budget. One premature audit invocation occurred before SUMMARY.json existed and failed with FileNotFoundError; the audit was rerun after process completion. No fit or prediction computation was repeated.

## Interpretation

The preregistered inexpensive control gate passes; it licenses a matched strong CNN comparison only. The raw-map linear control is weak (21.01%), so the large difference is not evidence of improvement over an existing CNN. The source diagnostic score78.53% must not be directly compared with historical eightfold93.4417%, whose models/users/training roles differ. No eightfold run, accuracy improvement over strong CNN, significance, independent validation, or research novelty is claimed. Users1–6 already have development history; no users7–8 gesture labels were indexed for fitting, selection or evaluation. Whole-file digests are integrity checks, not label-based selection.

## Protocol and reproduction

Four source users fit, next source user reserved validation, one held source user diagnostic; no validation tuning occurred. Author transform bias fitting and readout scaler/model use training rows only. Full3×189 fast-time channels,90 slow-time samples; axes confirmed against archived vendor/config.py and uwb_osr/data.py. No dimensional compression in the candidate. Seed20261029 resets both NumPy and Numba RNGs. Fixed alpha1 balanced-user/class Ridge differs from the author's RidgeCV; this is declared baseline adaptation, not exact end-to-end author reproduction. See PROTOCOL.json, run.py, README.md, REPRODUCIBILITY.json, run01/identity.json and AUDIT.json. Local artifacts: /workspace/uwb-accuracy-loop-20261029/run01; six transform checkpoints, twelve readout models, source IDs, per-role IDs, predictions and logs retained privately. No raw arrays/models/row IDs uploaded to Git.

Literature: Dempster, Schmidt, Webb, KDD2021, DOI10.1145/3447548.3467231, https://arxiv.org/abs/2012.08791 ; unchanged author code from https://github.com/angus924/minirocket at commit specified in protocol. Read author README, complete multivariate code and primary manuscript abstract/introduction/related work/start of method through kernel construction (pages0–2). Full-paper review is not claimed.

CPU PyTorch2.15.0+cpu installed successfully from official CPU index, CUDA unavailable. Original789-file source remains untouched. The automation prompt now encodes the user's accuracy-first loop while preserving the hourly schedule, privacy, no test-label tuning, and meaningful-change-only notifications.

## Next continuation

Preserve this exact MiniRocket configuration and all negatives. Do not tune bank/seed/alpha to rescue it. Complete the HYDRA primary-method/multivariate code review and freeze one default configuration, compatibility/cost gate, then a matched source-only run using the same roles. CPU torch is now available for HYDRA and the matched strong CNN baseline; install is not a running job. See NEXT_METHOD.md. Continue practical literature-to-implementation-to-measured-result cycles without a repeated confirmation request. Only methods clearing matched strong-baseline evidence merit further eightfold confirmation, and contribution evidence remains separate.
