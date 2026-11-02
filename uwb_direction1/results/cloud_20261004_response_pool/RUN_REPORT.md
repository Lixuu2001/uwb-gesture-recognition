# Small temporal-response pooling: source development screen

UTC 2026-10-04T16:13:59.335133+00:00. Preregistered implementation/protocol commit `77e9cf5b4e86dacd28c39c6c5f325f428a724a08` was pushed and read back before source-only cost measurement and real scoring. Protocol SHA256 `4faf88318d982438ac6faa62f0cb9f83f24203132d2058d31e70922ae084ee23`.

**Joint continuation gate: False. User's effective original-contribution goal remains unfulfilled.** No eight-fold experiment was launched from this screen. This is an independently implemented, substantially modified MiniRocket/MultiRocket-inspired baseline, not a new invention or exact author reproduction. Author READMEs and bounded multivariate code were read; full papers were not acquired/read. See research/RESPONSE_POOL_REVIEW.md and receipt.

## Fixed experiment

7100 quality-retained clips from source users3–8 only; six rounds, four users fit/one next-user validation/one held development-user diagnostic. Same role/sample IDs as earlier CNN runs. Users1/2 signals are not accessed. Legacy prepared channels0/2, each radar separately, raw and temporal-difference signals.14 fixed length9 kernels and8 averaged range bins per kernel, dilations1/3/9, valid responses;504response streams. Bias quantiles calibrated on one randomly fixed **training** clip per stream, never validation/held clips. Float32 feature storage followed by float64 weighted ridge.

Primary multi_pool: proportion, mean positive excess, normalized mean positive location, normalized exact longest run (2016dimensions). Controls: single-threshold PPV504; four-threshold PPV2016; same four statistics with deterministic post-convolution time-position permutation2016. The latter preserves proportion and positive-excess mean, tests pooling response order, and is not a control for convolution's own temporal dependence. Weighted mean/scale/ridge fit only four training users; three fixed lambdas selected on validation maxmacro/minMSE/larger-lambda.24 selected readouts,72fits. Historical frozen CNN readout and tail predictions copied only after all new selections, with identical IDs/labels and SHA-verified provenance; no neural network retraining.

## Equal-user mean macro accuracy

| Arm | Accuracy |
| --- | ---: |
| ppv_single | 84.063131% |
| ppv_four | 88.001263% |
| multi_pool | 87.238636% |
| temporal_null | 86.522727% |
| cnn_readout | 93.063131% |
| tail_cnn | 92.722222% |

| Primary versus | Mean gain (pp) | Wins | Worst paired gain (pp) | Pass |
| --- | ---: | ---: | ---: | --- |
| ppv_single | +3.175505 | 6/6 | +0.916667 | True |
| ppv_four | -0.762626 | 2/6 | -3.750000 | False |
| temporal_null | +0.715909 | 4/6 | -1.454545 | False |
| cnn_readout | -5.824495 | 0/6 | -10.000000 | False |
| tail_cnn | -5.483586 | 0/6 | -10.500000 | False |

The preregistered gate required +0.5pp vs singlePPV, +0.25pp vs fourPPV and temporalnull, nonnegative gains vs both CNN comparators; each comparison required ≥4/6wins and no user loss worse than1pp. No secondary arm replaces the primary. These six users were repeatedly used in prior research: these are exploratory development scores, not an untouched external test, not eight-fold confirmation, and not significance or novelty evidence. A small fixed bank cannot rule out full-sized author MultiRocket; no adaptive bank/channel/dilation/threshold/pooling/seed/lambda search is warranted from this failed screen.

## Validation and timing

Training-only cost estimate including2×margin 470.127s <1800s. Actual extraction/fitting/save 244.117s. Independent audit 47.955s verified 45artifact hashes, all3024calibration curves using scipy correlation, fixed bank/roles/source-only bias IDs, six data clips per fold (two from each role) with independent scalar statistics, all72sklearn Ridge refits/validation candidates and24selected diagnostics/model coefficients, reference predictions, scores and decision. Max bias error 0.000e+00, feature spot error 0.000e+00, prediction error 2.529e-13. **All7100 feature rows were not independently recomputed**; scope is precisely the listed spots plus full fit/artifact checks. Synthetic tests cover3948scalar convolution points, equality/empty/end runs, batch invariance, null marginal preservation and calibration isolation.

## Reproduction and artifacts

Cloud run `/workspace/uwb-response-pool-source-20261004`, cost `/workspace/uwb-response-pool-cost-20261004`; tools/run_response_pool.py CLI accepts data/prepared/reference/protocol/out and cost-only/cost modes. Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; all dependencies installed in the recorded cloud Python. Prepared input and restored raw/source label hashes in IDENTITY.json. Reference is `/workspace/uwb-energy-area-source-20261004`, whose manifest SHA is fixed in protocol. Each held folder keeps bank/biases/calibrationIDs, all feature matrices, split/selection, all candidate validation/final predictions and selected linear model checkpoints. Logs retained under `/workspace/uwb-response-pool-{cost,source,audit}-20261004.log`.

Private backup receipt verifies5 owner-only chunks,30,488,217compressed bytes and85archive members by authenticated re-download plus chunk/archive/member SHA256. The fitted-state bundle contains: all24readouts,6calibrated banks, IDs, predictions, configs/code/logs/audit, copied comparator predictions/models and prior private CNN-backup receipt. Deterministic feature matrices remain in the cloud; regenerate them from saved banks/biases/IDs and verified prepared inputs with response_pool.extract, casting float32. Original raw data remain in the existing private transfer, not duplicated in this backup or Git. Git contains only small code/records/results. Original789-file snapshot unchanged.

No training remains active after completion; automatic continuation stays enabled because the goal is not met. Next continuation must examine a distinct, evidence-based radar mechanism and prior art, not reopen this screen's parameter grid or advertise the known pooling operators as innovation.
