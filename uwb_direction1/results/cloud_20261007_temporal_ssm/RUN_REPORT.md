# Temporal self-similarity screen: effect gate failed

Protocol pushed before real fitting: `f384377e37f083f1925441c92f0b20194830d8a8`. SHA256 `f5f642998deed2fb748721bf13c03b52119dcdcff00896bcc4ec2bb74649177d`.

| Arm | Mean source-user macro accuracy |
|---|---:|
|ridge|93.063131%|
|ordered|93.242424%|
|lag_only|93.155303%|
|shuffled|93.199495%|

- Primary vs ridge: +0.179293pp, 4/6 wins, worst -0.250000pp; passed=False.
- Primary vs lag_only: +0.087121pp, 3/6 wins, worst -0.333333pp; passed=False.
- Primary vs shuffled: +0.042929pp, 3/6 wins, worst -0.416667pp; passed=False.

The small gain over ridge does not pass the fixed +0.5pp effect requirement; neither +0.25pp ablation margin passes. The ordered representation is not convincingly better than stationary lag statistics or per-record time permutation under this screen. Close this fixed block/normalization/seed/readout configuration: no eightfold expansion, parameter sweep or control promotion. This is not a claim that every temporal-relation model fails.

Six repeatedly reused development users3–8, four training users/next validation/held diagnostic; users1/2 excluded, user6class3 excluded by existing quality policy. Exactly7100 records and the same128dim frozen epoch24 features and strong ridge as prior screens. Existing eightfold94.897727% ridge uses a different protocol and is not directly compared. Extensive development-user reuse limits evidence; these are not fresh external test results.

For each clip/radar, compute float64 Hilbert envelope along189fasttime samples, remove each bin's90frame mean, divide each frame by its range-profile L2 norm floored at1e-12. Mean each5frames to18profiles, form18x18 cosine-profile Gram; pooled diagonal need not equal1. Upper triangle including diagonal gives171entries/radar,513total; append128hidden features. Pooling is algebraically equivalent to averaging5x5 blocks in full90x90 Gram, verified independently. Full-clip mean excludes causal claims. Low-energy frame normalization may amplify noise. Orthogonal invariance is only a profile-space algebraic fact, not proof of physical viewpoint or speed invariance; absolute spatial direction information can be lost.

lag_only replaces each matrix diagonal-offset with its mean. It has513 stored columns but54 independent values, so it is not equal intrinsic capacity. shuffled uses a separate deterministic18block permutation per global record ID, shared across three radars, applied to both matrix axes. It preserves Gram spectrum and entry multiset but removes a consistent absolute temporal layout. ordered and shuffled have equal513dimensions. No MI/RBF/kernel/bin/seed search was run.

All augmented arms use training-only equal-user/equal-present-class weighted standardization, intercept and ridge. Fixed lambda .001/.01/.1 is selected on next validation user by macro accuracy, then onehot MSE, then larger lambda. All54fits and all six folds' models, candidate validation scores and diagnostic predictions were persisted before any diagnostic labels were scored. Global label arrays are loaded for source/validation eligibility, so labels are not technically sealed, but they are not passed into diagnostic fitting/selection. Original baseline's cached common−.5score offset is explicitly checked. Save all candidate coefficient matrices, exact IDs/splits/selections, descriptor hashes and predictions; no duplicated large descriptor arrays are needed for regeneration.

Synthetic independent Hilbert/full90Gram, pooling, batch independence, reversal covariance, permutation eigenvalues/multisets, zero-input and profile-space orthogonal checks passed. CPU synthetic16clip cost0.036783s; fit-family0.135162s; conservative estimate37.510743s under900s budget. Real run16.849659s (raw SHA verification before timed body). No new CNN training.

Independent audit uses SciPy Hilbert and explicit90x90 Gram block averages, then separate unpenalized-intercept642x642 normal equations for all54ridge fits. 7100distinct clips/21300clip-arm descriptors checked; max descriptor difference1.6653345e-15; max score difference1.1363133e-13, all argmax identical. All41artifact SHA, source roles, weighting, all candidate coefficients, validation choice, predictions and gate arithmetic passed. Audit time50.161449s. The audit imports the independent reference from the synthetic test; it does not use the production Gram implementation to produce reference features.

Known prior art: He etal2014 [radar SSM paper](https://repository.tudelft.nl/file/File_20f97e3f-1d5f-41f5-92ba-8bebdcd22518), DOI10.1017/S1759078714000087; Junejo etal2008 [publisher abstract](https://link.springer.com/chapter/10.1007/978-3-540-88688-4_22). Read scope and distinctions documented in research/TEMPORAL_SSM_REVIEW.md. The present cosine envelope Gram screen is not a reproduction of mutual-information/aligned range-Doppler SSM and is not novel by combination. No verified original contribution has been obtained.

Reproduce with prior private encoder/feature/readout artifacts restored and fresh output paths:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python uwb_direction1/tools/test_temporal_ssm.py
python uwb_direction1/tools/run_temporal_ssm.py --protocol uwb_direction1/research/TEMPORAL_SSM_PROTOCOL.json --out /workspace/ssm-replay-cost --cost-only
python uwb_direction1/tools/run_temporal_ssm.py --protocol uwb_direction1/research/TEMPORAL_SSM_PROTOCOL.json --out /workspace/ssm-replay --cost /workspace/ssm-replay-cost/COST.json
python uwb_direction1/tools/audit_temporal_ssm.py /workspace/ssm-replay
```

Raw arrays remain outside Git; original789snapshot unchanged. Private backup includes all readout checkpoints/candidate predictions, IDs, configs, hashes, logs, code, and prior Drive receipts linking encoder/features/readout provenance. Final receipt is authoritative; immutable archive may precede final receipt append. No background training or next queue. Goal unmet; continuation remains enabled, with next experiment requiring a new evidence-based hypothesis and preregistration.

私有备份已逐份下载复验：2块、14243831字节、72成员SHA全部一致，元数据shared=false且仅owner；未更改共享权限。
