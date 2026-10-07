# Known temporal scattering baseline: effect and ablation gates failed

Preregistered and pushed before real fitting: `ba6199707838f90115aeb1131091fc13a9d5fb60`. Protocol SHA256 `90af453a4640dad230ae7a1c9c6e3f19e4439fd371d71e4a2ae8a058e7d19224`.

| Arm | Mean source-user macro accuracy |
|---|---:|
|ridge|93.063131%|
|first|93.434343%|
|second|93.398990%|
|shuffled|93.544192%|
|standalone|85.439394%|

- Primary vs ridge: +0.335859pp, 5/6 wins, worst -0.818182pp; passed=False.
- Primary vs first: -0.035354pp, 3/6 wins, worst -0.545455pp; passed=False.
- Primary vs shuffled: -0.145202pp, 2/6 wins, worst -0.454545pp; passed=False.

Second-order scattering does not reach the fixed +0.5pp gain over ridge and loses on average to first-order and shuffled controls, failing both +0.25pp ablation requirements. Close this fixed J/Q/T, range-band, temporal-pool, normalization, seed and readout configuration. No eightfold expansion, grid search or control promotion. This is evidence against this particular second-order complement, not every scattering representation. standalone is a descriptive known-method baseline, not a gate or an alternative selected after the results.

Six repeatedly reused development users 3–8; each fold uses four training users, the next user for validation (8 wraps to 3), and the held diagnostic user. Existing quality policy excludes user6 class3, leaving7100 distinct records; users1/2 are not used. Same128dim frozen epoch24 features and ridge as prior source screens. The previous full eightfold94.897727% ridge uses a different protocol and is not directly compared. Extensive development-user reuse prevents treating these as fresh external test results.

Descriptor: divide raw RF by per-clip/view RMS (floor1e-12), Hilbert envelope along189fasttime samples, average each21range samples into9bands, log1p, yielding27 independent90frame temporal series. Fixed Kymatio0.3.0 Scattering1D J3/Q1/T8/max_order2/oversampling0 gives8paths x11positions. Pool positions into disjoint4/4/3 means. Full648features include orders0/1/2; first405 include only0/1. Concatenate the existing128features except for standalone. Combined dimensions are533/776/776/648. This loses fine range/phase information; it is not calibrated IQ, Doppler, physical viewpoint invariance, or exact speed invariance.

shuffled retains order0/1 and permutes only the11positions of second-order paths before pooling, with SeedSequence[20261007,globalID] shared across channels/paths/radars. Same width as primary; prepool multiset preserved, but pooling changes variance. First-order has lower dimension, so is not a matched-capacity control.

All arms use training-only equal-user/equal-present-class weighted standardization and an unpenalized intercept. Select among fixed ridge lambdas .001/.01/.1 using next-user validation macro, then onehot MSE, then larger lambda. All72fits,24selected readouts, candidate validation predictions and all six folds' diagnostic predictions were saved before any diagnostic scoring. Global labels are available for source/validation eligibility and quality masks, not technically sealed; diagnostic labels are never used in fitting/selection. Cached baseline common−.5score offset is checked. All candidate coefficients, sample IDs, splits, choices and predictions are preserved; descriptor hashes permit regeneration without storing large feature arrays in Git.

Synthetic numerical checks passed before real data use: independent SciPy Hilbert/range pooling and direct circular convolution with explicit time decimation, batch independence, shuffle multiset, zero input, clip gain normalization. Maximum synthetic descriptor discrepancy4.440892098500626e-16. Cost16synthetic clips=0.063219s, fit family=0.203942s, conservative estimate=65.896405s under900s. Actual real-run timed body=28.755208s, excluding initial raw SHA verification; no encoder training.

Independent replay checks 7100 distinct clips/28400 clip-arm outputs and all72 weighted ridge systems with separate augmented-intercept normal equations. All41 artifact hashes, roles, weights, coefficients, validation predictions/choices, diagnostic predictions/labels and gate arithmetic pass; max descriptor error=1.5543122e-15, max score error=6.0473848e-13; all argmax equal. Audit=731.184667s. Numerical reference uses the same pinned official filters, with a distinct time-domain convolution/decimation computation; it is not an independently designed filter bank. standalone duplicates the full descriptor, so28400 arm outputs are not28400 distinct clips.

Direct prior: Al-Jumaily and Khushaba2023, *Deep Hand Gesture Recognition: A Wavelet Scattering Alternative to Convolutional Networks*, SSP438–442, DOI10.1109/SSP53291.2023.10208011, [institutional abstract](https://ro.ecu.edu.au/ecuworks2022-2026/3131/). This already studies eight-user/twelve-gesture UWB WST and reports approximately95%. Only abstract/metadata and author bibliography were verified; full split, radar combination, preprocessing and parameters remain unknown. Our configuration is not a replication and their95% is not compared with our source/LOSO numbers. Andén and Mallat2014 [Deep Scattering Spectrum](https://arxiv.org/html/1304.6763v2), DOI10.1109/TSP.2014.2326991, supports the known modulation cascade and qualified small-warp stability, not these radar results. Read scope is in research/TEMPORAL_SCATTERING_REVIEW.md. No original contribution has been validated.

Kymatio top-level NumPy import fails on SciPy1.17 because its unrelated3D module imports removed sph_harm. Use the existing direct1D frontend, as in checked-in code; no library patch, dependency downgrade or filter change. Package Python hashes, environment versions and install log are preserved.

Reproduce with prior raw/encoder/feature/readout artifacts restored, Kymatio0.3.0 installed, and fresh output paths:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python uwb_direction1/tools/test_temporal_scattering.py
python uwb_direction1/tools/run_temporal_scattering.py --protocol uwb_direction1/research/TEMPORAL_SCATTERING_PROTOCOL.json --out /workspace/scattering-replay-cost --cost-only
python uwb_direction1/tools/run_temporal_scattering.py --protocol uwb_direction1/research/TEMPORAL_SCATTERING_PROTOCOL.json --out /workspace/scattering-replay --cost /workspace/scattering-replay-cost/COST.json
python uwb_direction1/tools/audit_temporal_scattering.py /workspace/scattering-replay
```

Raw arrays remain outside Git; original789snapshot unchanged. Private backup includes all readout checkpoints, candidate predictions, IDs, configuration, hashes, logs, code and prior receipts linking encoder/features/readout provenance. Immutable archive predates final receipt append. Final Drive receipt is authoritative. No background training or next queue. Goal unmet; continuation remains enabled. A next experiment requires a distinct evidence-based hypothesis and preregistration.

私有备份已逐份下载复验：3块、19393766字节、77成员SHA全部一致，元数据shared=false且仅owner；未更改共享权限。
