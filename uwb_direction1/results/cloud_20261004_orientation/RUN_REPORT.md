# Local range–time orientation: real structure, failed recognition mechanism

UTC 2026-10-04T17:22:43.889993+00:00. **Source image feasibility passed; classification continuation gate failed. The requested effective original contribution is not established.** No outer-eight-fold expansion or neural training was launched.

## Evidence and prospective sequence

Diagnostic code/protocol commit `dc3def9defa2d757239cf02589e70976bfe0f4aa` was pushed and read back before cost measurement/real metrics. Reused exactly568 previous source-only stratified IDs:8 per available user3–8/class, existing user6/class3 exclusion. No user1/2 signals. Envelope=abs(Hilbert(realRF,range)),3bin range averaging to63, sqrt(per-image-max normalization). This analytic envelope is not hardwareIQ. Separate2D Sobel time/range gradients, Gaussian sigma2 over their products, fixed126interior grid points. Eligible trace>=original75th percentile and>1e-12; coherence>=.6 and obliqueness>=.3. Compare original versus sampleID-seeded shuffled90frames, shared acrossviews; same original trace threshold, separate eligible denominators/coverage reported. Coordinate scale is fixed and not calibrated speed.

| View | Original usable fraction | Shuffled | Passing users |
| --- | ---: | ---: | ---: |
| Left | 28.671480% | 4.214844% | 6/6 |
| Top | 30.684777% | 3.369426% | 6/6 |
| Right | 32.253196% | 3.504253% | 6/6 |

All18user/view gates passed (>=20% original usable texture and>=10pp excess; require4/6users in2/3views). Cost estimate5.917s with2×margin, extraction3.325s. Independent audit recomputed all568clips,429408tensor locations/1288224components from independent FFT, separable correlation and symmetric eigendecomposition; max tensor error 6.939e-17;18statistics and decision verified. **This is internal image structure, not correspondence/physical hand motion, classification accuracy or novelty.** Permutation changes temporal spectra/eligible coverage as well as locality.

Only after this pass, separate readout protocol `ead63d58813c328950378ccc084f89772c79f704` was pushed/read back before classification cost or scores. See both frozen research/ORIENTATION_*PROTOCOL.json files. No threshold or sample-grid changes after seeing outcomes.

## Recognition controls: six reused development folds

Full7100quality clips from sourceusers3–8; each round four trainusers, next-user validation, one held source diagnostic. Source-only balanced ridge with fixedlambda .001/.01/.1 selected by max validationmacro/minMSE/largerlambda. No held-label fit or selection. Same frozen128dim source epoch24CNN hidden vectors as the previous energy-area study, SHA-checked with identical sampleIDs/roles. Threeviews×2time×3range cells×9cyclic linearly interpolated orientation bins=162descriptordims; each augmented readout290dims. Per-cell trace×coherence votes normalized by eligible trace sum. Primary coherence/obliqueness gate removes unstable/axial positions; all-orientation control keeps all eligible positions; sign_null independently flips crossproduct sign while retaining coherence/angle magnitude; cell_null permutes sixhistogram cells per sample/view while retaining globalhistogram. Same fixed dimension and ridge selection across four arms. Baseline CNNreadout/tail predictions have identicalIDs and verified provenance, copied after selection.24selected readouts/72ridge fits; no encoder training.

| Arm | Mean user macro accuracy |
| --- | ---: |
| all | 93.364899% |
| coherent | 93.289141% |
| sign_null | 93.049242% |
| cell_null | 93.339646% |
| cnn_readout | 93.063131% |
| tail_cnn | 92.722222% |

| Primary versus | Mean gain (pp) | Wins | Worst user gain (pp) | Pass |
| --- | ---: | ---: | ---: | --- |
| all | -0.075758 | 2/6 | -0.583333 | False |
| sign_null | +0.239899 | 4/6 | -0.333333 | False |
| cell_null | -0.050505 | 3/6 | -0.583333 | False |
| cnn_readout | +0.226010 | 3/6 | -0.166667 | False |
| tail_cnn | +0.566919 | 4/6 | -0.250000 | True |

Gate required primary+0.25pp vs each matched descriptor control,+0.5pp vs both CNN comparators; each comparison >=4/6wins and no paireddecline>1pp. Joint gate failed. Improvement over tail alone is insufficient; the stronger CNNreadout and matched controls prevent attributing an effect to coherent signed/located texture. **Do not promote secondary all-orientation or change the primary after evaluation.** No bin/cell/gating/seed/regularization search or eight-fold expansion follows this result. This small fixed readout failure does not disprove all HOG/tensor or end-to-end methods.

## Audit, compute and reproduction

Training-only cost estimate with2×margin 80.523s <1800s. Actual descriptor extraction/fitting/save 40.503s. Readout audit independently recomputed all7100raw-image descriptors: 2683800tensor locations and 4600800descriptor values, maximum tensor error 3.469e-17, descriptor error 3.497e-15. All72sklearn Ridge refits and24selected models/validation choices/heldpredictions reproduced, maximum prediction error 1.168e-13; 34artifactSHA, IDs/roles/baselineprovenance/metrics/gates checked. Audit 99.940s. FrozenCNNhidden vectors reused with SHA/provenance, **not independently re-forwarded in this turn**; earlier checkpoint audits remain referenced.

CLI tools/run_orientation_screen.py and tools/run_orientation_readout.py expose data/IDs/protocol/output paths, and the latter screen/reference/prepared/cost-only/cost. Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. Cloud folders `/workspace/uwb-orientation-screen-20261004`, `/workspace/uwb-orientation-readout-20261004`, and `/workspace/uwb-orientation-readout-cost-20261004`; logs have matching prefixes plus audit. Audit scripts use these recorded paths. Inputs/protocol/corehashes and software versions in IDENTITY records. All sampling and source-model choices are deterministic; outputs retained rather than overwritten.

Private backup receipt verifies19 owner-only chunks,133,841,531compressed bytes and102archive members by authenticated re-download plus chunk/archive/member SHA256. It contains: diagnostic tensors/permutations/IDs, all7100 tensors/descriptors,24selected readout checkpoints, all candidate validation/final predictions, configs/code/logs/audits, sourceCNNhidden vectors and comparator provenance. Existing originalRF/prepared data and fullCNNcheckpoints remain in prior private backups, referenced by their receipts. No raw arrays uploaded toGit;789original snapshot unchanged.

## Literature and limitations

Read scikit-image HOG explanatory source (Dalal–Triggs CVPR2005 and Lowe2004 references) and v0.25.2 structure_tensor docstring/implementation. **Original papers were not acquired/read this turn.** Bounded repository searches are not exhaustive novelty searches. Gradient orientation histograms, structure tensors and coherence gating are known; their combination with aCNN is not an original contribution by itself. No calibratedvelocity/3Dtrajectory/sensor-synchronization or all-scaleinvariance claim. These six sourceusers have already been repeatedly researched: scores are exploratory development findings, not an untouched external test or eight-fold confirmation.

No next experiment or training is active at completion; automatic continuation remains enabled because the usergoal is unmet. Next work needs a distinct evidence-based mechanism and prior-art check, not another gate/scale/bin grid for this failed readout.
