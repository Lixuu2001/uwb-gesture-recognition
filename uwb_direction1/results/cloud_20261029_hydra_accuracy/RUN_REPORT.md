# HYDRA default multivariate source experiment — 2026-10-29

## Result and disposition

The fixed author experimental multivariate HYDRA configuration fails its registered source-only gate against matched MiniRocket. Close this exact raw567channel/defaultg64k8/seed20261029/alpha1 configuration; do not sweep its bank/seed/alpha to rescue it. This does not rule out all HYDRA representations or radar models. No accuracy gain over an existing CNN or original contribution has been established.

| Held development user | HYDRA completed linear solve | HYDRA truncated LSQR200 | Matched MiniRocket |
|---|---:|---:|---:|
| 1 | 63.5833% | 70.0000% | 85.3333% |
| 2 | 64.6000% | 72.1000% | 84.8000% |
| 3 | 55.7500% | 62.6667% | 80.3333% |
| 4 | 54.0000% | 64.7500% | 78.3333% |
| 5 | 61.6667% | 71.5833% | 83.5833% |
| 6 | 46.6364% | 55.4545% | 58.8182% |
| User mean | 57.7061% | 66.0924% | 78.5336% |

Primary completed-solve difference vs MiniRocket: -20.8275 percentage points;0/6 user wins; gate false. These are six repeatedly developed source users, four train users/next validation/held diagnostic. Labels of users7–8 were not used. Macro recall includes true classes present per held user, equally averaged over users. Historical eightfold93.4417% has different roles/models and is not a matched comparator.

## Numerical engineering correction

All72 original class-specific LSQR solves hit the fixed200 iteration cap. The original artifact/split/metric audit passed99 checks, but it did not establish optimizer convergence. Preserve the original66.0924% and its logs; do not infer a fully optimized method conclusion from it. Before correction, HYDRA_NUMERICAL_AMENDMENT.json froze completion of the same alpha1 weighted Ridge objective with float64 direct Cholesky. The saved4096features, randomweights/channelchoices, source roles, training-only author SparseScaler, labels, seed, and regularization were reused. No feature resampling, new seed, grid or accuracy-based selection occurred. This deterministic solve was restarted from the same fixed objective, not a warm-start claim.

Weighted centered normal-equation residuals were checked BEFORE each corrected diagnostic score; maximum relative residual 7.1e-15, below1e-6. The corrected result is57.7061%, which is retained even though lower than truncated66.0924%. The original audit passes99 checks and the completed-solve audit passes90 checks; corrected audit also recomputes all six training scaler statistics and verifies exact comparator prediction alignment.

Cumulative analysis budget: 121.137CPU seconds / 117.277wall seconds, below895CPU/900wall. Parent timeout, processCPU rlimit and singlethread constraints enforced; no timeout. Compatibility checked deterministic weights/channel IDs, feature finiteness, batch invariance and row independence.

## Sources, code and artifacts

Dempster, Schmidt, Webb, HYDRA, DMKD37(2023)1779–1805; https://arxiv.org/abs/2203.13652 ; https://github.com/angus924/hydra . Read primary abstract/introduction/related work/method overview/defaults and full author multivariate/SparseScaler code. Fullpaper not claimed. Author explicitly marks multivariate implementation untested/experimental; this experiment checks its execution on this dataset, not equivalence to published univariate benchmarks. Author code imported unchanged at pinned144bb7aa3186654042f00f36b078370c620acb06; Ridge readout is a declared adaptation.

Protocol and source: uwb_direction1/research/hydra_accuracy_20261029/. Private local artifacts /workspace/uwb-accuracy-loop-20261029/hydra_run01/ and hydra_readout_completion/: random transform tensors W/I (not an empty module.state_dict), features/IDs, six original and six corrected readouts, per-role IDs and predictions, receipts, model/file hashes. Raw arrays and row-level/model arrays were not committed. No active training remains.

## Next continuous experiment

Prioritize a paper-backed strong CNN on the same UWB-Gestures corpus: Xue etal2026 MV-GLFNet, https://doi.org/10.3389/fnins.2026.1864448 . Under the user's latest instruction, same-corpus prior is a reason to test applicability, while preventing false novelty claims. Current CPU torch2.15.0+cpu enables implementation. Next review original method tables/code availability, freeze a paper-faithful implementation and explicitly list unresolved details, fixed source roles/seed/training schedule and baseline, run gradient and CPU-cost checks, then perform real source-only training if within an explicit budget. Do not directly compare the paper's six-train/two-test98.29% with project LOSO. Complete matchedstrongCNN evidence before any claimed improvement or eightfold expansion. Continue papers→implementation→real measured experiments; do not return to novelty-only no-go loops.
