# Full-eightfold frozen readout loss and weighting comparison

## Result

This run completed the preregistered, fixed-feature eightfold comparison. It is a known classifier-readout baseline, **not a novel method**. The independent replay refit all 96 candidate models (4 arms × 3 regularization values × 8 held users) with scikit-learn. Coefficients, validation/held predictions, selected regularization, and predicted classes matched the stored run within the audit tolerances. All eight user folds and the 42 source-run artifacts passed hash/role checks. No encoder was trained.

| Readout arm | Eightfold macro accuracy |
|---|---:|
| Balanced-weight ridge | 94.897727% |
| Uniform-weight ridge | 94.840909% |
| Balanced-weight multinomial logistic | 94.884659% |
| Uniform-weight multinomial logistic | 94.883712% |
| Frozen CNN decisions | 94.184280% |

The preregistered primary comparison, balanced logistic versus balanced ridge, was **-0.013068 percentage points** (4/8 held-user wins; worst held-user change -1.400000 pp). Its thresholds were +0.5 pp mean, at least 6/8 wins, and no held-user drop below −1 pp. The gate therefore failed. Balanced logistic was +0.700379 pp over the frozen CNN on average (4/8 wins), but this readout baseline does not establish a contribution and is not the primary gate.

The runner completed all eight folds and persisted the predictions, selected models, splits, and hashes, then failed while assembling its final summary because of a post-fit gate-key typo. No model fitting was rerun to recover the report: the stored outputs were independently refit and rescored by the audit script. The runner and its original protocol remain unchanged so the completed run is traceable.

## Frozen protocol and scope

- Protocol SHA-256: `821a16048eb018fe1e8aa9d4d1db4c2f31954127dfb3f1ac707613642c2b0010` (fixed before real-data fitting).
- Runner commit: `0f6f10a8659726fd2216210297d4cd6ad522ae01`; runner SHA-256: `af3d3e47901702122650b10be12bc9d4c3c557e33d137b5213395ff5a0bbd5d0`.
- Data: 9,300 quality-filtered rows from the restored UWB-Gestures collection; known duplicate/mismatched labels excluded as declared in the protocol.
- Same frozen 128-dimensional full8 CNN representation and historical eight-user leave-one-user-out design; six source users train, next user validates, held user is diagnostic.
- The validation user selects one of three fixed regularization values per arm using the preregistered macro-accuracy / probability-MSE / larger-regularization tie-break. Held labels are used only for final scoring.
- Balanced-vs-uniform sample weighting and squared-error-vs-multinomial-logistic loss form a 2×2 baseline comparison. No architecture, features, grid, or encoder training was added.
- Cost-only estimate: 48.32 seconds; the real stored run reports 48.36 seconds for fitting, scoring, and audit replay.

## Literature and interpretation

Frozen-feature L2 logistic regression is established prior art: Kumar et al. (ICLR 2022) use logistic regression on frozen representations with validation-selected regularization ([paper](https://arxiv.org/abs/2202.10054)). The project source also already contains multinomial logistic readouts. The scikit-learn documentation describes multinomial logistic regression and sample-weighted fitting ([documentation](https://scikit-learn.org/stable/modules/linear_model.html)). This experiment fills a same-checkpoint, same-split controlled comparison; it is not a claim that logistic regression or class/user balancing is new. Eight historical users are reused, so this is not external-user confirmation.

**Conclusion:** the balanced logistic readout did not outperform the balanced ridge readout under the frozen gate. This candidate is closed; do not retune, expand seeds, or promote a secondary arm. The broader goal—at least one valid, reproducible research contribution—remains unmet.

## Reproduction artifacts

`PROTOCOL.json`, `IDENTITY.json`, `SOURCE_FILES.json`, and the source `RUN.log` identify the frozen run. `AUDIT.json` contains fold scores and effects; `AUDIT.log` records the independent replay. The private backup archive (`uwb-readout-loss-factorial-20261008.tar.gz`, Drive ID `1BndNlV2SiX_QVrrwjpnMKcex8o2LoULK`) includes all 42 source-run artifacts (splits/sample IDs, predictions, selected coefficients and validation choices, evaluations, file hashes), the protocol, cost record, and logs. Its 15,492,503 bytes and SHA-256 `458c618445e691eb515a1bdd9443b03e65280cd9b0399f6901c602c40eac5a4a` were checked against an authenticated download; the 42 per-run artifacts and 9 report artifacts also passed member hashes. The Drive folder metadata remains owner-only (`shared=false`). Raw arrays/checkpoints remain in the existing cloud data/reference locations and are not added to Git.
