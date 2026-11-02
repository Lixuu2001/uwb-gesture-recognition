# Fixed-scale Huber source readout failed the effect gate

Preregistration pushed before fitting: `850e167d98c91502ba95be3d61259b6b74038cb8`. Protocol SHA256 `50779b3e880fa6776c5d00b9de15565a47c266f0e746a4acee5e8d956edd1968`.

| Arm | Mean source-user macro accuracy |
|---|---:|
| Full-source ridge |93.063131%|
| Fixed-scale Huber (primary) |92.934343%|
| One-step residual reweighting |92.902778%|
| Within-user/class shuffled influence |93.022727%|

Primary vs ridge −0.128788pp,2/6 strict wins,1tie,3losses, worst−0.5pp. Primary vs one-step +0.031566pp,3wins (below required+.25pp/4wins); vs shuffled−0.088384pp,1win. All comparisons fail. The fixed threshold/scale/lambda/seed route is closed: no eightfold extension or post-hoc parameter grid/control promotion. This does not rule out all robust losses. The previous user-deletion failure was not rerun or relabeled.

Same six reused development users3–8, four training/next validation/held diagnostic split, exact quality IDs and frozen128dim epoch24 averaged source CNN features as the earlier semantic-code study. Original full-source training mean/scale, class/user weights and selected lambda are inherited. Per output, source ridge residual MAD fixes the scale (normal convention0.6744897501960817, floor1e−4), threshold1.35 times that scale. The convex objective sums source-weighted Huber losses plus lambda/2 times squared coefficient norm; intercept is unpenalized. L-BFGS-B starts original ridge. Acceptance requires infinity-norm gradient≤1e−7; no evaluation-based checkpoint/threshold selection. Source influence ratios min(1,delta/abs(residual)) are saved with sample IDs.

One-step control solves weighted ridge using initial ridge residual weights. Shuffled control permutes converged primary weights within source(user,trueclass) cells for each output with fixed SeedSequence([20261007,held,output,user,class]), then solves weighted ridge. Neither clipped weight set is renormalized; this preserves the objective penalty convention. Permutation preserves each cell's weight distribution/mass while breaking clip assignment; it is a conditional mechanistic null. The primary clipped16.20–20.83% of source clip/output residuals across folds, so failure was not simply zero activation. These are output-specific influences, not a single sample-removal probability. No claim of mislabeled clips or causal isolation of all nuisance factors.

L-BFGS-B completed72 scalar-output fits and controls144 linear-output fits. Independent IRLS starting from zero reconstructed all72 primary fits (at most108 iterations); maximum objective difference8.50652e−14, maximum role-score difference2.74115e−6, and every class argmax matched. Independent normal equations checked144 control outputs. Maximum primary gradient8.74123e−8; all saved-model predictions replayed bitwise. Audit verified51 artifact hashes, exact roles/IDs, inherited selection, fixed scales, permutation distributions, truth alignment, macro means and gate arithmetic. This is numerical verification, not independent external accuracy validation.

Synthetic CPU cost0.290541s, sixfold estimate with3x margin5.229742s; actual fitting/scoring/checking/I/O16.296312s, still below600s budget. Synthetic timing underestimated real-conditioned optimization and I/O. No new encoder training or background queue. Source/validation/diagnostic predictions and parameters were persisted before evaluation. Global label array is loaded for source indexing; held labels are not used for fitting or choosing weights but are not technically sealed from memory. All users have been evaluated previously, so this remains an exploratory source screen.

Literature/math review is in `research/HUBER_READOUT_REVIEW.md`: [SciPy's Huber formula](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.huber.html), [sklearn1.8 HuberRegressor](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.HuberRegressor.html), and its versioned loss/gradient source. Huber1964 DOI10.1214/aoms/1177703732 is cited by SciPy; its original full text was not accessed. Our fixed-scale objective differs from sklearn's optimized-scale estimator. Bounded residual derivative does not bound influence of arbitrary feature leverage or establish UWB generalization. Huber/IRLS are established techniques, and no original research contribution is claimed.

Reproduction requires the private prior encoder/feature/readout backups and fresh outputs:

```sh
python uwb_direction1/tools/test_huber_readout.py
python uwb_direction1/tools/run_huber_readout.py --protocol uwb_direction1/research/HUBER_READOUT_PROTOCOL.json --out /workspace/huber-replay-cost --cost-only
python uwb_direction1/tools/run_huber_readout.py --protocol uwb_direction1/research/HUBER_READOUT_PROTOCOL.json --out /workspace/huber-replay --cost /workspace/huber-replay-cost/COST.json
python uwb_direction1/tools/audit_huber_readout.py /workspace/huber-replay
```

`AUDIT_IDENTITY.json` records installed versions. Prior energy-area/semantic-code private receipts recover source encoders/features/ridge; config pins their manifests. Current model coefficients, source IDs/scales/influences, scores, optimizer traces, logs and code are privately archived. No large arrays in Git and original789-source snapshot unchanged. Final Drive receipt is authoritative; archived report can precede final receipt. The94.897727% eightfold ridge belongs to a different protocol, not compared to these six-source means. Research goal remains unmet and automatic continuation remains enabled.

私有备份已逐份下载复验：7块、6842683字节、80成员SHA全部一致，元数据shared=false且仅owner；未更改共享权限。
