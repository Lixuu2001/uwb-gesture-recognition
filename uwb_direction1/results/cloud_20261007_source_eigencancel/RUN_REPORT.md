# Source-only gated eigencancellation: failed source gate; fixed screen closed

Preregistration f786ffd5145f441de495845762c338ae9b1aebf0 was pushed and remote-read back before training-only timing and real classification. Protocol SHA25680cbe22a67ac43ad5b576caa4f53c1bc2f234613983f83bf3d78cd89f5e71d99.

| Arm | Mean source-user macro accuracy |
|---|---:|
|ridge|93.063131%|
|selective|93.027778%|
|all_views|92.536616%|
|random_selected|92.957071%|

- Primary vs ridge: -0.035354pp, 3/6 strict wins, worst -0.545455pp; passed=False.
- Primary vs all_views: +0.491162pp, 4/6 strict wins, worst -0.166667pp; passed=True.
- Primary vs random_selected: +0.070707pp, 4/6 strict wins, worst -0.416667pp; passed=False.

No eightfold expansion, rank/threshold/normalization/seed/lambda sweep or promotion of a control. Failure is limited to this frozen-encoder intervention, not all subspace filtering.

The prior mechanism diagnostic found selective empty-energy suppression on left/right. That energy result is distinct from classification and does not establish nuisance ground truth. The classifier does not hard-code the global six-user view result: each held fold learns selection using only its four training users. Inner leave-one-source-user checks train empty subspaces on the other three and use fixed first20/class source samples for the preregistered energy criteria. A radar qualifies on at least3/4 inner users; final empty basis is refitted on all four training users. No validation/held labels select a radar or rank. All sample IDs and internal gate rows are retained.

Each basis uses per-clip RMS-normalized uncentered range moments, equal user weights, and smallest rank explaining90% energy. Rank32 is a gate, not a truncation. The primary filters qualified radars, all_views ignores the gate, and random_selected uses the same chosen radars/ranks with fixed Gaussian-QR bases. All apply raw residual X-(XB)B.T in float64, cast float32, then recompute Hilbert increments, per-clip normalization, float16 rounding and fixed range pooling. Untouched views use the identical original cache. Query labels are unnecessary. Shared-view computation is only an optimization, checked against full-input functional inference.

All arms use the same frozen epoch24 averaged encoder and128-dimensional features, with own training-only user/class-balanced means/scales and ridge. Three lambdas0.001/0.01/0.1 are selected on the next validation user by macro accuracy, onehot MSE, then larger lambda. All selected predictions precede diagnostic scoring. Six historically reused development users3–8 are not independent confirmation. Full-eightfold strong ridge94.897727% belongs to a different protocol and is not compared directly to these six-source means. No CNN retraining or extra feature capacity.

Independent audit rebuilt18 per-user moments,72 inner and18 final eigensystems, and17040 inner clip-view residual checks. It reconstructed all127800 complete CNN forwards with independent SciPy Hilbert preprocessing and functional convolutions, then solved54 augmented-intercept ridge equations. Maximum projector error4.19109191796e-15, hidden error0, score error8.99280649946e-14; every candidate/selected argmax agrees. 63 artifact hashes, source roles/IDs, validation choices and gate arithmetic passed. sklearn cholesky independently checked54 candidate fits during production. Actual main1101.539453s; cost estimate2718.012585s under3600s budget. Audit elapsed943.848980s includes waiting for completed folds while overlapping production. After preregistration only auditor scheduling changed; fitting protocol/code unchanged. Final manifest verifies all per-fold snapshots.

Literature: [Liu et al.,2018](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-rsn.2017.0482) establishes eigensubspace clutter suppression and target-contamination concerns. Read scope and distinctions are in research/EMPTY_SUBSPACE_REVIEW.md and SOURCE_EIGENCANCEL_REVIEW.md. This range-only, uncentered, already-ClutterRemoved RF screen does not replicate airborne STAP or centered PCA. Source-only gate plus known projection is not an original contribution without comparative evidence and further prior-art checks. A positive energy proxy cannot compensate for a failed classification gate. The frozen encoder was learned on unfiltered observations, which limits interpretation of a negative result. Also, original per-clip/channel standardization is reapplied after filtering: raw energy attenuation need not remain as amplitude attenuation at the CNN input. The energy proxy therefore cannot predict classification benefit by itself. This observation does not authorize a post-result normalizer sweep. Supplemental primary-paper checks are recorded in research/SUBSPACE_PRIOR_ART_UPDATE_20261007.md, with precise reading scope; they change no experiment.

Reproduce after restoring data/prepared arrays and prior private checkpoint/feature/readout receipts, using fresh output directories:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python uwb_direction1/tools/test_source_eigencancel.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python uwb_direction1/tools/run_source_eigencancel.py --protocol uwb_direction1/research/SOURCE_EIGENCANCEL_PROTOCOL.json --out /workspace/eigencancel-replay-cost --cost-only
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python uwb_direction1/tools/run_source_eigencancel.py --protocol uwb_direction1/research/SOURCE_EIGENCANCEL_PROTOCOL.json --out /workspace/eigencancel-replay --cost /workspace/eigencancel-replay-cost/COST.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python uwb_direction1/tools/audit_source_eigencancel.py /workspace/eigencancel-replay
```

Basis/selection states, hidden arrays, fit coefficients, candidate scores, predictions, sample IDs, configurations and logs are retained locally and in the final private Drive receipt. Archived provenance can precede the final receipt. No raw arrays in Git;789-file source snapshot unchanged. Research goal remains unmet. No active encoder training or next experiment queue after completion.

私有备份已逐份下载复验：13块、77951207字节、100成员SHA全部一致，元数据shared=false且仅owner；未更改共享权限。

Post-backup source-context check: the2026 Frontiers paper was already cited in original/project/experiments/p1_20260905/FIRST_PART_REPORT.md:118 and original/project/proposal_v2/31_第一部分新路线_用户校准与多雷达预算识别_调研方案.md:151, including the split mismatch. This turn adds a primary-text reading receipt, not a new prior-art discovery or new research route. Immutable private provenance predates this context paragraph and the final backup receipt; no code, protocol or result changed.
