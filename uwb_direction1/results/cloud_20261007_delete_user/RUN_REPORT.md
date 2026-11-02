# Source-user deletion readout screen: fixed minimum rule failed

Preregistered and pushed before real fitting: `f9ec25f54c0be7b221dc80ecab0c7894fa343aa9`. Protocol SHA256 `1d55d7e3f53a58bcffa7fbe3f2caef06d9db17da0f45db1391cde5dbae8239da`.

| Arm | Mean source-user macro accuracy |
|---|---:|
| Full-source ridge |93.063131%|
| Delete-user minimum (primary) |93.049242%|
| Delete-user mean |93.174242%|
| Matched row-deletion minimum |93.049242%|
| Matched row-deletion mean |93.061869%|

The primary changes ridge by −0.013889 percentage points, with1/6 strict wins,2ties,3losses (worst −0.25pp). It also fails all three aggregation/deletion controls: vs domain_mean −0.125pp, row_min0pp, row_mean−0.012626pp. Equal aggregate accuracy to row_min does not imply identical predictions. Joint preregistered gate fails. Close this fixed deletion/minimum/normalization/seed/lambda configuration. No eightfold extension, no secondary-control promotion or post-hoc weighting/quantile/seed/lambda sweep. This result does not rule out all source ensembles.

Protocol uses six previously reused development users3–8: four source training users, next user for existing ridge validation selection, held source user for diagnostics. Identical quality exclusions/IDs and frozen128-dimensional epoch24 averaged CNN features from energy-area/semantic-code studies. The encoder saw all four source users: deleting a user only from the readout cannot estimate a fully unseen-user model or confidence interval. No encoder training occurred.

Each fold fits four heads after deleting each source user, and four matched random-row-deletion heads. Each random deletion removes exactly the same per-class counts as its corresponding user deletion, sampling without replacement with SeedSequence([20261007,held,deleted,class]). Every head refits mean, scale, intercept and ridge coefficients using retained source labels only, with equal-user/equal-present-class weights. Lambda is inherited unchanged from the original full-source ridge selection; no new validation or diagnostic selection. Missing user6/class3 and varying retained group counts mean this count matching is not causal isolation of user identity. All retained sample IDs and fitted parameters are saved.

For each class, primary takes the minimum of four raw head scores, equivalent to worst expected score over convex mixtures of these four fixed models. Controls average those scores and apply both rules to matched row deletions; original ridge unchanged. Raw scores are not probabilities, and this finite-model identity supplies no target generalization bound or coverage. Existing baseline cached onehot scores have a common−.5 offset; the original ridge replay checks that offset and identical argmax, without changing its predictions.

Official sklearn1.8 ensemble documentation, pinned blob028a4d380dfca3c584b75b6a4240df9b8311198a, supports without-replacement pasting and averaging as established techniques. It cites Breiman1996 bagging and1999 pasting; original papers were not read this turn. The primary is a finite-model minimum, not a bagging replication. Existing project OSR proposal also mentions support-sample jackknife calibration; the different closed-set/data role does not establish novelty. This is a failed research screen, not an original contribution.

Synthetic-only CPU timing0.084086s, conservative six-fold estimate1.513546s below300s gate; actual run7.152006s. The underestimate includes extra real scoring/checking/I/O absent from the synthetic timing and does not affect the completed budget. All48 fitted heads were checked against sklearn Ridge on both roles (96 comparisons). A separate replay independently rebuilt deletion masks, source roles, weighted moments,48 ridge systems and96 role predictions (maximum score difference1.825207e−13, exact argmax). It verified51 artifact SHA hashes, inherited selection, per-class count matching, no replacement, training-only IDs, aggregation, evaluation alignment and all gate arithmetic. Predictions/models/masks were saved before evaluation label use. The global label array is loaded for source indexing; labels are not technically sealed from memory, but held/validation labels do not fit or choose these new heads.

Reproduce from the private recovery chain and committed tools:

```sh
python uwb_direction1/tools/run_delete_user.py --protocol uwb_direction1/research/DELETE_USER_PROTOCOL.json --out /workspace/delete-user-replay-cost --cost-only
python uwb_direction1/tools/run_delete_user.py --protocol uwb_direction1/research/DELETE_USER_PROTOCOL.json --out /workspace/delete-user-replay --cost /workspace/delete-user-replay-cost/COST.json
python uwb_direction1/tools/audit_delete_user.py /workspace/delete-user-replay
```

Use fresh output directories. NumPy2.3.5, sklearn1.8.0, Python3.12, CPU single-thread numerical operations. Config references immutable previous feature/readout paths and SHA manifests; prior private energy-area and semantic-code receipts preserve the frozen encoders/features/baseline. Current logs, parameters, predictions, retained sample IDs, literature snapshot and code are archived privately. No large arrays are committed. Final `DRIVE_BACKUP_RECEIPT.json` is authoritative; archived report may precede receipt verification. Original789-file source snapshot unchanged. Full-eightfold strong ridge94.897727% belongs to another protocol and is not directly compared to six-source means. No background training or queue is running. Original research goal remains unmet; continuation has not been paused.

私有备份已逐份下载复验：7块、17970544字节、79成员SHA全部一致，元数据shared=false且仅owner；未更改共享权限。
