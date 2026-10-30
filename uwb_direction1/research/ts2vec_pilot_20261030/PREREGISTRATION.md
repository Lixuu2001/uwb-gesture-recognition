Fixed published-method accuracy pilot, not a new architecture or contribution.
This protocol is frozen before any TS2Vec real UWB values are read or fit. The
previous TSLANet exact configuration closed by its original registered gates,
passed its complete audit and has a verified private composite model recovery.

Train users3–6(4700), validation2(1000), diagnostic1(1200), existing6900-source
manifest/quality exclusions. Keep all189rangebins×3views and90frames with the
existing exact per-record/view RMS maps; no resize or fitted cross-record input
normalization. Exclude users7–8 labels from fitting/selection/scoring.

Full TS2Vec constructor defaults:673472parameters, output320, hidden64,
depth10 yielding11actual residual blocks, batch16, temporal_unit0, no lengthcap.
The original train_data.size rule selects600iterations here, not600epochs.
CLI batch8 differs; its maxlen3000 has no effect on90frame inputs. CPU,
seed20261029 and source-user/RMS protocol are explicit adaptations. No smaller
architecture, external augmentation or architecture/loss change is used.

The original author minibatch AST is extracted unchanged: overlapping random
contexts, independent timestamp binomial masks/dropout, hierarchical instance+
temporal loss, AdamW.001/defaultdecay.01, no clip/scheduler, and initial-inclusive
averaging. Three-step original-fit comparison matches raw/average models, losses
and all RNG states exactly. Full checkpoint restoration matches the next two
synthetic updates bitwise for raw+average+optimizer/all RNG states. For the real
driver only the data order changes to a fixed dedicated epoch seed+epoch sampler;
its boundary cases are tested. All600actual16-ID batches are logged privately.
Exactly600updates are used; there is no validation-based encoder selection.

Extract320D full-series max-pooled features from the601-state averaged encoder.
Use the original CLI RBF-SVM helper:all tenC values including infinity, fivefold
CV strictly within four training users, original raw feature geometry and refit.
Only n_jobs5→1 is changed. Training CV IDs and all CV results are preserved; any
nonfinite CV score, failed fit or convergence warning makes execution incomplete,
not an accuracy negative. Validation2 labels do not fit the SVM or choose C.
The unused old linear helper is not part of this protocol.

Freeze both validation and diagnostic IDs/predictions/decisions before reading
their labels. Report present-class macro from classifier.predict (SVM decision
argmax is not substituted). The only gate is diagnostic macro at least.5pp above
the frozen strongest same-held1 reference, with1e-12 comparison tolerance. The
reference is max(TimeMIL64/GAP64/fullCNN24/localCNN24 and completed TSLANet arms):
98.6666666667%. Pass licenses separately frozen known-baseline source confirmation;
fail closes this exact600/batch16/seed/configuration without rescue grids.
No independence, eightfold gain or originality is inferred from this pilot.

Preserve initial, first update and each50step/last raw+average+optimizer states,
all Torch/NumPy/Python RNG, iteration/sampler cursor/identity, actual batch IDs,
SVM/fold IDs/results, features/predictions and logs privately. No automatic retry
or automatic resume. A tested restore function and deterministic sampler make
checkpoints usable for a separately documented engineering continuation if needed.

Prospective synthetic full-scale cost963.335652seconds passed43200s CPU/wall.
Real child has43000/43100CPU soft/hard and43100wall, independently supervised.
This estimate is not a runtime guarantee or an accuracy result. Before reporting
the gate, require successful training termination, completed prediction freeze,
the independent completed-run audit and authenticated private preservation.
The audit guards all inputs, reconstructs all600training updates/states/RNG and
9600sample appearances, checks6900source RMS inputs and all learned features,
refits the original training-only SVM and replays2200predictions/metrics/gate.
Audit hard bounds:3600s wall/CPU. Audit/engineering failures retain receipts and
are repaired without altering the scientific configuration or consulting scores.

The source users and all eight users have been repeatedly developed historically;
the result is exploratory. Accuracy evidence and originality evidence remain
separate. Original789source snapshot stays untouched; raw arrays, checkpoint
tensors, learned arrays and actual sample-ID values are excluded from Git.
Source paper/author-code details and prior synthetic faults remain in
research/ts2vec_author_screen_20261030 and its results directory.
