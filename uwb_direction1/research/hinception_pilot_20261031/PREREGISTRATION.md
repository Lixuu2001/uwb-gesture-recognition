# Prospective H-Inception multivariate transfer: CPU100 source accuracy pilot

Freeze this protocol/implementation and byte-exactly read back its new remote
commit before reading new UWB values/labels or fitting. One configuration:
H-Inception sourcecf4d0e55ee11b7393ae3d317cdf5f22dcc51ba3e, full width32,
bottleneck32, depth6, kernels40/20/10, two residuals,17 fixed regular Conv1D
filters, native Keras3.11.3/Torch CPU; one seed20261029/B64/100 fixed epochs.
Total851434/trainable601006/50 trainable tensors/102 Adam variables.

This explicitly adapts one source input literal1→567,26 native Keras bindings
and one checkpoint suffix. The original method is univariate UCR/TensorFlow2.4.1;
this is a full-width/depth multivariate transfer, not original paper execution
or five-model H-InceptionTime. The source fixed convolutions sum variables
into17 features. They are not LITEMV's per-channel depthwise concatenation.

Prior OS-CNN CPU100 full29400-step audit and standalone private complete-model
readback succeeded; its diagnostic95.1667% versus fixed98.6667% failed the
registered gate. Exact LITEMV native Torch B64 resource policy also closed:
prospective1500/100 budgets both exceeded43200s, no UWB fit or accuracy measured.
This different source architecture's single prospective full-scale cost
profile passed CPU100:20890.104619972175s conservative estimate; original1500
310817.49064172595s is not run. No label-driven batch/width/depth/seed rescue.

Use source users3–6 training4700, user2 validation1000 and user1 diagnostic1200;
users7–8 labels never enter new selection/fitting. Full90×567 signed real RF
RMS maps from the immutable source cache, followed by the unchanged author
z-normalization function across90 time points per sample/channel, STD0→1.
Known task12-class taxonomy supplies train/validation one-hot arrays; do not
infer output size from diagnostic/test labels or refit an OHE on validation.
No resize, augmentation, clipping, balancing or architecture search.

Native Adam(.001,.9/.999,epsilon1e-7,no decay/clip), original ReduceLR(loss,
factor.5,patience50,min_delta1e-4,minlr.0001) and strict minimum training-loss
ModelCheckpoint (ties retain first). No early stopping or validation-based
checkpoint selection. Explicit dedicatedTorchGenerator(seed+epoch) permutations
of all4700 training rows:74 batches/last28,7400 updates/470000 appearances.
Label-blind validation/diagnostic order uses counts/seed+101/+102 and B64;
native DataAdapter or original TensorFlow RNG parity is not asserted.

Save initial/first/best/rollinglast/full finalprediction states, native
best_model.keras/last_model.keras, every actual training ID/batch/metric and
epoch loss/LR/callback states. Snapshot includes model/Adam/BN/metrics/callbacks/
module modes/allRNG/identity/sampler. Best checkpoint includes its own optimizer;
finalprediction state intentionally uses selected best weights with last
optimizer for inference audit, not a best-trajectory continuation claim.
Automatic retries/resume disabled; epoch-state restoration is tested.

Both validation and diagnostic IDs/probabilities/decisions and finalprediction
state are saved before diagnostic-label indexing. Then compute present-class
macro accuracy. Continue only if diagnostic gain over fixed .9866666666666667
is≥.005 (predeclared numerical tolerance1e-12). Pass permits a separately
registered source-user confirmation; fail closes this exact configuration.
No diagnostic-selected epochs/seed/input/width/depth/filters or new thresholds.

Independent auditor reconstructs all100 epochs/7400 updates/470000 actual ID
appearances, strict train-loss checkpoint/LR, initial/first/best/last/final
states including optimizer/callbacks/RNG/modes, native selected/last checkpoints,
all6900 raw RMS inputs and source z-normalization, label-blind batch partitions
and all2200 final probabilities/decisions. Training≤43200s; audit independently
bounded14400s. Expected local controller waits for this one actual run, audits
once and builds complete local archive; it cannot restart training, upload
Drive or write Git. Agent must verify owner-private complete readback separately.

Preserve first-update insurance and final complete model set with exact IDs,
logs,source/configuration/audit; metadata first, max3 authenticated rawfetch
pipelines, include_base64=false, immediate materialization/size/SHA and all-member
reassembly. Arrays/maps/model/runtime binaries stay out of Git. Whole-job/
environment resume is not claimed. A failed execution is an engineering failure,
preserve traces and repair before any scientific conclusion.

All8 users have historical development exposure; all new source evidence is
exploratory, not an independent test or legacy eightfold comparison. Published
baseline accuracy value is distinct from originality. Research goal remains
unresolved and existing hourly continuation stays enabled until achieved/stopped.
