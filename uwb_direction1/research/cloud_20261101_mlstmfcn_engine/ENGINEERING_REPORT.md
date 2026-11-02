# Complete MLSTM-FCN native epoch engine; synthetic verification only

The fixed full8-cell/Conv128-256-128/SE16 model's actual synthetic cost policy
250epochs/B128/seed20261029 passed: conservative21684.352822553366s (6.02343h)
under43200s, RSS822652KiB under14GiB. Original timing vectors, declaration,
successful bounded termination and exact remote preregistration are preserved
in the separate formal cost workspace. This is a resource result, not accuracy.

This engine preserves the full native model, inverse-frequency class weights,
unweighted validation, strict minimum weighted native training-loss selection,
source helper checkpoint/ReduceLR order, patience100, factor1/cuberoot2,
min_lr1e-4. A full native.keras save replaces the helper weights-only filename
for evidence retention; custom state also saves layer/global RNG and module
modes, which native files alone cannot resume exactly. No original author
files or global packages were changed. No original sources are redistributed.

The caller must supply explicit seed/count/epoch/batch settings. A dedicated
Torch Generator(seed+zero-based epoch) sampler covers all rows without drawing
global Torch RNG. This is an explicit planned protocol adaptation; no old
TensorFlow or native.fit DataAdapter shuffle/RNG equivalence is claimed.
Native Keras3 sigmoid rather than old Keras2.1.2 hard_sigmoid remains disclosed.
The fixed12 head and present-training-class weighting avoid fitting class
dimensions on validation/diagnostic labels. Callback monitoring uses training
loss; validation is recorded, and no diagnostic selection exists in the engine.

Every batch returns its exact row indices, optimizer cursor, native cumulative
metrics and loss recovered from tracker deltas. An optional callback supports
persistent sample-ID logs/first-update checkpoints in the future runner.
Snapshots include complete native model/Adam/BN/metrics/global+layer RNG/modes,
callbacks, explicit settings, identity and epoch/batch/cursor. Bad identity,
settings or cursor fail before model mutation. Predict is label-blind and
checks the explicit partition and normalized12-class probabilities.

Actual synthetic checks passed in ENGINE_CHECK_FINAL.log: full-size architecture
with5training/3validation rows,3epochs/9updates, B2 (last batch1); independent
class-frequency calculation; exact initial and first-update state restoration;
epoch1 disk save/restore followed by next2 epochs reproducing every recorded
metric, order, all state objects and predictions; native selected model weights
and Adam match strict minimum training loss;3identity/settings/cursor rejection
checks,7numeric shape/dtype/label rejection checks, sampler repeatability and
global-RNG preservation. Synthetic B2/seed7 and small fixture epochs are
engineering checks, not replacements for the fixed real candidate250/B128.
Native deserialization is observational and its RNG effects are isolated.
The first synthetic attempt also passed; the final adds initial/first-update
coverage and removes unused test code. Both real logs remain, no failure hidden.

There is **no actual UWB value/label load, runner, role protocol, full-data
numerical auditor, CPU run or accuracy result** here. A resource pass alone
does not authorize real fits: next implement and preregister exact user1held/
user2validation/other source users training IDs, fixed accuracy gate, epoch
runner, independent raw-input reference/full replay auditor, checkpoint and
private-preservation flow. Inspect old processes/checkpoints first. Only then
may a matching remote exact-readback marker permit data access. Users7–8 labels
remain excluded; historically reused source users make later scores exploratory.
Do not tune based on previous H/Disjoint diagnostic values or claim originality.

Source/paper references remain Karim et al., Neural Networks116(2019)237–245,
DOI10.1016/j.neunet.2019.04.014, arxiv1801.04503v2 and the pinned model blob
6d425bda53e84b0c9d6c620a016f29b903807ae6. Full source/native compatibility
preparation and independent masked LSTM/SE math checks remain unchanged in
the prior source workspace. This is a published-baseline implementation step;
the user's accuracy/contribution objective remains unresolved.
