# Fixed real MLSTM-FCN8 paper250/B128 source pilot

Literature: Karim et al., Neural Networks116(2019)237–245,
DOI10.1016/j.neunet.2019.04.014/arxiv1801.04503v2. Previously reviewed
architecture, training and ablation sections support this published baseline.
Use complete documented LSTM8/Conv128-256-128/SE16/Masking/Dropout0.8 native
architecture,860364parameters/859340trainable. Source commit85100ccfec7ee...
model blob6d425bda53e84b0c9d6c620a016f29b903807ae6 remains unchanged. Keras3
sigmoid differs from legacy Keras2.1.2 hard_sigmoid; no old TF parity claim.
250epochs/B128 are paper/README settings; dataset script1000/500 are distinct.

Exactly250epochs/B128/seed20261029, oneCPUthread, no ensemble, search,
normalization addition, resize or augmentation. Use frozen full90×567 signed
per-view RMS cache, transpose to567×90. Preserve LSTM567steps/CNN90steps.
Source ID manifest6900 quality samples/users1–6 remains fixed. Users3–6
train4700; user2validation1000; user1diagnostic1200. Verify source subject IDs
before indexing any gesture labels. Training and audit loaders only index
train/validation labels; neither can index diagnostic/user7–8 gesture labels.

Use inverse present-training-class frequency weights and fixed12head,
unweighted validation. NativeAdam.001/.9/.999/epsilon1e-7, no decay/clip.
Callbacks checkpoint thenReduceLR, both monitor weighted training loss,
strictminimum/tiesfirst; patience100/factor1/cuberoot2/minlr1e-4. Validation
metrics logged only. Dedicated TorchGenerator(seed+zero-based epoch), no
global RNG draw. Evaluation order is label-blind RandomState(seed+101val,
seed+102diag), fixedB128; no DataAdapter RNG equivalence claim.

Retain all9250updates/1175000training appearances and250epochs with sample
IDs, cumulative metrics/batchloss, callbacks/LR; initial/first/best/rollinglast/
finalprediction fullstates; nativebest/last, split/evaluation IDs and both
unscored predictions. Final prediction state uses selected weights+last Adam
for inference only. Best custom fullstate retains its own optimizer/RNG.
Native.keras alone does not restore layer RNG exactly; no automatic resume.

Actual preregistered full-scale synthetic cost passed21684.352822553366seconds
(6.02343h), RSS822652KiB; fixed policy already in3f0007c2... and verified
result inb1f5e054.... This pilot freezes exact evidence hashes. Training child
43000/43100CPU,43100wall; audit28790/28800CPU,28800wall. Sequential controller
starts full audit only after successful training; no retry/resume. No Drive,
Git, scoring or parameter-selection capability in the controller.

Full auditor verifies frozen dependencies, rawX→RMS and independentFP64
reference for all6900unlabelled source inputs, exact roles/IDs/partitions,
every batch order/metric/loss/classweight, strict selected epoch, initial/
first/best/last/final native/customstates, both predictions and native files.
Native file observation isolates RNG changes and proves live weights/Adam/
metrics/layerRNG/modes unchanged. Audit never reads diagnostic labels.

After successful full audit, create complete private backup, metadata-first
owner-only rawfetch/include_base64=false, immediate size/SHA, max3 pipelines,
full archive/every-member readback. Initial/first-only insurance explicitly
does not authorize scoring. Only finalize.py after complete private proof
may index user1 diagnostic labels once and compute the fixed primary gate.

Primary gate is the SAME bounded pilot rule used for previous candidates:
user1 mean present-class recall gain>=0.005 (tolerance1e-12) over fixed
TimeMIL64 reference0.9866666666666667 with exact same diagnostic ID set and
frozen prediction SHA; no new comparator chosen from new scores. H's already
known user1 score0.9941666666666666 is disclosed as context, not silently
overlooked. H source6 CPU100 failed its registered gate and is closed. A
pilot pass only permits separately registered source1–6 confirmation against
the frozen strong pipelines; it does not establish beating every known
single-user score, eightfold improvement or originality. A failure closes
exactfull250/B128/seed configuration without budget/seed/error-based rescue.

All source development is exploratory because of historical repeated access;
users7–8 labels stay out of selection. Published-baseline accuracy evidence
and original-contribution evidence remain separate. Goal unresolved.

Before any real MLSTM input/label decode: freeze protocol, all code/tests/
original source/native/runtime/cost/prior closure references and actual
byte hashes; commit from latest branch parent and exact-readback every file;
require matching remote marker at controller, trainer and auditor entry.
Original789/runtime unchanged; raw/maps/actualIDs/model binaries never Git.
