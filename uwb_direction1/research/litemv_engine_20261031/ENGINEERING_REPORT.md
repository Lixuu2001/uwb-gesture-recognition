# Full author LITEMV reusable engine: synthetic checks pass

2026-10-31. Concrete implementation preparation while the existing OS-CNN
complete replay audit is active. No next real UWB configuration, timing policy,
formal cost measurement or data fit is selected or launched. This directory is
not a registered experiment or a complete executable UWB runner/auditor.

source_engine.py reuses the five unchanged author architecture methods through
the previous explicit native Keras binding. Source commit
`e62f64393104be8eda415b4d974987028ca68754`, model blob
`ef4aafa16864dfc7f95c7b312459e68573a86961`, GPL-3.0-only. The unchanged original
sources and full license remain in research/litemv_source_probe_20261031/vendor/LITE.
This engine and checks have the same GPL-3.0-only terms. It does not introduce
a smaller Torch substitute or a TensorFlow stub, nor reproduce original TF
execution. The original author fit's plotting/OHE/test machinery is not copied.

All caller settings are explicit: shape, fixed12-class taxonomy, batch, epoch,
seed and experiment identity. Current base Python/NumPy/Torch/Keras/backend
must match the pinned runtime; all source-preparation hashes and1302 existing
isolated runtime files are checked. No global package replacement or download.

Training uses native Keras train_on_batch/test_on_batch and original source
loss-monitoring callbacks. Supplied row permutation and actual IDs are logged
at each update, including the short final batch. Validation does not choose a
model; original ModelCheckpoint monitors training loss. Prediction accepts only
input/IDs/order, with no labels. Preflight refuses duplicate/non-permutation IDs,
train–validation overlap, invalid taxonomy, shapes and nonfinite values before
an optimizer update. This is generic input checking; a future actual-data
runner still needs full protocol/role/quality/hash verification and an auditor.

Checkpoint state contains all native model/Adam/BN/metrics and Python/NumPy/
Torch/Keras-global RNG, source/runtime/settings identity, original callback
state, module training modes, and explicit caller sampler/position. Optimizer
iteration must equal the claimed position. Identity or settings mismatch
rejects before restore mutation. Epoch-boundary recovery and the next two
epochs are tested; an automatic job-resume/mid-epoch runner is not implemented.

ENGINE_CHECK.json reports a full90×567/12-class synthetic comparison against
one uninterrupted native Keras fit: three epochs/six updates, B4/last2/seed19,
six training and three validation fixture rows. Every native batch/epoch
metric, model/Adam/BN/metrics, callback state and selected checkpoint matches
exactly. An observer recording exact synthetic IDs has identical results and
all RNG to no observer. Saved epoch1 restored in a fresh engine continues
through epochs2–3 with all states/module modes/RNG exactly equal. Prediction
ID ordering and missing validation-class handling pass. Native DataAdapter
global-RNG equality with the explicit loop is not asserted. Initial successful
engineering log is retained; final semantics were rerun after adding preflight
guards/base-runtime checks and removing an unused fixture model construction.

INPUT_GUARDS_CHECK.json separately verifies eight invalid role/order/taxonomy/
finite-input cases refuse before training, with model/Adam/callbacks/metrics/
module modes/RNG unchanged; a ninth check rejects a false Adam-step cursor.
READINESS_TESTS.log has eight passing test cases. Guards require previous full
29400-step/100-epoch source_final_last audit, both successful termination
records, standalone authenticated private complete-model readback and matching
archive/member/audit hashes before opening diagnostic SUMMARY. A prior gate
pass routes to source confirmation instead of replacing the method. Missing
fixed settings or a mismatched remote preregistration marker refuses entry.

CURRENT_READINESS.json records actual refusal: the existing OS-CNN audit is
not finished. It did not open producer scores, choose a new configuration,
read UWB values/labels, run formal cost, or train. Guard tests use temporary
synthetic proof files and are not claims of actual private backup completion.

The author's exact znormalisation function is separately extracted without
edits and checked against independent NumPy, including zero variance. It is
not automatically applied to these engine training fixtures. Future raw/RMS/
z-normalization input choices must be specified prospectively, without
diagnostic-driven rescue. The paper's1500 epochs and current constructor2
remain distinct; neither is a selected real-data budget in this directory.
Synthetic fixture runtime is not a formal CPU resource gate.

Reproduce in separate processes:

```bash
python /workspace/uwb-litemv-engine-20261031/test_engine.py
python /workspace/uwb-litemv-engine-20261031/test_input_guards.py
python /workspace/uwb-litemv-engine-20261031/test_readiness.py
```

Next finish existing OS-CNN audit/private complete model preservation/registered
source gate, without restarting it. Only after an exact gate failure and
verified closure may a prospective full-LITEMV CPU-cost policy, preprocessing,
one configuration/seed and actual runner/full auditor be finalized, published
and byte-exactly read back before UWB fitting. A resource failure does not
measure accuracy. Users7–8 labels remain excluded. Existing all8 development
makes future source results exploratory. Published accuracy evidence remains
separate from original contribution; no accuracy/novelty claim is made here.
