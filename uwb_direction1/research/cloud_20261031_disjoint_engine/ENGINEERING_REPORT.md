# Default Disjoint-CNN explicit epoch and state engine

Concrete executable preparation using the previously pinned full author
DCNN_2L/native Keras adapter. This adds source-weighted train steps, unweighted
validation, original validation-loss callbacks and full epoch-boundary replay.
It does not choose an actual UWB configuration, preprocessing, cost policy or
seed, and has no UWB executable/launcher/auditor. Existing H-Inception training
continues independently; its frozen dependencies and science are unchanged.

`source_engine.py` has no default epoch/batch/seed. It obtains the optimizer,
mini-batch rule and callbacks from the original author fit AST prefix. It uses
the source class-weight function on training labels only. All training rows
appear once in a supplied permutation, including the last incomplete batch;
validation has no class/sample weights. Validation batching matches the
original effective mini-batch. Label encoding has fixed12 columns without
examining any diagnostic/test labels. Native strict-minimum val_loss model
checkpoint and ReduceLR remain authoritative.

The explicit sampler uses a dedicated Torch Generator with seed+epoch; its
choice must be declared in any future real protocol. Native DataAdapter shuffle
RNG parity and original TensorFlow execution parity are not asserted. An
explicit fixed-order fixture is used solely to compare original native fit
with the new engine. Snapshot/restore covers model/Adam/BN/metrics, callback
state, all native/Python/NumPy/Torch/Keras RNG, module training flags, settings,
caller identity and epoch/update cursor. This is epoch-boundary engineering;
whole-job/mid-batch automatic recovery is not implemented or claimed.

`input_guards.py` checks role-disjoint source users1–6, integer unique IDs,
no role overlap and no label-bearing metadata. The actual-data entry calls
the existing full-audit/private-preservation/preregistration guard before
any provided metadata/value/label loader. Its label loader receives only
training/validation roles. A later real protocol still needs complete frozen
raw-map preprocessing, ID provenance, persistent artifacts and independent
all-step replay. This API is not itself that registered protocol.

The synthetic integration check compares the full original weighted native fit
with this engine on25 training/three validation fixtures for two epochs:
effective batch2, last training batch1,26 updates. It checks weights/Adam/BN/
metrics/callbacks and selected native checkpoints, then saves epoch1 and
replays epoch2, predictions and post-prediction states. A wrong checkpoint
identity must refuse before state mutation. Sampler tests check full coverage,
repeatability, changed epoch order and untouched global Torch RNG. Seven
numeric guards reject wrong input dtype/shape/nonfinite values and invalid
fixed12 labels. Six separate leakage/refusal tests use only temporary
synthetic metadata; the mocked passing guard is explicitly not the actual
H-Inception audit or backup. Actual current readiness remains refused until
the running study fully closes.

Paper-access refresh used title/DOI searches and found the same author source,
institutional abstract and closed-access DBLP record. No new complete paper
was obtained; the previously documented IEEE challenge was not bypassed.
Do not claim full-paper review, benchmark reproduction, UWB accuracy gain or
original contribution from this engineering work. Published baselines remain
valuable for actual accuracy validation after the existing registered run.

Run the synthetic checks in separate processes:

```bash
python /workspace/uwb-disjoint-engine-20261031/test_input_guards.py
python /workspace/uwb-disjoint-engine-20261031/synthetic_engine_check.py
```

Real source users/IDs/maps/labels were not used for these fixtures. The user's
accuracy/contribution goal remains unresolved. Users7–8 stay excluded from new
selection; historical all8 development means exploratory evidence. First
finish current H-Inception complete audit/private preservation and frozen
gate. Pass requires source confirmation; an exact verified closure permits
the next single cost policy/config/seed and remote preregistration before UWB
values or labels. Do not launch from this engine preparation alone.
