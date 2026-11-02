# Full-default Disjoint-CNN cost utility; no policy/profile selected

Adds concrete prospective cost arithmetic and a guarded synthetic profiler
for the complete default DCNN_2L, using the existing pinned native source
and weighted epoch/state engine. This is tooling, not a cost result or
registered UWB experiment. No policy file, actual batch/seed/epoch budget,
timed model execution or real-data preprocessing is chosen here.

The actual-data decision boundary remains the current H-Inception study's
complete numerical audit and private complete-model preservation. The utility
calls that guard before reading a supplied future policy, creating outputs
or importing the native epoch engine. A failing or partial previous study
cannot trigger a new profile. A passing current accuracy gate instead requires
separate source confirmation, as already registered.

The formula uses the source effective batch `int(min(4700/10, requestedB))`,
full4700 training/1000 validation/1200 diagnostic counts, and the maximum of
each timing vector. Its conservative estimate is
`2*(E*trainBatches*trainMax+(E*valBatches+valBatches+diagBatches)*evalMax
+(E+1)*nativeSaveMax+(2E+4)*fullStateSaveMax)+120`.
It includes worst-case best-and-last full-state snapshots every epoch. A
fixed ordered list of prospective budgets is tried in its declared order;
all failing time/memory gates close that exact policy without an accuracy
claim. These arithmetic test fixtures are not prospective real budgets.

When a later fixed policy is actually authorized by the prior-closure guard,
the CLI uses generated Gaussian full90×567×1 inputs and synthetic fixed12
one-hot labels. It obtains the effective mini-batch/Adam/callback setup from
the unmodified source fit prefix, checks2,587,468 total/2,586,956 trainable
parameters, profiles eight weighted training batches/four unweighted eval
batches/three native saves/three complete-state saves, then applies the
declared time and RSS gate. It records preprocessing intent without opening
or preprocessing any UWB arrays. Generated synthetic values are timing
fixtures; reported timing can never establish UWB accuracy or TF parity.
CPU590/600-second soft/hard limits and a610-second alarm apply only to that
future guarded invocation. Any actual cost launch still needs its frozen
policy/dependencies and independently bounded supervisor/termination receipt.

Four arithmetic/refusal tests passed: independent formula/effective-batch
calculation; ordered-budget and memory/time failure; nonfinite/invalid policy
and measurement refusal; and missing prior-audit refusal before policy reads,
output creation or native epoch-engine import. Actual current guard refusal
is recorded separately. No timed profile or model fit was performed for
these tests. Existing source/engine/runtime dependencies remain unchanged.

No UWB runner/auditor/configuration is selected by this utility. After the
current study closes, preregister an actual method/config/seed/preprocessing
and all-step persistent runner/auditor before opening UWB values or labels.
All8 historical development remains exploratory; users7–8 labels stay out of
new selection. Published-method accuracy value is separate from originality;
the user goal remains unresolved. New code uses GPL-3.0-only alongside the
previous MIT author source; the included GPL notice is retained unchanged.
