# Full default Disjoint-CNN: source and synthetic preparation only

This is a concrete published-method implementation candidate while the frozen
H-Inception CPU100 run is still active. No new real-data configuration, formal
cost policy/profile or UWB fitting is selected or launched here. Published
accuracy-baseline value remains separate from originality and is not rejected
merely because this architecture is already published.

## Primary support and actual access

Foumani, Tan and Salehi, *Disjoint-CNN for Multivariate Time Series
Classification*, ICDMW2021,pp760–769,
DOI [10.1109/ICDMW53433.2021.00099](https://doi.org/10.1109/ICDMW53433.2021.00099).
The [Monash institutional publication record](https://research.monash.edu/en/publications/disjoint-cnn-for-multivariate-time-series-classification/)
verifies the citation and abstract. Its stated approach separates temporal
feature extraction per variable from a subsequent operation mixing variables.
The author README and source were read in full for the files listed below.
The IEEE document page requires a JavaScript verification challenge; it was
not bypassed and no paper PDF/full methods/results were obtained. An unrelated
arXiv title was discarded, not used as evidence for this paper. Do not claim
full-paper review or the paper's reported benchmark results reproduced here.

## Unmodified author source and exact implemented scale

[Navidfoumani/Disjoint-CNN](https://github.com/Navidfoumani/Disjoint-CNN/tree/38180ccc1f050a3fee6961f5b4ea3bc132016ab3)
commit38180ccc1f050a3fee6961f5b4ea3bc132016ab3. Default model
classifiers/DCNN_2L.py blob7d46ed906334b2603e39d123b494e4797f11ec64.
Twelve original UTF-8 files were materialized and verified by Git-blob SHA-1,
size and SHA-256 in SOURCE_INDEX.json. Original MIT notice is retained in full;
the wrapper/checks using the prior GPL native-state helper are GPL-3.0-only.
Original sources are never edited.

Main.py selects **DCNN_2L**, epochs500, default batch8 (special datasets use1).
Its helper's10/16 default arguments are overridden by these main settings.
This preparation implements that whole default two-block network, not a
user-chosen reduction of a three/four-block model. Its shape argument already
supports multivariate input; (90,567,1) uses the existing author API, with no
input-channel literal patch. All four Conv2D layers retain64 filters, biases
and he_uniform initialization; kernels(8,1),(1,567),(5,1),(1,64), four BN/four
ELU(alpha1), original Permute(1,3,2), MaxPool(3,1) and GAP/Dense12softmax remain.
Actual total2587468/trainable2586956 parameters,18 trainable tensors and38
native Adam variables. Large input dimension567 makes the first spatial
kernel substantial; no CPU-efficiency claim is made before a future cost gate.

native_disjoint.py executes the original constructor/build_model/fit AST,
binds the original unqualified layer imports to native Keras, and changes only
three artifact suffixes: model_init.h5→model_init.weights.h5,
best_model.h5→best_model.keras, model.h5→model.keras. No TensorFlow stub or
replacement Torch architecture is used. A fit-setup prefix through the original
self.callbacks assignment is exposed with an observational return; compile,
mini-batch and callback expressions are unchanged except checkpoint suffix.
This is Keras3.11.3/Torch CPU compatibility, not TensorFlow2.11 execution parity.
Old requirements are retained for provenance, not installed globally.

Original fit uses min(ntrain/10,batch_size), native Adam, **validation loss**
ReduceLR(.5,patience50,minlr.0001) and strict minimum val_loss checkpoint.
Training class weights are max(1,log(2*N/count_c)), computed from training
one-hot labels only. They are not inverse-frequency sklearn weights or
renormalized to mean1. Validation has no class/sample weights; accuracy metric
is unweighted. These source choices differ from H-Inception's train-loss
selection and are preserved in the synthetic source-fit test.

## Genuine synthetic results and limits

SOURCE_PROBE.json verifies every layer scale/shape, independent NumPy
full-variable spatial convolution with bias, weighted categorical CE, original
class-weight mathematics, three native updates/full state restore/next two
model-Adam-BN-metrics-allRNG updates exactly, and native `.keras` weight/optimizer
reload. Fixtures B2/seed7 are engineering only. Native math references use
explicit small floating-point tolerances; saved/restored states remain exact.

FIT_SEMANTICS_CHECK.json invokes the original weighted fit body on25 synthetic
training/three validation rows for two epochs/26 updates. Requested source
batch8 becomes2 under the original min rule, with final batch1. A local test
wrapper disables native fit shuffling solely to expose the same known order;
source fit AST stays unchanged. Original class_weight fit and explicit
sample_weight native steps have exactly equal weights/Adam/BN/metrics, all
batch/epoch metrics/callback states and selected/final native checkpoints.
Epoch1 restoration continues through the next epoch with all explicit-loop
states/RNG exact. Native DataAdapter global-RNG parity is not asserted.
An independent251-step plateau rule follows val_loss while train loss moves
oppositely. CHECKPOINT_RULE_CHECK.json verifies strict val-loss minimum,
ties-first and that decreasing train loss cannot override validation selection.

First source-probe attempt failed because the test required exact equality of
the author's float32-derived class weights and an independent float64 formula.
The measured maximum difference was3.8146973579955556e-8 under NumPy2.3.5.
Only that independent mathematical-reference comparison was corrected to
rtol/atol1e-7. Original class weights/architecture/optimizer were not changed;
all model/state comparisons still demand exact equality. Original attempt
script/log and ENGINEERING_REPAIR.json preserve this engineering correction.
It was not an accuracy or UWB execution failure.

## Driver pitfalls deliberately kept distinct from the tested model

Main.py combines train/test labels to construct dummy columns and draws its
validation subset with replacement from rows still used in training. Those
choices are not copied into any future UWB protocol: use fixed12 task taxonomy,
separate source validation user(s), and no diagnostic/users7–8 labels for new
selection. Main's legacy np.int and prediction-wrapper branches require
compatibility work; label-dependent inference batching is not a valid future
label-blind predictor. Source process_ts_data has normalise=False in Main;
future UWB preprocessing must be specified explicitly as an adaptation.

Unused DCNN_3L/DCNN_4L and legacy Disjoint_CNN.py connect a third spatial
convolution to conv2, bypassing the immediately preceding temporal branch.
Original bytes are retained; this is not silently repaired or mixed into the
default2L model. The README's PyTorch block fragment uses GELU rather than
the tested TensorFlow model's ELU and is not a full classifier; it is not
treated as an identical ready-made implementation.

## Continuation boundary

Original789 and latest four remote records had no exact disjoint-cnn/
disjointcnn markers; this finite lexical scope is not exhaustive semantic
equivalence or originality proof. Supplementary original-Python activation/
initializer/Permute search is recorded for review, not novelty inference.
Runtime/source-state helpers reuse the existing pinned isolated dependencies;
no global package or running H-Inception science file changed.

Three readiness test cases verify refusal before diagnostic-summary access
when current full audit is missing/nonpassing/partial, and reject unset
configuration or mismatched registration. CURRENT_READINESS.json records the
actual expected refusal: current H-Inception full audit has not completed.
No next real configuration, formal CPU timing or UWB fit was attempted.
First-update insurance alone cannot satisfy the future complete-model guard.

Next finish existing H-Inception training/audit/private complete preservation
and its frozen gate. A pass requires separate source confirmation. Only a
verified exact closure permits choosing one new bounded cost policy/config/
seed and registering an actual source-only runner/auditor before UWB data.
Source defaults500/8 and these synthetic fixtures are not a selected real
budget. No test-label tuning, equivalent rescue sweep or novelty claim.
All8 historical development means exploratory evidence; users7–8 labels stay
excluded. The user's accuracy/contribution goal remains unresolved.

Reproduce source-only checks in separate processes:

```bash
python /workspace/uwb-disjoint-source-probe-20261031/synthetic_check.py
python /workspace/uwb-disjoint-source-probe-20261031/fit_semantics_check.py
python /workspace/uwb-disjoint-source-probe-20261031/checkpoint_rule_check.py
python /workspace/uwb-disjoint-source-probe-20261031/test_readiness.py
```
