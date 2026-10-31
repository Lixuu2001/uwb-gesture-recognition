# LITEMV: pinned full architecture, synthetic implementation preparation

2026-10-31. This is concrete preparation for a published accuracy baseline,
not a real UWB accuracy result or an original contribution. Existing OS-CNN
CPU100 training completed normally; its full replay audit is still active.
No new real-data configuration, formal timing gate, or LITEMV training is queued.

## Primary paper and scope actually read

Ismail-Fawaz et al., *Look into the LITE in deep learning for time series
classification*, DOI [10.1007/s41060-024-00708-5](https://link.springer.com/article/10.1007/s41060-024-00708-5),
IJD Sc Analytics 20, 4029–4049 (2025); [author preprint](https://arxiv.org/abs/2409.02869).
Read publisher abstract, sections 3.2, 3.3.1–3.3.3 and 4.1–4.2, including
equations and captions, plus the preprint's opening abstract/introduction.
The accessible preprint is 26 pages; the full paper and figure visuals were
not completely reviewed.

The paper combines separable convolution, parallel kernel sizes, dilation and
fixed filters. Its multivariate variant applies fixed filters independently
to channels and concatenates their outputs. It specifies 32 filters, kernels
40/20/10, dilation 1/2/4, batch 64 and 1500 epochs, with Adam and the best
training-loss model. A five-model ensemble is a separate method. These are
published precedents, not evidence of accuracy on this UWB dataset.

## Author code, preserved without source edits

[MSD-IRIMAS/LITE](https://github.com/MSD-IRIMAS/LITE/tree/e62f64393104be8eda415b4d974987028ca68754)
commit `e62f64393104be8eda415b4d974987028ca68754`.
LITEMV model blob `ef4aafa16864dfc7f95c7b312459e68573a86961`.
All ten retrieved UTF-8 files retain their original bytes and Git blob SHA-1,
size and SHA-256 in SOURCE_INDEX.json. GPL-3.0-only license is retained in full.
The adapter and checks accompanying the extracted source are distributed under
the same GPL-3.0-only terms. Adapter change: execute only five original class
architecture methods and explicitly bind their 21 `tf.keras` references to
native Keras. No TensorFlow stub or replacement Torch architecture is used.

Code inspection establishes:

- All three first-layer branches use SeparableConv1D; subsequent separable
  layers use dilation 2 and 4. Input is the full synthetic 90×567, output 12.
- Seventeen fixed per-channel DepthwiseConv1D filters use alternating signs
  at sizes 2/4/8/16/32/64 and five squared-ramp peak kernels at 6/12/24/48/96.
  Their concatenation contains 9639 features; with the learned branches the
  first BatchNormalization has 9735 features. This can dominate memory and
  CPU cost; the univariate paper parameter count cannot be quoted for it.
- Actual model: 621680 trainable parameters, 889624 total including fixed
  kernels and BN statistics; 18 trainable variable tensors and 38 native
  Adam variables after initialization/update.
- The source constructor currently defaults to **two** epochs, unlike the
  paper's 1500. Neither is a chosen real-data budget here. A two-epoch default
  run cannot close the full paper setting as an accuracy failure.
- Original callbacks monitor training `loss`: ReduceLROnPlateau factor .5,
  patience50, default min_delta1e-4 and min_lr1e-4; ModelCheckpoint saves
  strict minimum training loss to `best_model.keras`. No early stopping.
- Original fit fits independent one-hot encoders to train and validation.
  This would mishandle our known missing validation classes. Any future
  adapter must use the existing fixed 12-class task taxonomy for both roles,
  without learning a mapping or class count from diagnostic/test labels.
- Original main's default five-model ensemble and test-derived output size
  are not copied. This probe is one full LITEMV, not LITEMVTime. The stored
  `use_multiplexing` flag is not forwarded by build_model; only unchanged
  author defaults are exercised, not an alleged false-flag ablation.
- The author loader applies per-sample channel-wise z-normalization. The
  current probe has no data adapter. A future RMS-only input would require
  explicit disclosure as an input adaptation, not silent author equivalence.

## Verified implementation scope

SOURCE_PROBE.json and SYNTHETIC_ATTEMPT01.log record analytical equality of
all 17 filters, nontrainability and unchanged values after three native
updates; native model/Adam/BN/metrics/all RNG restore and next two updates
match exactly. Native `.keras` weights and optimizer reload match exactly.
Independent NumPy checks cover dilated separable convolution and even-SAME
fixed depthwise convolution. This does not assert original TensorFlow parity.

The separate fit-semantics check compares native Keras fit against native
explicit batches, including a short final batch, callbacks and checkpoint
selection. See its actual JSON and logs for pass/failure; no claim extends
beyond those checks. Full native fit DataAdapter RNG parity is not asserted.

First fit-semantics attempt failed on log-key comparison after model,
optimizer and metric equality had passed. CallbackList.pythonify_logs makes
a new dictionary and ReduceLROnPlateau adds `learning_rate` inside the
callback chain. The explicit test had retained its pre-callback dictionary.
The repair records Recorder's post-callback dictionary. No keys are removed
from the equality assertion; the original script and failure log are retained.
This is an engineering fixture correction, not a scientific negative result.

## Runtime and reproduction

Reuse the existing isolated Keras3.11.3/Torch CPU dependencies; no global
package installation or replacement occurred. The author's README warns of
fixed-filter training bugs in old Keras and requires >=3.6. We verify actual
fixed kernels, not just a version string. DEPENDENCY_VERIFIED.json verifies
1302 existing runtime files and ten wheels against the retained manifests.
Base Python/NumPy/Torch versions are in dependencies/RUNTIME.json. This is
dependency recovery, not a whole-job resume snapshot.

The helper currently reuses
`/workspace/uwb-resnet-keras-screen-20261030/deps`. In another cloud workspace,
restore those hash-pinned dependencies there using the earlier registered
ResNet screen's restore_dependencies.py and manifests; or adapt this explicit
path before a future protocol freezes the helper. Do not install the author's
TensorFlow/pandas/codecarbon plotting stack globally just to construct a model.

Run in separate processes, one CPU each:

```bash
python /workspace/uwb-litemv-source-probe-20261031/synthetic_check.py
python /workspace/uwb-litemv-source-probe-20261031/fit_semantics_check.py
```

Fixtures B2/B4, seeds7/9/31 and three engineering epochs are synthetic only;
they do not choose a real UWB epoch/batch/seed configuration. Checkpoint
fixtures use temporary directories and are not UWB-trained models. No actual
sample IDs, raw arrays, maps, dependency binaries or model binaries enter Git.

## Continuation boundary

History search found no exact LITEMV markers in the inspected original789 and
four latest remote records; details and limits are in HISTORY_CHECK.json.
This is not a novelty or full semantic-equivalence proof. LightTS is distinct.

First finish the existing OS-CNN full audit and private complete-backup/gate.
Only then may a new bounded cost policy/configuration be selected. If this
candidate proceeds, retain full author architecture, specify preprocessing,
batch/epoch adaptation and one seed prospectively, and preregister/read back
before actual UWB values, labels or fitting. Use source train/validation
roles and the existing source-only gate; users7–8 remain excluded. Preserve
IDs/logs/models and full replay evidence. A failed exact configuration closes
that configuration; a resource failure is not an accuracy failure. Published
accuracy evidence and originality remain separate. All8 historical development
makes any later source result exploratory, not an independent final test.
