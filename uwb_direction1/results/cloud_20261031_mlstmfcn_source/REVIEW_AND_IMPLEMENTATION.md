# MLSTM-FCN source and native compatibility preparation only

This adds another concrete published-method candidate while the existing
H-Inception full numerical audit runs. It does not replace the prepared
Disjoint-CNN candidate, choose a next real method/configuration, or launch
another UWB profile or fit. Accuracy-baseline value and originality remain
separate; no accuracy improvement or new contribution is claimed here.

Karim, Majumdar, Darabi and Harford, *Multivariate LSTM-FCNs for Time Series
Classification*, Neural Networks116(2019)237–245,
[DOI10.1016/j.neunet.2019.04.014](https://doi.org/10.1016/j.neunet.2019.04.014),
[arXiv1801.04503v2](https://arxiv.org/pdf/1801.04503).
Selected fulltext architecture/input/training/results/ablation sections were
read, with PubMed citation crosscheck. Figures were not visually inspected;
benchmark results were not reproduced. The paper's reported250epochs/B128
and LSTM8/64/128 search are source information, not new UWB settings.

## Source identity and actual implemented variant

Reviewed coauthor repository titu1994/MLSTM-FCN commit
85100ccfec7ee08106b3a25906a0c9c707e1b7ab. The article links
houshd/MLSTM-FCN, whose checked head is84a2e6c0fdef811920483723ac1aaf29d6b8dc7f.
Both repositories' gesture_phase_model.py are byte-identical blob
6d425bda53e84b0c9d6c620a016f29b903807ae6. Nine fetched files are verified by
Git-blob SHA/size/SHA256 in SOURCE_INDEX.json. The reviewed repository's
license metadata is null; original files stay local in local_author_sources/
and are excluded from Git exports. Only provenance, our adapter/checks and
actual synthetic logs/results are published. Missing originals can be
retrieved by the GitHub connector at the pinned paths/ref and verified again;
no manual data upload is needed. The adapter/state-helper code is GPL-3.0-only;
this does not assign a license to the original repository's source.

The implementation executes only original generate_model and
squeeze_excite_block AST from gesture_phase_model.py, binds native Keras
imports and replaces one `_keras_shape` attribute with `.shape`. Original
source bytes stay unchanged. Full Conv128/256/128 kernels8/5/3, all biases,
three BN/ReLU, both SE16 blocks, Masking, LSTM8 and Dropout0.8 remain. Input
metadata is567variables×90timepoints with fixed12 task classes. Original
Permute returns90timepoints×567channels to the CNN; the LSTM scans567steps.
No source stride, branch removal or recurrent-width reduction is introduced.
This is the complete documented8-cell variant, not a reduced replacement of
the separately documented128-cell Uwave variant or an attention model.

Actual native total860364/trainable859340 parameters,21 trainable tensors,
44 Adam variables and two layer RNG variables. Independent integer accounting
matches. The queue has no selected real8/64/128 choice: fixture variant8 and
seed7/B2 are engineering only. Dimension shuffle has no assumed speed benefit
when variables567 exceed timepoints90; an actual bounded cost gate is needed.

## Compatibility and source protocol differences

The old requirements give lower bounds TensorFlow-GPU1.4/Keras2.1.2; they do
not identify the paper's exact environment. Keras2.1.2 source at commit
26d5e100148272166868da8f817e1a4fb49e34cc has default recurrent hard_sigmoid.
The current native Keras3.11.3/Torch adapter's unspecified LSTM gate is sigmoid.
This difference is explicit: no historical TensorFlow numerical parity or
complete original training reproduction is asserted. A future protocol must
declare its framework/gate choice before UWB data or cost selection.

The README's250epochs differs from the Gesture Phase script's1000 and Uwave
script's500; the helper defaults50. The helper monitors training loss with
ReduceLR plateau patience100/factor1/cuberoot2/minlr1e-4, rather than blindly
copying the paper's prose about reductions every100epochs. It returns inverse
class-frequency weights as a legacy array and uses label-dependent categorical
dimensions; future native class-weight mapping/fixed12 taxonomy must be
explicit. Its normalization flag defaults false although the article discusses
training-statistic normalization. None of these unresolved future real protocol
choices has been filled in by reading diagnostic/user7–8 labels.

## Genuine synthetic checks and repaired assumption

An independent NumPy masked567-step LSTM calculation matches the configured
native sigmoid model, and independent NumPy GAP/Dense/ReLU/sigmoid calculations
match both SE blocks. Three native updates, full custom checkpoint capture,
two more updates, fresh restore and those two updates replay exactly across
model/Adam/BN/metrics, global RNGs, layer RNG variables and module training
flags. Native `.keras` weights and optimizer reload exactly; its layer RNG
states do not match, so native model files alone are not claimed as training
resume artifacts. Custom complete state is required. There is no epoch-loader/
callback or real-data auditor implemented for this candidate yet.

Attempt01 incorrectly expected default Dropout's layer counters to advance.
The pinned Keras Torch backend explicitly uses the unseeded Torch dropout
fast path and global Torch RNG, leaving these layer counters constant. Only
the check was corrected to require global RNG change and unchanged layer
counters while still saving/restoring both. The original model/Dropout seed,
optimizer/backend and all exact state comparisons were unchanged. Original
attempt source/log and ENGINEERING_REPAIR.json retain the fault. Attempt02
passed in4.514796seconds, an engineering duration rather than a CPU cost
result. No UWB execution or accuracy failure occurred.

All1423 current science/runtime hashes and19 completed H-Inception artifact
hashes remain unchanged. Its existing auditor/controller continue; no score
SUMMARY was parsed. Current readiness refuses before next actual selection or
cost. Source helper/runtime references were reverified without installing or
changing global packages. Original789 and actual arrays/IDs/models/binaries
are not uploaded to Git.

Reproduce after fetching the nine pinned original files to their local-only
directory and restoring the referenced isolated runtime/state helper:

```bash
python /workspace/uwb-mlstmfcn-source-probe-20261031/synthetic_check.py
```

First complete existing H-Inception audit/private modelset preservation/gate.
A pass requires source confirmation; verified exact closure permits one actual
next fixed cost policy/config/seed and remote-preregistered UWB runner/auditor.
Prepared Disjoint-CNN remains the prior queued candidate; this source-only
MLSTM-FCN preparation is an additional option, with no real ordering or
selection inferred from unseen scores. Historical all8 development remains
exploratory; users7–8 stay excluded from new selection. The user's accuracy/
contribution goal remains unresolved and existing continuation stays enabled.
