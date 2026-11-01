# MLSTM-FCN complete documented8-cell native CPU cost policy

Proceed only after the completed H-Inception CPU100 source6 failure and verified
full private six-model preservation, and the completed Disjoint-CNN full500/B8
resource failure. The new closure guard verifies their exact metadata/hash
bindings before policy read, model import, directory creation or timing. No
existing experiments, source files, runtime packages or closure guards change.

Use the pinned published MLSTM-FCN8 variant: Conv128/256/128 with kernels8/5/3,
both SE16 blocks, Masking, LSTM8 and Dropout0.8, 860364total/859340trainable
parameters. Input metadata567variables×90timepoints preserves the original
dimension shuffle: LSTM scans567steps while CNN scans90. No recurrent/conv
shrinkage, stride or branch removal. Select one fixed **paper/README250epoch,
B128, seed20261029** policy, no8/64/128-cell sweep or smaller-budget fallback.
The gesture script's1000 and Uwave500 epochs are distinct source choices;
250 is not misrepresented as that script's default. Native Keras3.11.3/Torch
LSTM sigmoid differs from old Keras2.1.2 hard_sigmoid; no legacy TF numerical
equivalence claim. Native adapter/source exact hashes must match preparation.

Synthetic-only profile: 8 weighted training batches of128×567×90,4unweighted
eval batches,3native model saves,3custom full model/Adam/BN/metrics/global and
layer RNG/module-mode saves. Inverse-frequency weights use generated synthetic
12-class counts and a fixed12-output head; no real class counts or UWB values
are accessed. Future input intent is full signed RMS with no added
normalization/augmentation, matching helper normalize_timeseries=False.

Before timing publish and exact-readback POLICY, source/runtime and previous
closure bindings, profiler, arithmetic/tests, guard, independent supervisor,
driver and documentation. Independent child limits590/600CPU and620wall;
no automatic retry. Fit cost uses actual8 training/4 eval/3+3save vectors'
maxima, full4700/1000/1200 source counts and worst-case best+last snapshots:

`2*(250*37*maxTrain+(250*8+8+10)*maxEval+251*maxNative+504*maxFull)+120`.

The fixed run must fit43200s and14GiB RSS. A pass only permits separately
registered actual-UWB epoch runner, selection protocol and complete auditor;
this cost policy authorizes no real-value or label loading. A failure closes
exact250/B128 native policy without accuracy inference or rescue. Existing
synthetic exact native state-restoration tests support model compatibility;
there is not yet a real epoch/callback loader or auditor for this candidate.

Literature: Karim et al., *Multivariate LSTM-FCNs for Time Series Classification*,
Neural Networks116(2019)237–245, DOI10.1016/j.neunet.2019.04.014, arXiv1801.04503v2.
Previously reviewed architecture, training and ablation sections remain the
paper basis; no new full-paper review is claimed. titu1994/MLSTM-FCN commit
85100ccfec7ee08106b3a25906a0c9c707e1b7ab and paper-linked houshd repo84a2e6...
share gesture model blob6d425bda53e84b0c9d6c620a016f29b903807ae6. Original source
license metadata was null, so source remains local-only; exact fetch index,
adapter and original repository references already exist in Git. No original
source redistributions, UWB arrays, real IDs or model files go into this commit.

Published-baseline accuracy value and originality are separate. All source
development is exploratory because of historical reuse; users7–8 labels are
excluded. The research goal is unresolved; continue the existing automation.
