# H-Inception full author scale with explicit multivariate input adaptation

This is a different published architecture after the exact full-LITEMV native
Torch resource policy failed, not a reduced LITEMV or an accuracy claim.
Source: MSD-IRIMAS/CF-4-TSC
`cf4d0e55ee11b7393ae3d317cdf5f22dcc51ba3e`, model blob
`bb8c08bf24cd3b3dcdfd90f079403900ab4df1eb`. Seven unedited source files retain
Git-blob SHA-1/size/SHA-256 in SOURCE_INDEX.json and full GPL-3.0-only license.
This adapter and checks have the same GPL-3.0-only terms.

Primary support: Ismail-Fawaz et al., *Deep Learning For Time Series
Classification Using New Hand-Crafted Convolution Filters*, IEEE BigData2022,
DOI10.1109/BIGDATA55660.2022.10020496. Author PDF:
https://maxime-devanne.com/delegation/publis/ismail-fawaz_BigData2022.pdf.
Previously read abstract/introduction,III.A–D,IV.A andIV.B opening. Ten-page
PDF was read through the web tool, not locally materialized/SHA-pinned; not
all results/tables/figure visuals were reviewed. The paper defines fixed trend
and peak filters alongside learned convolution features. H-Inception retains
the Inception residual architecture; H-InceptionTime is a five-model ensemble.
Neither precedent establishes a UWB accuracy gain or new contribution here.

The current author source fixes Input to `(length_TS,1)` and targets univariate
UCR. The adapter explicitly changes **one input-channel literal1→567**, uses
26 native Keras bindings in five unchanged architecture methods, and changes
one checkpoint suffix `.hdf5→.keras`. No TensorFlow stub, narrower width,
shallower depth or replacement Torch architecture is used. This is a
multivariate transfer, not unmodified original TensorFlow2.4.1/UCR reproduction.
The original bottleneck code already supports multivariate input; that branch
is exercised under the disclosed input adaptation.

All source defaults remain:32 filters, bottleneck32, depth6, kernels40/20/10,
two residual connections,17 fixed custom filters, batch64 and epochs1500.
The fixed filters are regular Conv1D: each sums over567 variables into one
output, giving17 fixed features, not LITEMV's9639 per-variable fixed features.
First inception concatenation has128 learned plus17 fixed features. This is
the published source's summation rule, not a newly invented channel reduction.
It may cancel signed variable responses; no radar-trajectory or robustness
meaning is assumed. Only a future registered real accuracy experiment can
answer whether the different architecture helps.

Actual adapted model:601006 trainable parameters,851434 total including fixed
kernels/BN,50 trainable variable tensors and102 native Adam variables. Exact
analytical values of all17 fixed kernels and nontrainability are checked;
after three updates they remain unchanged. An independent NumPy even-SAME
fixed convolution verifies summation over variables. Complete native model,
Adam,BN,metrics/all RNG recovery and next two updates match exactly; native
`.keras` weights and optimizer reload match. See SOURCE_PROBE.json.

The separate fit check exercises three synthetic epochs/six updates/B4/last2,
native fit versus explicit native batches, original callbacks/train-loss
checkpoint, epoch1 restoration/next2epochs, plateau251constant-loss rule and
strict train-loss minimum/ties-first/validation ignored. Its actual result and
scope are in FIT_SEMANTICS_CHECK.json. Native DataAdapter global-RNG parity
with an explicit order is not asserted. Source fit's separate train/validation
OHE and test-dependent label dimensions are not copied; any future runner
must use the fixed12-class taxonomy and exclude users7–8 labels.

The original fit uses min(batch_size,ntrain//10); at4700 training records this
is64. Synthetic B2/B4 fixture updates are engineering checks, not a new real
batch/seed/epoch budget or evidence about1500 epochs. Single H-Inception is
prepared; five-model H-InceptionTime is not implemented or claimed.

Native Keras3.11.3/Torch CPU runtime is reused without global installation.
The author README requires Keras>=3.6 to keep filters fixed; actual immutability
is tested. Generic snapshot primitives reuse the pinned LITEMV native helper,
which constructs no LITEMV instance here. Their source/dependency manifests
and1302 runtime-file/10wheel checks remain separately pinned. Original legacy
requirements and Dockerfile are retained for provenance, not executed.

History: no exact H-Inception/hybrid Inception/CF-4-TSC/HCCF markers in the
original789; latest remote mentions are prior-art identification, not a real
H-Inception run. Lexical checks are not exhaustive equivalence or novelty
proof. Prior old Inception/dilation or reduced models do not count as this
full six-module source architecture; do not claim an unrun ensemble.

Next freeze one prospective full-architecture cost policy after prior closure,
then only if feasible implement/register/read back the fixed source-user
runner/auditor before actual UWB values/labels/fitting. No new real experiment,
formal cost or accuracy result is claimed by these synthetic checks. All8
historical development remains exploratory. Known published accuracy value
and originality stay separate; the research goal is unresolved.
