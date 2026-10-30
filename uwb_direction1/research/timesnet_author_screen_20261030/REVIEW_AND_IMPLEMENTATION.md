Next published baseline: Wu et al., TimesNet: Temporal 2D-Variation Modeling
for General Time Series Analysis, ICLR2023. Author primary references:
https://github.com/thuml/TimesNet
https://openreview.net/forum?id=ju_Uqw384Oq
https://arxiv.org/abs/2210.02186

Author landing repo2b39e14cd16448e0ed94e4df81c4f5dc86f658d9 now contains
README directing code to thuml/Time-Series-Library; direct model/script fetches
there returned404, not an experiment failure. Canonical author code pinned at
4e938a1767106324dd753b2a44832bf870a0252e,12fullsource/license blobs verified by
GitblobSHA1+SHA256. MIT license. Read complete TimesNet core, Conv_Blocks,
Embed, classificationdriver, run arguments, official classification script,
UEA data/normalizer/collation/factory and EarlyStopping helper. Primary paper
search/open attempts returned empty content/isErrortrue without an explanation;
paper full text/formulas/figures have not been obtained in this turn. Do not
claim a full paper review. No network bypass or invented provider failure cause.

Use exactly the official UWaveGestureLibrary gesture preset:2TimesBlocks,
d_model32,d_ff64,top3periods,6kernels1/3/5/7/9/11 in each Inceptionblock,
batch16,learningrate.001,RAdam(defaultdecay0),CE,clip4,30epoch cap/patience10.
Input90frames/567variables/12classes are shape adaptations; model2433260params.
This is the complete current author gesture preset, not a manually reduced
generic CLI512width or a claim that all dataset-specific presets are identical.
Choice is based on the named gesture task before real TimesNet UWB values;
no search among published presets on UWB scores. UWave is an accelerometer
dataset and is not an independent UWB cohort. Seed20261029/CPU/RMS/source-user
roles are explicit adaptations, not reproduction of original dataset/GPU study.

Implementation imports original core/embedding/conv code without alteration,
extracts exact author training AST while removing logging/counters only, and
uses original EarlyStopping AST. Three synthetic original-loop updates match
all model/RAdam/loss/allRNG states bitwise. Full restore matches the next2steps
including stopper states/RNG. NumPyFFT independently matches three cases.
All53active optimizer states and unused temporalembedding parameter retained.
Original stopper accepts equal-best scores as a new save/reset, stops only
after strict declines exhaust patience; synthetic tests verify this. No scheduler
is used by classificationdriver. A synthetic logging float(tensor) conversion
warning was retained in the log; it did not alter gradients or optimizer state.

FFT frequency ranking averages input amplitudes over the batch. Original
batch-context behavior is preserved; synthetic16-vs-two8 batches differ in
logits by max.016018629. This is not independent per-record prediction. Any
real protocol must prospectively fix label-blind validation/diagnostic ID order
and batch partitions, log them and replay the same partitions. Do not search
prediction batches or tune them by diagnostic labels. Original driver reuses
TEST for validation/early stopping; adapt selection exclusively to user2 and
freeze held1 predictions before its labels, never copy original TEST-label
selection. Fixed RMS replaces original dataset normalizer; no real values read.

Synthetic cost8train+4eval fullbatch exit0/no timeout:5.834896supervisorwall/
5.817838childCPU. Conservative full30epoch8820update estimate7386.213932s
passes43200s CPU/wall. Cost process capped600s. All4700records retained each
epoch (294batches, no drop_last), not shortened to fit budget. No automatic
size/epoch rescue. Latest STATUS/memories and original789 text search finds
no exact TimesNet/FFT_for_Period/classTimesBlock match; bounded prior-run check,
not an exhaustive equivalence or novelty proof. Full source-user protocol,
real driver/guards/completion audit and remote preregistration are still needed.
No TimesNet real UWB arrays/labels/fits or training launcher yet. Accuracy and
originality evidence separate; published prior art does not stop baseline tests.
