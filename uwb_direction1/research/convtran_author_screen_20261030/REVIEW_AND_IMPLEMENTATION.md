ConvTran now has executable, byte-pinned author cores and synthetic evidence in
this cloud environment. Its two prospective CPU profiles exceed the fixed
12-hour single-thread resource ceiling, so neither is eligible for a real-data
launch under this policy. This is a resource result, not an accuracy rejection.
The original TSLANet experiment is still running without configuration changes.

Paper: Foumani, Tan, Webb and Salehi, *Improving position encoding of transformers
for multivariate time series classification*, Data Mining and Knowledge Discovery
38, 22–48 (2024), published online 2023-09-05,
https://doi.org/10.1007/s10618-023-00948-2.
Read scope: publisher HTML sections 4.1–4.3 and 5.3, plus relevant metadata. The
whole paper, figures, statistical comparisons and all dataset tables were not
reviewed. No paper PDF has been locally preserved in this study.

The paper motivates length/dimension-scaled absolute time encoding (tAPE) and
input-independent scalar relative position encoding (eRPE). Its combination with
temporal and cross-variable convolutions gives an existing classification
baseline worth transferring. This does not establish that it improves UWB
recognition, nor a new contribution. The reported benchmark results cannot be
compared numerically to this project's source-user or legacy eight-fold results.

Author source https://github.com/Navidfoumani/ConvTran is pinned at
`148afb6ca549915b7e78b05e2ec3b4ba6e052341`. Twelve reviewed files were fetched via
the GitHub connector, materialized as exact UTF-8 bytes, and independently checked
against their Git blob SHA1 and SHA256. The MIT notice is retained. Einops0.8.1
is copied from the already pinned TimeMIL dependency bundle with 33 file hashes
and its license/metadata; no package installation or dataset download occurred.

`author_core.py` executes only the selected original AST class bodies, preserving
expressions, operations and constructor/forward methods. It avoids importing
the author's dataset/download/reporting driver and unused pandas dependencies.
All 567 variables (3 views × 189 range bins) and 90 time points are retained for
the synthetic input shape. That channel organization and any future per-view RMS
would be explicit task adaptations, not an exact paper experiment.

Important findings from the pinned code, retained rather than silently repaired:

- Paper-width64 author core has 9,340,708 parameters. Its temporal expansion is
  `4*embedding=256` filters, followed by a spatial convolution across all567
  channels. The paper describes64 temporal filters. Thus this is the code at
  paper embedding width, not an exact paper architecture reproduction.
- `main.py` defaults to embedding16, FFN256, 8 heads, dropout.01, tAPE+eRPE,
  batch16, 100 epochs. `cli_core.py` parses and asserts these literal defaults;
  its model has592,324 parameters and64 temporal filters. It is an official CLI
  configuration, not an accuracy-selected smaller modification.
- The actual driver selects original RAdam at lr.001/weight_decay0, despite the
  paper's Adam description, and clips gradient norm at4.0. The original RAdam
  class is executed unchanged. No optimizer substitution is made.
- Attention uses embedding-wide `embedding**(-.5)` scaling and adds the learned
  relative bias *after* softmax. It preserves the source residual from the
  convolution embedding, its ReLU FFN and GAP-only classifier. These must not be
  relabelled as a conventional Transformer or silently changed to match a diagram.
- The actual checkpoint path loaded by `main.py` is saved by minimum validation
  loss on every epoch in `train_runner`, despite a separate key-metric path and
  a nominal validation-interval flag. There is no active early-stop break in
  that fixed-loop function. The UEA loader uses a record-stratified validation
  split; future user-disjoint selection would be a declared protocol adaptation.

History check was bounded to the latest remote STATUS and both memory files at
commit4ba042e, plus keyword search through the original789-file project snapshot.
No ConvTran/tAPE/eRPE implementation or real result was identified there. Earlier
TimeMIL/GAP Inception convolutions and TSLANet's spectral blocks do not execute
this core. This is evidence against an equivalent rerun in the checked scope,
not an exhaustive novelty search or proof of absence from every unindexed file.
The existing same-corpus local/global/fusion prior art still applies. Published
ConvTran is treated as a potential known accuracy baseline, never new architecture.

Full64 synthetic checks passed: independent NumPy tAPE and nonzero-bias eRPE
forward calculations, exact relative-index layout, original loss-wrapper mean
versus CE, finite gradients for every parameter, all24 parameter tensors changed,
and two stochastic optimizer steps replayed bitwise after a six-step checkpoint
(including post-rectification RAdam state, all model/BN buffers and Torch RNG).
The full synthetic test used batch2, not16; it is not a resource or accuracy test.
Supervisor exit0:12.757225wall/12.704269CPU seconds. CLI16 separately passed
its literal-default assertions, positional formula, finite gradients, all24
parameter changes and the same6+2step stochastic resume. Its test took2.563689s.

Resource policy `COST_POLICY.json` was locally frozen with hashes before timing,
not remotely preregistered as a real-data experiment. The fixed order prefers64
if affordable, otherwise tests only the literal official CLI16 defaults. It
forbids further widths, batch/seed/epoch reductions under this screen. Both use
one warm-up plus three timed full batch16 steps for training and evaluation;
the estimator uses twice the maximum steady full-batch times for100 epochs of
4700train/1000validation records plus1200diagnostic inference and120s reserve.
No actual UWB map, ID or label values were read in either profile.

| Configuration | Parameters | Max train batch16 | Max eval batch16 | Conservative100epoch estimate | 12h gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Paper-width64 author code | 9,340,708 | 7.951136s | 2.995557s | 505,840.138555s (140.51h) | fail |
| Literal CLI16 defaults | 592,324 | 1.484511s | .551257s | 94,437.766924s (26.23h) | fail |

Cost supervisor exited0 without timeout:68.872178wall/68.757109childCPU seconds.
Peak RSS was4,637,044KiB. Timing occurred while the original TSLANet worker used
another CPU; all five CPUs were not dedicated to this probe. These conservative
prospective estimates are not measured complete-training durations. Passing or
failing this cost gate says nothing about accuracy. No ConvTran UWB fit, held-user
score, source-confirmation protocol, model checkpoint or real-data launcher exists.
No smaller/shorter configuration was promoted after an accuracy result.

The full64 and CLI16 implementations, source hashes/licenses and real synthetic
logs remain reusable. Do not automatically run either after TSLANet. Future
hardware or a separately justified, prospectively registered resource plan would
need a fresh gate, complete source-only protocol, IDs/logs/checkpoints and no
users7–8 label use. Current work should continue the existing TSLANet experiment,
complete its guarded audit/private preservation, and then move to another
published method that passes an actual CPU feasibility test. Accuracy evidence,
originality evidence and resource eligibility remain separate.
