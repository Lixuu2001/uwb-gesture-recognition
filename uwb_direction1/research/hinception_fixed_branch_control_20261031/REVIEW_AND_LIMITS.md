# Fixed-filter information ablation preparation

The active fixed H-Inception source-sixfold confirmation continues unchanged.
Its single source-held1 positive result does not establish which part of the
pipeline helps. Its LocalCNN/GAP controls differ in input processing, optimizer
and training budget; they cannot isolate the hand-crafted filter branch.

This preparation addresses that attribution question. It is not a new candidate
selection, training protocol or real UWB experiment, and does not replace the
current confirmation queue. A real ablation remains conditional on the complete
source-confirmation gate, full audit/private preservation, and separate frozen
cost/protocol/runner/auditor registration before new fits/diagnostic access.

Primary references rechecked through author repositories:

* Ismail-Fawaz et al., **Deep Learning For Time Series Classification Using New
  Hand-Crafted Convolution Filters**, IEEE BigData2022, pp972–981,
  [DOI10.1109/BIGDATA55660.2022.10020496](https://doi.org/10.1109/BIGDATA55660.2022.10020496).
  Pinned author CF-4-TSC `cf4d0e55ee11b7393ae3d317cdf5f22dcc51ba3e` implements
  17 nontrainable trend/peak convolutions and concatenates their ReLU responses
  only in the first Inception module. Under our disclosed input1→567 transfer,
  these regular convolutions sum variables. They are not radar trajectories or
  independent per-variable fixed features. Existing source-review records cover
  selected paper sections; this turn does not claim a new full-PDF/figure review.
* Ismail Fawaz et al., **InceptionTime: Finding AlexNet for Time Series
  Classification**, Data Mining and Knowledge Discovery2020,
  [DOI10.1007/s10618-020-00710-y](https://doi.org/10.1007/s10618-020-00710-y).
  The author README and complete `classifiers/inception.py` were reread and pinned
  at `952d115e83fb3a66d75c5858c702f8dd5eeb18b7`. Default single network has
  width32/depth6/bottleneck32, effective kernels40/20/10 and two residuals;
  ordinary first-module concatenation has128 channels. The README separately
  identifies the ensemble. Historical project notes already reviewed InceptionTime
  in temporal-scale work. This is a source clarification, not a newly discovered
  paper, standard InceptionTime reproduction or originality evidence.

Simply deleting 17 features changes downstream weight dimensions/initialization
and parameter count. The prepared **zero-information control** instead calls
the unchanged full H-Inception hybrid branch and sets its17 post-ReLU responses
to zero using one registered, weightless layer. It retains all fixed convolution
computation, the145-channel first concatenation, all trainable tensor shapes,
851434total/601006trainable parameters,50trainable tensors and102Adam variables.
The128 learned first-module responses are unchanged at paired initialization.
Full-mode calls the existing native author adapter directly without intervention.

This tests availability of input-dependent information from the fixed branch
while preserving dimensions/initial learnable parameters. Downstream BN still
has17 affine dimensions; its learned offsets may become constants. Therefore
this is not structural branch removal or vanilla InceptionTime. It offers no
inference-cost reduction; the extra zero operation can alter wall time slightly.
Later trained representations are expected to differ. It does not isolate RMS,
channel z-normalization, longer training, optimizer, multi-radar fusion or signed
cross-channel cancellation. Those require separately designed comparisons.

Synthetic checks with seed7/B2 and generated Gaussian values passed:

* Full mode versus original native H initial and3updates: all native model/Adam/
  BN/metrics/global RNG states exact.
* Full versus zero at paired initialization: identical native states and every
  variable shape, all17 fixed kernels retained; original128 learned first-join
  features exact and only17 fixed response channels zero.
* Fixed kernels unchanged after3updates; saved zero state and next2updates restore
  all model/Adam/BN/metrics/global RNG and module modes exactly.
* Native registered-layer `.keras` safe reload preserves weights, optimizer and
  predictions exactly; no arbitrary Lambda deserialization. Unknown modes refuse.

These are engineering checks, not measured accuracy, a formal UWB CPU profile,
callback/complete-run audit or whole-job resume proof. Source constructor defaults
1500epochs/B64 are merely retained by the architecture helper; no real ablation
epoch/seed/batch budget has been selected. Synthetic timing is not a cost gate.
No raw/maps/source-ID/label arrays were read by these checks. Current1465 frozen
training dependencies remain unchanged. Public source plus GPL-3.0 license and
references are retained; actual data/models/ID values remain outside Git.

All source users have historical development exposure, so future comparisons
remain exploratory. No users7–8 labels are used for new selection. Published
methods and a useful ablation do not by themselves establish a contribution.
