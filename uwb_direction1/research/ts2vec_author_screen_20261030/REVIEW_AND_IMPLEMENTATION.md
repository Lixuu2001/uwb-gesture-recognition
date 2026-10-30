TS2Vec's complete default encoder, original two-step synthetic fit, independent
loss checks and author CLI SVM protocol are now executable in this cloud. A
full-size synthetic resource profile passed:963.335652seconds conservative
estimate for600 encoder updates, all6900 source encodings and the full4700-record
ten-C/five-fold SVM plus2200 inferences. This is a feasible next published-method
candidate; no real UWB values, IDs, labels, fitting or accuracy scores were used.
The current TSLANet experiment remains the only active real training job.

Paper: Yue etal., *TS2Vec: Towards Universal Representation of Time Series*,
AAAI36(8):8980–8987(2022), DOI https://doi.org/10.1609/aaai.v36i8.20881.
Primary PDF: https://cdn.aaai.org/ojs/20881/20881-13-24894-1-2-20220628.pdf.
Read scope: selected Method passages on architecture/contextual consistency,
hierarchical contrasting and Algorithm1 on8981–8983, plus Table1 caption and
publisher metadata. Not the whole eight-page paper, figures or appendix.

TS2Vec aligns overlapping timestamps under independently masked/cropped contexts,
with instance and temporal contrastive objectives evaluated at several pooling
scales. Its existing representation-learning design is relevant to gesture time
series and differs from supervised frozen-head penalties and masked reconstruction.
It does not establish a new mechanism or improvement on this UWB corpus. Published
benchmark numbers are not protocol-matched accuracy evidence here.

The paper's GitHub alias `yuezhihan/ts2vec` currently resolves through repository
metadata to canonical `zhihanyue/ts2vec`. The canonical main commit is pinned to
`b0088e14a99706c05451316dc6db8d3da9351163`. Twelve source files and the MIT license
were materialized as exact UTF-8 bytes and independently checked against Git blob
SHA1/SHA256. Original model, loss, augmentation and fit methods remain unchanged;
the adapter imports the source without importing the dataset/CLI driver. History
checks of the latest remote STATUS/two memories and original789-file snapshot
identified no TS2Vec/TSEncoder/hierarchical_contrastive_loss implementation in that
bounded scope. This is not an exhaustive originality search.

Literal code findings to preserve in a future real protocol:

- Constructor defaults:output320, hidden64, depth10, lr.001, batch16,
  temporal_unit0 and max_train_lengthNone. With567 input variables it has673,472
  trainable parameters. The hidden list plus output block yields11 actual
  residual blocks; it must not be silently relabelled as ten executed blocks.
- The current CLI defaults batch8 and max_train_length3000. This prototype uses
  the full constructor's batch16 and no length cap, with90-frame inputs. The cap
  difference has no effect at90frames. CPU and seed20261029 are explicit task
  adaptations, not an exact original CLI/GPU experiment.
- With neither epochs nor iterations provided, the original fit sets200 updates
  for train_data.size<=100000,600 otherwise. For4700×90×567 it selects600
  **iterations**, not600epochs. This exact expression was evaluated from the
  pinned AST;600 real updates have not run.
- Original fit uses AdamW(lr.001), current default decay.01, original independent
  binomial timestamp masks, dropout.1, overlapping random crops and no clip.
  Its cumulative AveragedModel includes the initial network:one initial sample
  plus one after every update. The original full-series max pooling is retained.
- `train.py` explicitly requests the RBF-SVM evaluation protocol (Table1 caption
  also identifies SVM). The helper function's optional default linear path is
  different. The SVM uses unscaled representations, the original ten-C grid
  including infinity, five-fold training-only CV and final refit. Its only adapter
  change is GridSearchCV n_jobs5→1; all grid/solver/selection settings stay intact.
  A synthetic comparison to the original AST using a parallelism-only constructor
  wrapper produced exact support vectors, dual coefficients, decisions and labels.
  Infinity C works on sklearn1.8. The unused old linear helper's multi_class='ovr'
  keyword is incompatible with sklearn1.8 and is not claimed reproduced.
- Original save/load contains only the averaged encoder. Before a real run, a
  full driver must preserve raw and averaged models, optimizer, all RNG states,
  iteration/sampling position, sample IDs/logs and class readout. Repeated calls
  to fit recreate its local optimizer and are not an exact continuation.

Synthetic checks passed five independent NumPy instance/temporal/hierarchical
loss cases, including odd lengths and B=1/T=1, plus finite-difference gradients.
The first test incorrectly assumed float64 hierarchy accumulation even for
double inputs. The original initializes a float32 scalar and accumulates in place;
the independent reference was corrected to reproduce every float32 rounding
boundary exactly. Original source was not changed. The original attempt failed
before model construction/fitting; its script/trace/exit1 receipt are retained.
Corrected attempt02 passed, exit0 in4.026840wall/3.979872CPU seconds. Its original
full-width fit performed two synthetic updates with finite gradients for all
parameters and48 changed state tensors. The average matched initial+two updates,
and full-series encoding matched direct max-pooling of the averaged encoder.
It is not a full stochastic training-resume proof or a real accuracy result.

Readout parity/compatibility check passed in.167791seconds. No real labels were
used for its grid. The original UEA loader and original driver were read for
protocol differences; neither was executed. Existing source-user split/RMS/input
adaptations still need prospective registration, without users7–8 labels.

`COST_POLICY.json` and implementation hashes were locally fixed before timing,
not remotely preregistered as a real-data experiment. One warm-up and three timed
batch16 full90-frame two-view updates retain independent masks/dropout and original
hierarchical loss, AdamW and initial-inclusive average. These full contexts bound
the variable shorter random-crop work for a conservative profile; they are not
substituted into the future real method. Encoding uses the actual author encode
method. One full original ten-C/five-fold/refit SVM was timed on synthetic4700×320
features, with2200 predictions/decisions. All stages used one thread/job.

| Timed component | Actual measurement |
| --- | ---: |
| Max full batch16 two-view update | .399574533s |
| Max full batch16 full-series encoding | .034078730s |
| Entire4700×320 ten-C/five-fold/refit SVM | 163.752859482s |
| 2200 SVM predictions and decisions | 3.448235645s |
| Conservative600update+6900encoding+readout estimate | 963.335652492s |

The estimator is2×all terms+120s reserve; gate43200wall/CPU passes. The cost
supervisor exited0 without timeout:173.521906wall/173.428167CPU seconds, peak
RSS600208KiB. This measures synthetic workloads, not complete600step training
or real representation difficulty; a later real run still requires independent
hard resource limits. No width/epoch/seed search or synthetic accuracy selection
was performed. The optional linear helper was not used as a fallback.

One execution read transiently failed with exec-server transport disconnected;
the managed environment recheck reported running/connected, retry succeeded and
the original TSLANet process remained live. No environment configuration or
training restart occurred. Source alias lookup and numerical-reference correction
are retained in `PREPARATION_ERRORS.json`.

Next: finish/audit/preserve the currently registered TSLANet pair first and apply
its fixed source gate. If it licenses confirmation, complete that confirmation.
Otherwise this TS2Vec core is a CPU-feasible next candidate for a separately
frozen real source-user protocol and complete resumable driver/IDs/logs/audit.
There is no TS2Vec real-data launcher, registered real protocol,600step checkpoint
or real accuracy result yet; do not claim or auto-launch one. Preserve all prior
negatives. Accuracy improvement, originality and independent validation remain
separate and unresolved; historical development makes source results exploratory.
