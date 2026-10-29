# TimeMIL CPU adapter and fixed64epoch source pilot

This is a published-method accuracy baseline transfer, not an original contribution. No equivalent completed TimeMIL run was found in the latest STATUS/memory records or the original snapshot's searchable md/py/json/txt files. The historical InceptionTime review and failed generic temporal routes do not test this complete author TimeMIL model. This is a bounded history check, not an exhaustive novelty claim.

## Primary evidence and read scope

Chen et al., *TimeMIL: Advancing Multivariate Time Series Classification via a Time-aware Multiple Instance Learning*, ICML2024, PMLR235:7190–7206. Publisher: https://proceedings.mlr.press/v235/chen24af.html . Official code: https://github.com/xiwenc1/TimeMIL, pinned7542c1c275aae3a9db35ee5c9bdfc0ccf386cd5c.

Read publisher PDF method Sections3.1–3.3, the main implementation paragraph and AppendixF.2–F.4, plus author model/backbone/attention/optimizer/driver files. This is not a claim to have reviewed every proof, dataset or appendix. The PDF was retrieved from the publisher-linked mlresearch GitHub source:7,996,245bytes are recorded precisely in SOURCE_REVIEW.json; SHA2562a0739c41828bcb8228a9201c9872b3d8b161a1147a9d7f3f99e18d9f4e791f1. The UEA benchmark claims do not establish UWB effectiveness.

TimeMIL uses per-time feature embeddings and a class-token pooling mechanism with learnable wavelet positional encoding and Nystrom attention. This is a plausible existing-data baseline for sparse/local discriminative time patterns. Attention and wavelet parameters are not motion labels or physical ground truth; any gain would be predictive evidence only. The matched GAP arm keeps the same initial Inception backbone/head and removes the complete MIL pooling. It does not isolate the individual contributions of wavelets, attention, warm-up or the BCE objective, and does not constitute a new mechanism.

## Source/code discrepancies fixed or preserved before outcomes

- The original main.py uses CUDA-only operations, derives some dimensions from the test data, and selects the best checkpoint by test accuracy. We do not run that driver. Our source-only driver fixes input/class dimensions and uses validation user2 for checkpoint selection. The diagnostic user's labels are indexed only after both arms' predictions freeze; users7–8 labels are not indexed.
- adapter.py changes the author's wavelet function's hard-coded CUDA grid allocation to the parameter's CPU device/dtype. The arithmetic matches an independent NumPy formula. No source model file is edited, but this is a device adapter, not bitwise reproduction of the old GPU environment.
- Main.py uses embedding128 and the backbone emits128; the main paper also says128. AppendixF.2 says512/256landmarks, which is inconsistent with this code path. We use the actual author main configuration128/64landmarks, not a hidden512-to128 rewrite. Full parameter counts:850,973TimeMIL and617,612GAP. There are11author parameter tensors unused by forward/backward; these are retained and listed in the synthetic receipt.
- The author's wave1/wave1_, wave2/wave2_, wave3/wave3_ are distinct Parameters sharing storage. These aliases, including their optimizer update semantics, are preserved and recorded rather than silently corrected. The unused projections/norm/alpha also remain in the full model.
- We use the paper's AdamW lr0.001/weight_decay0.0001, not main.py's lr0.005. StableLookahead preserves the source update rule (alpha0.5,k6,lazy first slow buffer) while using stable parameter indices in serialization. The original optimizer stores slow-state keys as object IDs; our adapter avoids dependence on legacy Optimizer internals and object IDs. An independent vector update comparison and a full stochastic model restart across synchronization boundaries pass exactly.
- The paper allows masking p0 or0.5. We freeze p0(no augmentation) before any new data outcomes. The original optional masking driver computes window length from batch size rather than time length; it is not used. No mask grid is run.
- Warm-up preserves the actual author model literal0.1*MIL+0.99*GAP, and its driver epoch<10 condition (epochs1–9). This differs from the paper's0.01MIL term. The mismatch is explicit; no claim of exact paper reproduction is made.

## Cost decision and frozen experiment

The complete width567-input,128embedding author model passes CPU forward/backward with batch64, finite active gradients, matched backbone/head initialization, wavelet formula checks, preserved aliases, and complete model/optimizer/dropout-RNG restart. Synthetic one-thread timings are3.023729seconds/trainbatch for TimeMIL,0.656492forGAP. The pair300epoch estimate with2xmargin is174,601.108seconds(48.50hours), exceeding the12hour allocation. Before any new real-data outcomes, freeze exactly64epochs per arm:37,248.236seconds with the same margin. No epoch grid is run. A failed64epoch pilot cannot reject the300epoch author setting.

PROTOCOL.json fixes all choices:6900existing quality-retained source IDs; train users3–6(4700), validation2(1000), diagnostic1(1200); seed20261029; both arms64epochs/batch64/same epoch permutations; unweighted12-class OvR BCE; AdamW+Lookahead/clip2; validation present-class macro, then lower BCE, then earlier epoch. Each signed clutter-removed record/view is divided by its own RMS(floor1e-8), with all90frames and189range bins retained as90x567input. No fitted cross-record calibration or resizing. The RMS operation is a declared input adaptation, not a contribution.

Both fitted models, optimizer/RNG states, sample roles, logs and unscored predictions are retained. TimeMIL continuation requires >=0.5pp over matchedGAP and within0.5pp of the stronger prior full/localCNN; GAP separately requires >=1pp over that prior reference for baseline expansion. This permits only a separately frozen source-sixfold study. On failure close this exact configuration without a seed/LR/epoch/normalization rescue grid. All users have development history; a single source-user result is exploratory, not independent validation or eightfold improvement.

## Execution status and safeguards

No TimeMIL real-data fitting has started. The driver was actually invoked against the unfinished predecessor and rejected before creating a run directory or reading project arrays/labels. It requires the current CNN/MultiRocket controller's completed state and passing MultiRocket result audit. The current locked controller is unchanged; no third automatic queue was added.

The launcher reuses the already tested bounded subprocess supervisor, with43000CPUsoft/43100hard and43100wall+termination grace within43200seconds. It retains incomplete receipts/partial artifacts on engineering, numerical or budget failure; such failures are not accuracy negatives. Per-epoch last checkpoints contain stable optimizer, slow buffers and RNG states, but this driver has no automatic restart. Diagnose retained artifacts and account for actual remaining aggregate budget before any later explicit engineering recovery. No new trial or resampling is authorized by a restart count.

Recreate the code dependency by cloning the pinned author repository into vendor/TimeMIL; install only einops0.8.1 into vendor/site-packages and reuse the tested CPU torch2.15.0 runtime. Verify every FROZEN_HASHES entry. When the existing queue has truly completed and been audited, an explicit future invocation is:

```bash
python /workspace/uwb-timemil-20261029/launch.py --run
```

Do not launch while the predecessor is active. Independent functional/metric checking of the eventual TimeMIL result is still required before claiming gains. Raw arrays, model tensors, fitted states and sample IDs remain private in cloud/Drive outsideGit. Overall accuracy improvement and original-contribution goals remain unresolved.
