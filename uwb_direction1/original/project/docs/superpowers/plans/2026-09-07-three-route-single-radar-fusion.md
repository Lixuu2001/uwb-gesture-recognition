# Three-route single-radar fusion implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for bounded implementation/review tasks and execute experiments sequentially. User approved all three routes in this conversation. Do not ask again between routes.

**Goal:** Test whether missing-aware fusion, shared-specific learning, and prototype completion improve held-user Right-only classification, in that order.

**Architecture:** Reuse each fold's frozen Right amplitude-plus-phase encoder and classifier. Pool temporal/spatial tokens, fit a small residual classifier using training-only extra views, and retain the original Right logits as an anchor. All variants in a stage share preprocessing, initialization, minibatches, training budgets, and validation selection; this is a bounded frozen-feature adaptation study, not a faithful end-to-end reproduction of the cited papers.

**Tech Stack:** Existing PyTorch environment, NumPy caches, eight quality-filtered LOSO folds, local CUDA.

## Evidence and protocol

- New files only under `experiments/p1_20260905/missing_view_fusion/` plus this plan. Existing dirty files and outputs are retained; dedicated branch and experiment directory isolate additions without copying the large untracked research tree.
- 9300 quality rows, 12 logits, eight held subjects. Reuse `pipeline.split(...,2026)`; source-person 10% validation, held-person test. This is NOT a newly independent validation-person protocol. Prior repeated use of these eight users makes this a development benchmark, requiring external confirmation for final generalization claims.
- Frozen encoder source: `results_clean/sv_R_label_global_s2026_h{fold}.pt`; verify prediction train/val/test IDs. All arms use identical source, trained on Right only. Initialization origin is separate from extra-view fusion supervision.
- Inputs amplitude + phase concatenation: `[B,1,90,378]`; pool temporal into 10 bins, feature-axis tokens into 42 bins, each 32 channels: 1664 dimensions. No calibrated physical velocity claim.
- Read Left/Top raw arrays ONLY for outer training rows. Cache validation/test Right features and logits only. Train-only per-view mean/std; test Right scaler only. Deployment API accepts rank-2 Right features plus original Right logits; no hidden-view input.
- Readout is `original_logits + correction`. Last correction layer starts at zero. Same-capacity Right-only controls remove the explanation that extra head capacity alone produced gains. This residual adapter changes the previous plain-fuser proposal to preserve the established classifier and isolate incremental fusion value.
- Eight folds, model seed 2026 first. Model seed 2027 confirms passing candidates and their controls. Batch 256, Adam lr 0.001, weight decay 0.0001, 40 epochs, minimum 12 epochs, patience 8, best Right-only validation hard CE. Training base loss matches existing global smoothing objective. Sampler seed 7777+fold, separate mask seed 31337+fold. Normalize subset classification loss by number of passes.
- Gate: paired mean gain >=2 percentage points and >=6/8 improved users versus matched Right-only control; also compare frozen anchor and strongest historical reference. Gate is investment criterion, not statistical significance. Save all per-fold effects, F1, confidence intervals, parameter count, timing, IDs, logits, checkpoint/config/code hashes. No test-driven hyperparameter search.

## Task 1 — data contract and frozen extraction

- [ ] Add tests that pooling preserves separate amplitude/phase token blocks; train/val/test sets disjoint; hidden-view deployment inputs rejected; held-view values cannot affect deployment output.
- [ ] Implement `data.py`: import frozen experiment code with explicit paths; verify checkpoint/split lineage; extract train three-view and val/test Right-only tensors in small batches; compare loaded classifier predictions to saved reference; save atomic caches and manifests.
- [ ] Execute one-fold extraction, verify cache identity and original test accuracy, then reuse fixed extraction for all routes.

## Task 2 — route 1 missing-aware fusion

- [ ] Test expected failures before implementing `models.py`: missing marker gradients, all subset gradient paths, Right-only deployment, finite supervised sigmoid contrast, equal initial residual predictions.
- [ ] Implement fusion MLP `[3*1664,256,128]`, zero-initialized 12-class correction, optional learned missing vectors. Four target-containing masks in L/T/R order: 001,101,011,111.
- [ ] Arms: `R_ONLY`, `RANDOM_ZERO` (four independently sampled masks per step, compute-matched), `JOINT_ZERO` (enumerate all four), `JOINT_TOKEN`, `JOINT_CONTRAST` (+balanced supervised sigmoid loss on normalized fusion codes, weight 0.05, temperature 0.2). Frozen original classifier is reference, not a trained arm. Record adaptation from the two-modality MICCAI method; do not label generic balanced sigmoid loss an exact reproduction.
- [ ] `runner.py` trains every arm with identical IDs/seed, validates only Right, exports state/scalers and predictions. Smoke run to separate output directory before full first-route eight-fold run.
- [ ] `report.py` recomputes saved logits/accuracy/selection and paired effects. If a candidate passes, run a same-user/class global deranged pairing control and second model seed; otherwise stop broad tuning and move to route 2.

## Task 3 — route 2 shared-specific features

- [ ] Add tests and implement `SharedSpecific`: shared projection 1664->128, separate per-view private projections 1664->128, per-view fused code `shared + fusion(shared,private)`, missing slots filled by mean available shared codes. Head consumes three slots and outputs residual correction.
- [ ] Arms `SS_R_ONLY`, `SS_JOINT`, `SS_ALIGN`: same architecture; target-containing subsets jointly trained; additional shared alignment plus private view-ID CE in SS_ALIGN (each weight 0.05). Same-person same-instance cross-view shared MSE is an adaptation, not exact ShaSpec domain-confusion replication. Keep private features unconstrained by cross-view equality.
- [ ] Eight folds after route 1 audit, same gate and seed confirmation policy. If frozen adapters fail, conclusion is limited to this feature/adapter family; do not claim full end-to-end ShaSpec ruled out.

## Task 4 — route 3 prototype completion

- [ ] Add tests and implement `PrototypeCompletion`: train-only 12-class per-view prototype banks; Right MLP predicts 12 soft weights, aggregates missing Left/Top prototypes, fuses with Right code and classifies through zero-start residual head. No true labels supplied to deployment; dictionary fitted strictly on training rows.
- [ ] Arms `PROTO_R_ONLY` (same-size dictionaries formed only from Right features), `PROTO_MULTI` (Left/Top banks), `PROTO_MLP` (same-size unconstrained output dictionary initialized from Right bank). All weights, training, deployment inputs identical apart from dictionary source/trainability. Report unequal trainable parameter count for MLP capacity control explicitly.
- [ ] Eight folds after route 2 audit. Compare multi vs Right dictionary first, then strongest anchor; no extrapolation from retrieval papers to radar without evidence. This is a class-prototype adaptation of PMCH's weight-based prototype completion, not a PMCH reproduction.

## Task 5 — verification and record

- [ ] Run model/data/trainer integrity tests and independent review before full queue.
- [ ] Preserve per-epoch logs, current stage, completion/failure status and exact rerun command. Resume only if code/config/cache hashes agree; changes require a separate output directory.
- [ ] Complete sequential initial screens and report actual outcomes. If elapsed compute spans an interaction, distinguish implemented, running, completed, and not started; do not silently abandon later stages or fabricate completion.

## Primary method sources

- Missing-aware fusion: https://papers.miccai.org/miccai-2025/0483-Paper2038.html
- Shared-specific: https://arxiv.org/html/2307.14126v1
- Prototype completion: https://www.ijcai.org/proceedings/2025/281

Code changes, model seeds and extraction hashes are recorded in each run; this plan is frozen before new test results.
