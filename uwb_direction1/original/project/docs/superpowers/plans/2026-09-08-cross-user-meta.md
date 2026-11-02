# Cross-user multiview meta-training implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Execute continuously under the user's explicit request to continue until verified results.

**Goal:** Implement and evaluate A0–A3 with strict six-user training, held-user validation, and Right-only deployment; quantify multiview contribution with matched paired effects.

**Architecture:** Retain the original Right Classifier encoder and flattened-token head. Add a training-only normalized pooled three-view auxiliary classifier. An exact differentiable SGD inner step distinguishes A1/A3 from A0/A2; all deployed weights remain the unadapted learned Right classifier.

**Tech Stack:** Existing Windows PyTorch environment, torch.func.functional_call, autograd, NumPy, scikit-learn, existing vendor encoder architecture; independent new directory experiments/p1_20260905/cross_user_meta.

## Locked interfaces and provisional resource settings

Subject split: test h, validation h%8+1, remaining six train, same quality mask as previous work. No old trained weights or old fitted statistics may initialize a new fold. Reuse only deterministic per-sample prepared inputs and fixed pooled raw Right features; estimate the global label-smoothing matrix with new six-user OOF proxies.

Warmup: original Right architecture, Adam lr=1e-4, wd=1e-4, batch128, 40 epochs maximum, minimum12/patience8, whole-validation-user CE selection. Shared selected weights initialize all four continuation arms, including A0. Source losses follow original CE + .5 * label_loss(T=2). Warmup and continuation include clear training provenance, best-state and resume state.

Continuation: same encoder/head starting weights, same auxiliary initialization, Adam encoder lr=1e-5, Right head lr=1e-5, auxiliary lr=1e-4, wd=1e-4. Maximum16 epochs, minimum6/patience5, epoch0 warmup state retained. Episode per-user count locked at4 after source-only GPU benchmarks: per-user8 spills beyond physical8GB and per-user16 OOMs; per-user4 full-pipeline A3 peak5.06GiB. Each episode uses five support users and one query user, same IDs/order across arms. Subject roles rotate uniformly. Number of episodes per epoch ceil(n_train/(6*per_user)), sampled classes balanced over available classes.

S = criterion(R_support) + .5 * criterion(MV_support) for A2/A3; S = criterion(R_support) for A0/A1. Q always criterion(R_query). A0/A2 optimize S+Q(theta). A1/A3 optimize S+Q(theta-alpha*grad(S)), alpha=1e-4, exact higher-order derivative. Meta-test means training query users only, not true validation/test. A2/A3 use real support MV; A0/A1 receive Right only, internally repeat to three encoder slots to match shapes. Query has one Right view. No auxiliary dropout; paired arms share dropout seed policy.

The full mathematical objective is the first implementation. Resource or kernel incompatibility must first be resolved with math attention and smaller episodes, without silently detaching the inner update. A first-order fallback requires a recorded amendment and explicit limits before any formal results.

## Task 1 — Split, source identity and input access (root)

- [x] Write failing tests for subject-disjoint split, exact quality retention, all classes in six-user train, refusal of nontraining multiview loading, and episode support/query identity and deterministic replay.
- [x] Implement protocol.py with subject_split(y,subjects,held), SubjectData, EpisodeSampler, digest/dump helpers. Store exact source hashes and split manifests.
- [x] Verify same sampler state yields same next episode and IDs never leave the six-user partition. All inference access checks happen before array reads.
- [x] Implement new six-user OOF global smoothing preparation using only deterministic raw Right proxies; verify scaler/logistic fits exclude outer val/test users and each proxy held training user.

## Task 2 — Model and exact meta objective (implementer)

- [x] Write toy double-precision tests comparing meta-gradients to finite differences and proving query loss affects auxiliary parameters through the shared encoder update.
- [x] Implement model.py with MetaRadar, build_model(seed,device), right_only(x), support_outputs(x,mode), and strict Right-only forward. Original Right state keys remain under right.*.
- [x] Implement meta_step.py with episode_loss(model,support_x,support_y,query_x,query_y,arm,criterion,alpha=1e-4,mv_weight=.5). Return total loss and detached diagnostic losses; no real optimizer mutation inside this function.
- [x] Verify alpha0 meta/ordinary equivalence under deterministic evaluation, actual fast-weight evaluation, no mutation of original parameters, and Left/Top influence only in A2/A3 support.
- [x] Verify real model backward with exact second order on CPU/small GPU after root authorizes resource use; report actual encoder and auxiliary gradient paths.

## Task 3 — Training queue and checkpoints (root)

- [x] Test that resume mismatch is rejected before any state restoration and that checkpoint selection uses first minimum whole-user validation CE.
- [x] Implement trainer.py with preparation, common warmup, A0/A1/A2/A3 continuation and exact configuration matching. Completed outputs verify hashes before skipping. Save optimizer, RNG, sampler, best-state and source snapshots atomically each epoch.
- [x] Test full short warmup plus four-arm run in smoke output, including interruptions and Right-only evaluation. Measure GPU memory/time without tuning on test accuracy.
- [x] Freeze resource choice and source snapshots in separate formal runs root; run 8 warmups and 32 matched continuations sequentially.

## Task 4 — Independent audit and deploy replay (reviewer/implementer)

- [x] Independently recompute saved argmax, accuracy, macro-F1, all IDs/labels, best CE, valid early-stop, identical common initialization and episode manifests.
- [x] Validate raw source, new OOF matrix and warmup provenance exclude true validation/test users from all fitted parameters.
- [x] Compare A3-A1, A3-A2, A3-A0 and interaction; equal-user weighting and separate seeds; report low-performing users and budget costs.
- [x] Load exported Right-only graph independently, replay all formal predictions at matching input/compute precision. Refuse incomplete 32-arm acceptance.
- [x] Independent specification review precedes quality review; resolve blocking findings before formal execution or final claims.

## Completion and stopping

- [x] Produce machine-readable audit and Chinese results report, distinguishing exact implemented objective from literature guarantees and 74.773% historical reference from the new baseline.
- [x] Primary investment gate: A3-A1>=2pp, at least6/8 improve, A3>A0 and A3>A2 in mean. Positive requires second initialization and same-user/same-class global pairing derangement control. Negative ends this fixed configuration; do not tune outer-test-selected variants.
- [x] Leave previous experiments and unrelated work untouched. Current branch is already codex/missing-view-fusion-20260907; no reset, broad staging or unrelated commits.

## Verified outcome — 2026-09-08

All 8 warmups and 32 continuations completed. The full audit reports verified, no errors or missing runs; all 32 Right-only validation/test replays have zero maximum absolute logit error. A3−A1 = −0.0729167 pp, 0/8 improved users, gate failed. Stop this fixed configuration; second-seed and pairing-derangement expansion was conditional on passing and was not launched. See experiments/p1_20260905/cross_user_meta/runs_s2026/RESULTS.md and cross_user_meta/COMPLETION.md.
