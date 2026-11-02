# Mild time augmentation implementation plan

> Execute with superpowers:subagent-driven-development under the user's approval to implement mild time augmentation. Continue to measured, audited results.

**Goal:** Compare the original Right classifier with a single fixed mild time augmentation on the existing six-user training / whole-user validation / held-user test protocol.

**Architecture:** Keep the original model, loss, optimizer, sampler and checkpoint selection. Apply an endpoint-preserving monotone time warp to signed clutter-removed Right radar frames, then recompute amplitude and phase-motion inputs with the existing per-sample transform. No multiview training or new branch.

**Tech Stack:** Existing PyTorch CUDA environment and cross_user_meta training/recovery implementation, independent new mild_time_augmentation directory. Preserve current research checkout and all unrelated changes; no moves, broad staging or commits.

## Fixed configuration and controls

- Time map g(t)=t+a*sin(2*pi*t/89), a selected uniformly from {-2,+2} with probability 0.5; otherwise original input. Endpoints remain 0/89, derivative stays positive. No cropping, reversal or amplitude changes added. This is a signal-space augmentation, not validated physical hand motion.
- Warp signed clutter-removed CIR with float32 linear interpolation; recompute existing abs-normalized amplitude and amplitude-weighted wrapped phase increment, concatenate as before and cache float16. Do not directly interpolate an already derived phase map.
- Model seed 2026, training shuffle seed 2026+fold, dropout seed 102026; Adam lr1e-4 wd1e-4; batch128/micro32, maximum40/minimum12/patience8, same label loss and six-user OOF smoothing. First minimum clean validation CE selects checkpoint; epoch0 random candidate retained.
- Augmentation uses a separate stateless NumPy schedule derived from source sampler RNG state and fold. It must not consume the model/dropout RNG, must replay on resume, and must not access val/test via training augmentation.
- Reuse old warmup as CONTROL only after exact old code/data/runtime/config/split/initialization/matrix/episode prefix checks. Validate the no-augmentation new wrapper on a three-step smoke run against the old smoke. If any mismatch, resolve or rerun the baseline, never use unmatched controls.
- Train eight AUG folds. Primary paired effect is clean outer-test AUG−CONTROL. Report macro-F1 and each user. Secondary pressure-test predictions for ±2 raw-signal warps on both models are evaluated only after model selection; they never select epochs or change configuration.
- Stop this fixed setting if it fails the established development gate (at least +2pp, at least6/8 improve). Positive signal would require a second seed before a stability claim. Improving perturbed inputs alone is robustness evidence, not clean-accuracy improvement.

## Tasks

Pre-training amendment: actual NumPy is 2.0.2 whereas the old ENVIRONMENT.json records 2.0.1. Torch/CUDA, old code/data hashes and full raw-to-prepared identity match. Per the fixed fallback, rerun eight CONTROL and eight AUG in the same current environment. Old runs serve provenance only; legal six-user OOF matrices are reused identically by both new arms. This amendment precedes all new formal outcomes.

- [x] Implement and test raw warp, identity transform agreement, per-sample separation, deterministic augmentation schedule and val/test guard. Files: experiments/p1_20260905/mild_time_augmentation/augment.py and test_augment.py.
- [x] Precompute two deterministic Right input caches with atomic finalization and file hashes; record original raw/cache identities, transform code and identity-control numerical error. No fitted parameters in caching.
- [x] Implement runner.py as a thin wrapper of unchanged cross_user_meta trainer; record augmentation plan hashes/counts, matched control ledger and full source snapshots. Reuse existing checkpoint recovery, CE selection and standalone Right export.
- [x] Run no-augmentation reproduction smoke, augmented smoke and interruption/restart check before formal eight-fold queue. Independently review contracts before final claims.
- [x] Train all eight fresh CONTROL and eight fixed AUG runs to completion and independently recompute all clean paired metrics, source exclusions, selection, state updates and sampling traces. Completed 16/16; independent NumPy/sklearn review agrees.
- [x] Replay both standalone Right paths on clean and fixed warped held-user inputs. Produced Chinese RESULTS.md, AUDIT_REPLAY.json and COMPLETION.md, distinguishing old-control reuse, new training and protocol boundaries. Clean/validation replay: 37,200 predictions, maximum logit error 0; fixed warped inputs: 37,200 predictions. Audit verified with no errors or missing runs.

Final result: CONTROL 71.2913%, AUG 72.2134%, equal-user accuracy effect +0.9222 pp; 6/8 users improve. The +2 pp development gate is not met. Stop expansion of this fixed setting; preserve the implementation and results as a small positive development result, without a stability claim or automatic adoption as the new default.
