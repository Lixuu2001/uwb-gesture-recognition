# Mild style augmentation implementation plan

> Use superpowers:subagent-driven-development for bounded implementation and independent review, under the user's instruction to continue to the next experiment. Preserve the current research checkout and previous results; do not commit or move unrelated work.

**Goal:** Measure one fixed mild same-class cross-user statistics augmentation against verified matched Right controls.

**Architecture:** A temporary training-epoch projection hook plus a guarded reader. Reuse unchanged cross_user_meta training/recovery and standalone export. No time augmentation or new trainable parameters. Full design is in docs/superpowers/specs/2026-09-09-mild-style-augmentation-design.md.

**Tech Stack:** D:/D_conda/envs/pytorch/python.exe, PyTorch CUDA, NumPy, existing prepared data and six-user OOF matrices.

## Fixed boundaries

New files only under experiments/p1_20260905/mild_style_augmentation, plus these design/plan files. Existing source trainer and completed experiments remain unchanged. Existing CONTROL reference: mild_time_augmentation/runs/CONTROL/fold1..8/warmup. Matrix reference: cross_user_meta/runs_s2026/matrices. No new external state or communications.

Source model seed 2026; same old config with added style metadata: probability=.25, share=.1, augmentation seed=20260909. Independent schedule seed is first eight little-endian bytes of SHA256 of sorted JSON {'sampler': sampler_state['rng'], 'fold': fold, 'augmentation_seed': 20260909}. For ascending training IDs, draw all masks with rng.random(n)<.25, then for each source choose sorted eligible other user uniformly and that user's sorted same-class ID uniformly. Log ordered int64 triples (anchor ID, donor ID, enabled 0/1) for every microbatch. Both pathways share the schedule.

## Tasks

- [x] Implement and test style.py. Public API: style_mix(x, donor, mask, share=.1); schedule(train_ids, labels, subjects, seed, probability=.25)->(donor_ids, masks) aligned with train_ids; sampler_seed(state,fold); StyleReader(base,labels,subjects,enabled=True), configure(seed), load_ids(ids,role,mode), epoch_context(model) context manager. Expose partitions/base/trace/counts (keys clean/style). Reader load_ids first calls guarded base; donors only loaded in train mode R within epoch_context. Verified hook names: model.right.encoder.temporal.proj and model.right.encoder.spatial.proj. Use F.linear for donor projections to avoid recursive hooks. Remove hooks and donor tensors on exit, including exception paths.
- [x] Test meaningful contracts before implementation: donor same-class/different-user/train-only, RNG isolation/replay, zero-share and unmasked identity, mixed statistics, no donor gradients, invalid inputs, clean val/test bypass and hook removal. Run `python -m pytest experiments/p1_20260905/mild_style_augmentation/test_style.py -q`.
- [x] Implement runner.py wrapping unchanged trainer: derive seed, configure reader, `with reader.epoch_context(model): row=original_epoch(...)`, then add seed/trace/counts. Freeze source/config/NumPy and control artifact ledger. Preverify completed control data/code/runtime/config/splits/matrix/init and source snapshots. Use original run_arm for checkpoints, clean CE selection, saved Right predictions. Smoke mode changes only the same original three-step/one-epoch limits. Formal STYLE mode executes all eight folds sequentially, with atomic queue state.
- [x] Run disabled-style smoke and enabled-style smoke; compare disabled logits exactly with mild_time_augmentation/smoke/CONTROL. Run actual interrupted/restarted epoch recovery check. Fix any implementation defect before freezing the formal experiment; no outer-test-driven settings changes. Save PRETRAIN_CHECK.json and environment/source evidence.
- [x] Independently review implementation contracts. Implement audit.py/test_audit.py using prior audit utilities only for unchanged baseline mechanics; independently reconstruct style donor/mask schedule and trace. Verify 8 STYLE plus 8 referenced CONTROL runs, clean saved predictions/selection, parameter and optimizer updates, exclusions and identity. Incomplete/smoke/error results never get a formal gate. Use independent NumPy/sklearn cross-check for final metrics.
- [x] Complete eight STYLE fits (fresh CONTROL only if reuse contract fails), then `python experiments/p1_20260905/mild_style_augmentation/audit.py --replay`. Strict-load original Right classifier, input only Right, compare validation/test logits and argmax; no donor selection at inference. Save AUDIT_REPLAY.json, RESULTS.md and COMPLETION.md. Assess predefined clean gate only after all runs verified, report single-seed limits, stop fixed setting when below gate.
