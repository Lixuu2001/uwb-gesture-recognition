# Paired-event cross-view consistency candidate screen

**Date:** 2026-10-11  
**Branch:** `codex/uwb-direction1-cloud-20260930`  
**Decision:** candidate remains unapproved; no training, parameter sweep, or test-fold inspection.

## Question screened

Could the event correspondence in the three-view UWB-Gestures corpus support a paired-event consistency method for transfer to an unseen physical radar view, distinct from ordinary multiview fusion?

## Evidence reviewed

1. **Exact-corpus prior art.** Xue, Yang, and Zhu (2026), [MV-GLFNet](https://doi.org/10.3389/fnins.2026.1864448), is a full-text, same-corpus IR-UWB paper. It uses the left/top/right views, early-fuses motion-enhanced time-range maps, then applies local/global branches and gated plus bilinear fusion. It reports single-view, every dual-view pair, and tri-view input settings under 6-subject train / 2-subject test. It does not report training on two physical views and evaluating on a held-out third view. The authors also discuss left/right asymmetry as potentially due to gesture execution, view-dependent scattering, or self-occlusion. Thus neither view permutation nor reflection/label equivariance can be assumed.
2. **Related multiview domain generalization.** Chen et al. (2025), “Domain-Generalized Gesture Recognition via mmWave Radar Signal Multiview Learning,” DOI [10.1109/TIM.2025.3637962](https://doi.org/10.1109/TIM.2025.3637962), is confirmed by bibliographic metadata and abstract as a multiview, OOD gesture-recognition method using domain knowledge and Dempster–Shafer evidence fusion on mmWave data. Its full paper remains inaccessible through lawful public routes in this executor, so its detailed protocol and any exact paired-view loss are **not** asserted.
3. **Adjacent radar gesture prior art.** Rodar (2024), DOI [10.1109/TMC.2024.3402356](https://doi.org/10.1109/TMC.2024.3402356), uses multiview radar fusion to address human-activity interference. This reinforces that generic radar-view fusion/robustness is established, though its sensor and task conditions differ.
4. **Project pairing scope.** Existing source audit documents the pairing key as participant, gesture, and repetition index and describes same-index records as the same gesture event under soft synchronization. It does not establish a shared trigger, per-frame timestamp correspondence, coherent phase alignment, or external motion truth. An event-pair objective, if later evaluated, must remain sample-level and must not be described as temporal/trajectory alignment.

## Assessment

A simple cross-view contrastive or consistency term would combine established multiview learning with the already-published UWB fusion setting. The exact two-source-view/held-third-view protocol appears unreported in the reviewed same-corpus article, but a new evaluation split alone is not a method contribution. The possible method-level distinction—using same-event repetition correspondences to train a view-consistent representation, then evaluating on an unseen physical view—has not yet cleared prior art or established that it adds anything beyond ordinary supervised class learning and fusion.

Therefore:
- do not call paired-view contrastive learning novel yet;
- do not start the held-view grid or fit test labels;
- do not make timing, calibration, 3-D, or mirror-equivalence claims from the pairing key.

## Next gate

Continue a bounded review for direct event-paired view-consistency/contrastive methods in radar gesture and the full text of the 2025 TIM multiview-DG paper. If no exact preemption is found, specify a single method-level hypothesis and source-only, participant-disjoint train/validation/diagnostic protocol; run a small CPU cost pilot only after that preregistration. Any later experiment must compare against fixed single-view and standard fusion baselines and score held-out users/views only after all model choices are frozen.

## Execution audit

The restored data receipt is present at `/workspace/uwb-restored-20260930`. This check found no active UWB Python training process; only defunct Python entries were visible. PyTorch is CPU-only, CUDA is unavailable, and no radar acquisition devices are exposed. No labels, tensor values, or checkpoints were read. No fit/training or data download occurred. The local source checkout remains partial; preserve the remote 789-file source snapshot and do not reconstruct it from the partial checkout.

**Conclusion:** no validated contribution yet; candidate screening continues.
