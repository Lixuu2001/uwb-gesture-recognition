# View-pairing provenance and 2025 TIM full-text access gate

**Date:** 2026-10-10T15:13:08Z  
**Branch parent:** 6838f9d3ff685901e1a9dab8ea59cd32ac6cd801  
**Decision:** the cache builder documents index-paired gesture-event correspondence, but this does not establish external timestamps, physical clock synchronization, phase calibration, or a method contribution. The 2025 TIM full paper remains inaccessible in this executor/browser session. Do not launch held-view training from this evaluation protocol alone.

## Checks performed

- Read-only review of the preserved original project source `uwb_direction1/original/project/uwb_osr/data.py` (SHA-256 `a8ab00ee6a7ffe276c92efc172f94fdcc5cda224459f2f74454b684bd8347628`). Its documented cache builder groups files by participant and gesture, fixes view order as Left/Top/Right, reshapes each per-view file into repetitions × 90 × 189, and stacks the same repetition index across the three files (truncating to the shortest view). Its source comment explicitly describes repetition index *i* as the same physical gesture event across views and calls the acquisition soft-synchronised.
- This is useful provenance for defining an event-level, index-paired classification sample across views. It is not an independent verification against raw per-repetition IDs/timestamps: those were not available in the restored cache inventory. The dataset documentation and this code do not establish a shared trigger, calibrated timing offset, coherent phase alignment, or 3-D ground truth. Do not claim those.
- Attempted the official IEEE Xplore article page for DOI [10.1109/TIM.2025.3637962](https://doi.org/10.1109/TIM.2025.3637962), article 11270504. The publisher page returned a JavaScript/robot verification wall; the DOI resolver was unavailable in the browser tool. Targeted public searches found metadata/abstract and the authors' public [MVRADAR implementation](https://github.com/Tkwer/MVRADAR), but no accessible accepted manuscript/full text.
- The reviewed author implementation documents multiple signal representations (RT/DT/RDT/ERT/ART) from one TI MIMO mmWave radar, feature fusion, Dempster–Shafer evidence combination, and an optional domain-adversarial branch. This is relevant prior art for generic multi-representation/evidence fusion/domain-generalization combinations. It does not itself verify a train-on-two-physical-X4-views/test-on-third-view protocol. Because the paper's full text remains unreviewed, do not infer the paper's precise view-OOD evaluation scope from its abstract or code.

## Execution and data boundaries

- Current executor check: CPU-only PyTorch 2.14.0; CUDA unavailable. No active research/training process was found (only defunct historical entries).
- The verified restored cache remains at `/workspace/uwb-restored-20260930`; no download or restore was repeated.
- No project labels or radar-array values were read; no model was fitted, trained, tuned, or selected; no process or source-snapshot file was modified.

## Gate result and next action

The view pairing is documented well enough to define a grouped classification split, but a held-subject + held-view split is an evaluation protocol, not by itself an original method. Novelty remains unresolved and no candidate is cleared. Do not start the 24-cell outer evaluation or a model grid. Continue only through a lawful author/institutional full-text route for the TIM paper and method-level prior-art review; then require a distinct mechanism and frozen source-only selection rules before any fit. Otherwise the current cache has no cleared route, and new calibrated radar measurements with synchronized external motion truth would be needed for physical-invariance claims.
