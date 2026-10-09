# Multi-view and sensor-dropout prior-art update — 2026-10-09

## Scope
Read-only review of primary papers to decide whether a radar-view dropout / resilient fusion experiment is sufficiently distinct to run on the existing UWB-Gestures data. No UWB labels or arrays were opened, no models were fit, and no training was started. The verified restore and original-source snapshot were not changed.

## Findings
1. **Same dataset, same sensors, same view subsets are directly covered.** Xue et al., “Neuromorphic-inspired multi-view global-local fusion for IR-UWB radar dynamic gesture recognition,” *Frontiers in Neuroscience* (19 June 2026), DOI 10.3389/fnins.2026.1864448, uses the same 12-class, eight-subject UWB-Gestures corpus and three X4 views. The full article reports single-view, all three dual-view combinations, and tri-view results; its 6-subject-train/2-subject-test protocol yields 98.29% for three views. Its authors also analyze view-dependent performance and sensor cost. The paper's reported split is not the same as this project's eight repeated one-user-held folds, so scores are not directly comparable; nevertheless, generic single/dual/tri-view fusion, view-subset comparison, and “complementary views help” are preempted.
2. **Multi-radar failure/incomplete-data handling is also prior art.** Zhao et al., “Gesture Recognition Using Multiple mmWave FMCW Radars,” IEEE VTC Fall 2023, DOI 10.1109/VTC2023-Fall60731.2023.10333652, explicitly evaluates incomplete feature maps caused by one radar failing. In the conference paper, an incomplete subset is the test set and an SVM achieves 95% accuracy. It is FMCW and not this X4 dataset, but it preempts a broad claim that multi-radar gesture recognition under one-view failure is itself new.
3. **Angular-diversity robustness predates this dataset.** Yang et al., “Sparsity aware dynamic gesture recognition using radar sensors with angular diversity,” *IET Radar, Sonar & Navigation* (2018), DOI 10.1049/iet-rsn.2018.5206, evaluates two simultaneous CW radars at different aspects, feature fusion, poor performance from one view, and the effect of orthogonal lines of sight.
4. **Randomly dropping modalities is not a novel training mechanism.** Neverova et al., “ModDrop: Adaptive Multi-Modal Gesture Recognition,” *IEEE TPAMI* (2016), DOI 10.1109/TPAMI.2015.2461544, introduces random channel dropping to train gesture models to handle missing modalities. A radar-specific implementation would need a substantive new mechanism and evidence beyond rebranding ModDrop.

## Decision
Do not run a generic sensor-dropout, subset-fusion, or angular-diversity experiment and present it as the requested original method. Such a run would likely be an engineering benchmark on familiar mechanisms, not a literature-supported contribution. No contribution is validated.

The remaining defensible paths are: (a) a materially different physical question with new synchronized ground truth/acquisition hardware; or (b) a precisely justified cross-dataset question that survives exact prior-art and licensing gates. The current executor has no acquisition device, remains CPU-only with restricted shell egress, and no independent dynamic IR-UWB gesture cohort has been verified. No experiment is running.

## Sources reviewed
- Xue et al. full article: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1864448/full
- Zhao et al. IEEE Xplore record: https://ieeexplore.ieee.org/document/10333652/
- Zhao et al. VTC 2023 paper PDF: https://www.ieeevtc.org/vtc2023fall/DATA/2023002930.pdf
- Yang et al. full text: https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-rsn.2018.5206
- Neverova et al. paper: https://arxiv.org/abs/1501.00102
- Current endpoint prior-art report: `results/cloud_20261009_endpoint_review/RUN_REPORT.md`
