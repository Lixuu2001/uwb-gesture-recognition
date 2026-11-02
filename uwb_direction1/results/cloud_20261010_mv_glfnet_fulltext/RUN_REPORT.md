# Same-corpus MV-GLFNet full-text prior-art review

**Date:** 2026-10-10T17:50:14Z  
**Branch parent:** a7a9e8f31f137b5b86e0140477e2799334fdef38  
**Decision:** Direct prior art now confirms that three-view IR-UWB gesture fusion on the same public UWB-Gestures corpus is already published. Generic three-view early fusion, local/global feature branches, and adaptive first-/second-order feature fusion are not novelty-cleared. The article reports a subject-held-out 6/2 split and view-subset ablations, but does not report training on two physical views and testing on a held-out third view. That evaluation gap alone is not a method contribution. No candidate is cleared; do not start held-view training from this finding.

## Primary full text reviewed

Xue, Yang, and Zhu, “Neuromorphic-inspired multi-view global-local fusion for IR-UWB radar dynamic gesture recognition,” *Frontiers in Neuroscience* (19 June 2026), [full text and DOI 10.3389/fnins.2026.1864448](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1864448/full).

The article identifies the public UWB-Gestures corpus, 12 gesture categories, and three XeThru X4 radars at left, top, and right viewpoints. It states 9,600 samples from eight subjects, with 7,200 samples from six subjects for training and 2,400 from two subjects for testing. Its reported tri-view accuracy is 98.29%. Table 6 reports single-, dual-, and tri-view input results (for MV-GLFNet: left 88.13%, right 81.29%, top 77.82%, left+right 91.06%, left+top 95.12%, right+top 96.71%, all three 98.29%).

Its method directly covers several otherwise tempting claims:
- MTI-filtered time-range maps, resized to 64×128 and stacked as three input channels (early fusion).
- A local ConvMixer branch and a global CNN/Transformer branch.
- Adaptive fusion combining gated first-order features and bilinear second-order interactions.

These are now direct same-corpus prior art, beyond the broad multirepresentation/domain-generalization prior art recorded earlier. The paper calls its protocol subject-independent; the reported 6/2 split is not the project's eight-fold leave-one-subject-out protocol, so its 98.29% must not be compared as if protocol-matched. The view-subset table changes input views under that subject split; the article does not report a held-physical-view evaluation. It also states cross-device and cross-dataset generalization as future work.

## Remaining access gate and project evidence

The separate 2025 IEEE TIM article, DOI [10.1109/TIM.2025.3637962](https://doi.org/10.1109/TIM.2025.3637962), remains a separate unresolved full-text review. Its official IEEE page was inaccessible in the current session; public abstract/metadata and author code are not a substitute for full-text method review.

The preserved project data-loader source documents event-level pairing by participant, gesture, and repetition index across Left/Top/Right under soft synchronization. That supports grouped classification samples only; it does not establish shared-trigger timestamps, phase calibration, or hand-motion ground truth.

## Execution and data boundaries

A fresh cloud check found no active research/training process; listed historical Python jobs are defunct. PyTorch is 2.14.0+cpu and CUDA is unavailable. The restored data receipt and the complete 789-file source snapshot remain present. This review read no project labels or radar-array values and performed no fit, training, tuning, or checkpoint selection. No data restore was repeated, and no original source file was changed.

## Gate and next action

Same-corpus three-view fusion and local/global adaptive feature fusion are preempted. Held-view evaluation remains a benchmark question only; no method novelty has been established. Keep the result exploratory and do not launch a 24-cell held-subject × held-view grid without a distinct, literature-cleared mechanism and frozen source-only selection rules. Continue with a lawful full-text route for the 2025 TIM paper. If no distinct mechanism can be supported by the available cached maps, the current cloud lacks the independent calibrated radar systems and synchronized motion truth needed for a physical-invariance claim.

