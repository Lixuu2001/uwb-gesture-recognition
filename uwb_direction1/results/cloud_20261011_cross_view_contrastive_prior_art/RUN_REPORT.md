# Cross-view contrastive prior-art screen

**Date:** 2026-10-11  
**Decision:** generic paired-event/view-consistency objective is not cleared as a method contribution; no training or parameter search.

## Candidate screened

The proposed candidate was to use the corpus's participant/gesture/repetition pairing to pull same-class, same-event samples from different physical X4 views together, then evaluate transfer to a held-out physical view. Pairing only supports a sample-level event correspondence under soft synchronization; it is not frame alignment or a physical-motion trajectory measurement.

## New primary-source evidence

1. **Shah et al., WACV 2023, “Multi-View Action Recognition Using Contrastive Learning.”** The official WACV proceedings page describes ViewCon, a hardness-aware contrastive loss for multi-view action recognition, and reports viewpoint-invariant representations plus improved domain generalization over supervised learning with synthetic multi-view data. This directly establishes that generic cross-view contrastive alignment for viewpoint robustness/domain generalization is prior art in action recognition. [Official proceedings page](https://openaccess.thecvf.com/content/WACV2023/html/Shah_Multi-View_Action_Recognition_Using_Contrastive_Learning_WACV_2023_paper.html); [official PDF](https://openaccess.thecvf.com/content/WACV2023/papers/Shah_Multi-View_Action_Recognition_Using_Contrastive_Learning_WACV_2023_paper.pdf). The official host blocked direct opening in this executor, so the abstract/indexed proceedings record was used; no claim is made about unverified implementation details beyond that record.
2. **Siddiqui et al., AAAI 2024, “DVANet: Disentangling View and Action Features for Multi-View Action Recognition.”** The official AAAI paper states that its method separates action and view representations using separate transformer queries and two supervised contrastive losses, and evaluates on four multi-view action datasets. [Official AAAI paper page](https://ojs.aaai.org/index.php/AAAI/article/view/28290), [DOI](https://doi.org/10.1609/aaai.v38i5.28290). This is a stronger method-level precedent for class/view disentanglement and supervised contrastive learning.
3. The already-reviewed same-corpus **Xue et al. 2026 MV-GLFNet** covers left/top/right X4 single-, dual-, and tri-view fusion, but not the two-views-train/third-view-held-out protocol. That evaluation gap does not make the generic objective novel. The 2025 TIM radar multiview-DG paper remains full-text inaccessible; its detailed method is not inferred from abstract metadata.

## Decision and scope

Applying generic cross-view supervised contrastive/view-invariant learning to X4 would be an application of established action-recognition methods, not a defensible contribution by itself. The leave-one-physical-view protocol may still be useful as a diagnostic benchmark, but it is not sufficient to satisfy the project's method-contribution goal. A distinct signal-specific objective would require a concrete hypothesis and direct prior-art review before any fit.

No labels, array values, or checkpoint contents were read. No data was downloaded, no model was fit, and no training process was started. Current executor check: the restore receipt exists at `/workspace/uwb-restored-20260930`; PyTorch is CPU-only (2.14.0+cpu), `nvidia-smi` is unavailable, no active UWB training process was found (only defunct Python entries), and no radar/UWB device node was found. This is a literature gate result, not a model result.

## Next action

Continue only with a bounded, primary-source search for a specific signal-level contribution beyond generic cross-view contrastive alignment and exact-corpus fusion. Do not launch the held-view grid unless such a method clears novelty and the pre-registered source-only/cost gates.
