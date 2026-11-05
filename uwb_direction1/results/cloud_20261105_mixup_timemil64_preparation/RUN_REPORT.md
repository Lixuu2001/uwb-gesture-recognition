# Fixed input Mixup transfer: preparation

## 2026-11-05 next published baseline fixed: input Mixup on full TimeMIL64

After the completed fixed ensemble failed and closed at59f8f0608380607abae8402dcbd90f128d12da0a, a DISTINCT published regularizer transfer is fixed BEFORE any new UWB fit: input Mixup alpha1.0, seed20261029, full850973parameter TimeMIL64/B64/three-radar RMS maps, native OvRBCE/AdamW/Lookahead/clip/warmup. Paper Zhang et al. ICLR2018 arXiv1710.09412 §2,§3.2,§3.3,§3.8 and official code eaff31ab397a90fbc0a4aac71fb5311144b3608b inspected. Author released default alpha1 and convex inputs/targets retained; CIFAR/architecture/optimizer/loss/protocol transfer deviations explicit. Speech paper includes both improvement and degradation; UWB performance unknown. This is a baseline transfer, not novelty.

Bounded historical/source check distinguishes old evidential manifold-negative Uniform(.4,.6) mixing and geodesic multimodal review from full-TimeMIL input-Beta1 soft-target training; no equivalent fixed full-TimeMIL run found in checked records. Source3–6 train/2validation/1diagnostic fixed, source1 old reference.9866666666666667, one post-freeze/private-backup diagnostic gate≥+.005/tol1e-12. All selection only inner train/validation; users7–8 labels forbidden; repeated development exploratory. No alpha/epoch/seed rescue after failure and no reopening closed ensemble. Full fresh64trajectory replay and selected-output audit/private archive required before one diagnostic read.

Exact runnable implementation and fixed synthetic resource plan now registered at research/cloud_20261105_mixup_timemil64_preparation. Synthetic author-function/gradient/state/RNG/slow/journal tests and eight-update full-B64 CPU cost run are the NEXT execution, with600s synthetic/43200s actual pilot wall/CPU and12GiB RSS caps. No UWB fitting or background training is started by this preparation. Accuracy/originality goals unresolved.

