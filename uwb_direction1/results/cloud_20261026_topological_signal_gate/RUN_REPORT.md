# Topological signal representation candidate gate

**Date:** 2026-10-26 02:32 UTC  
**Decision:** generic persistent-homology / Mapper features on the existing range-time maps are not cleared for fitting or training. No experiment ran.

## Candidate and falsifiable mechanism

Screened the idea of turning each clutter-removed range-time map into a threshold-filtration/topological summary (for example, persistence diagrams or Mapper summaries), then classifying the summaries. The proposed mechanism would be that persistent connected components/loops encode gesture-specific structure while suppressing small amplitude perturbations and clutter.

That is a plausible feature-engineering probe, but the screen did not identify a signal-specific claim beyond applying known topology summaries to a radar gesture map. It therefore does not clear a research contribution. No implementation or data computation is authorized from this review.

## Literature reviewed and scope

- Gao, *Through-the-Wall Radar Human Activity Recognition WITHOUT Using Neural Networks* (arXiv:2506.05169, 2025): the full arXiv text was reviewed. It starts from range-time and Doppler-time maps, segments micro-Doppler signatures, discretizes contours as point clouds, and matches topological similarity with Mapper for radar activity recognition. This is a different through-wall radar/task and not the same IR-UWB hand-gesture cohort, but it directly preempts generic “topological point-cloud matching for radar activity maps.”
- Żelawski & Hachaj, *The application of topological data analysis to human motion recognition*, Technical Transactions 118(1) (2021), DOI [10.37705/techtrans/e2021011](https://doi.org/10.37705/techtrans/e2021011): the indexed abstract was reviewed. It applies persistent topological features to motion-capture human-action recognition and reports perturbation/noise robustness. This is not radar evidence, but it preempts generic topology-based action recognition as the claimed novelty.
- Blumenstein et al., *Persistent Homology Approach for Human Presence Detection from 60 GHz OTFS Transmissions*, Sensors 23(4), 2224 (2023), DOI [10.3390/s23042224](https://doi.org/10.3390/s23042224): the publisher article page/abstract was reviewed. It applies persistent homology to radar-derived delay-Doppler structure for presence detection, not gesture classification.
- Ahmed et al., *UWB-Gestures: A public dataset of dynamic hand gestures acquired by impulse radio ultra-wideband radar*, Scientific Data 8, 102 (2021), DOI [10.1038/s41597-021-00876-0](https://doi.org/10.1038/s41597-021-00876-0): existing project review confirms the 12-class, three-view corpus and range-time representations. Xue et al. (2026), DOI [10.3389/fnins.2026.1864448](https://doi.org/10.3389/fnins.2026.1864448), already preempts generic same-corpus tri-view fusion. Neither paper was found to establish topological summaries as the exact same method on this cache.

This was a bounded screen, not an exhaustive search or a complete review of all related work. The exact same-corpus topological classifier was not identified, but that absence does not establish novelty. The generic candidate is closed unless a new, specific radar-signal mechanism—not merely a TDA feature family—is identified.

## Observable inputs and evaluation limits

The present restore exposes clutter-removed real-valued range-time maps and class/participant metadata, not phase-coherent raw echoes, per-view background captures, calibrated geometry, synchronized hand-pose labels, or independent motion truth. A topological summary computed from these maps could be tested descriptively, but it would not by itself identify physical trajectory or prove that topology is the causal invariant. All eight users have prior project-development exposure, so held-user folds on this corpus are not untouched independent validation.

No labels or array values were read. No fitting, feature extraction, download, or training was performed.

## Current execution check

- Managed executor: observed revision 1391, running; 5 CPU cores; network restricted/enforced; no configured capabilities, secrets, or outbound identities.
- Restored data directory exists at `/workspace/uwb-restored-20261020`; the original source subtree exists at `/workspace/uwb-cloud-current/uwb_direction1/original/project` with 789 files.
- PyTorch is absent. Process inspection found no active UWB/training process; a workspace scan through depth 6 found no `.pt`, `.pth`, or `.ckpt` checkpoint files.
- This check did not read data arrays, labels, or checkpoint contents.

## Disposition

Close generic persistence/Mapper classification of the current range-time maps as a standalone method candidate. Do not implement or train it, and do not reopen the already closed motion-primitive factorization, L/T subgroup, calibration, generic fusion, speed/trajectory, or time-reversal candidates.

The goal of a validated research contribution remains unresolved. A future candidate needs a concrete signal mechanism that is both distinguishable from the cited radar topology / action-recognition prior art and observable in the existing cache; otherwise, the next useful step is to define the study as a reproducibility/benchmark audit rather than to relabel generic feature engineering as novelty.
