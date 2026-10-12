# Empty-subspace classifier novelty gate

Date: 2026-10-12 UTC  
Branch head at review start: `43a8ebd4e6a927ad285534fbe7d3a3c301995884`

## Candidate under review

The 2026-10-07 source-only diagnostic learned an uncentered, per-view subspace from empty-class clips and tested whether projecting onto its orthogonal complement preferentially attenuates empty clips while retaining active-gesture energy. Its eligibility gate passed for Left and Right, not Top. This was an energy-selectivity diagnostic only: it did not train or evaluate a gesture classifier. The registered report explicitly says a pass only permits a separately preregistered classification study, not an efficacy or novelty claim.

The proposed next step would have been to compare an empty-derived projection with the existing clutter-removed input and matched-rank random projections under held-user folds. This review was conducted before implementing or running that study.

## Primary-source findings

1. The patent family for *Method and apparatus for gesture recognition* (US20200242342A1; priority date 2017-02-16) explicitly covers radar gesture recognition, including IR-UWB. Its disclosed pipeline subtracts a signal measured when no gesture is present, obtains covariance eigenvectors from training signals for the recognizable gestures, projects gesture signals by inner products, and classifies the resulting coefficients. This is not identical to learning the nuisance subspace exclusively from empty-class clips and removing it, but it establishes prior art at the intersection of empty-reference background subtraction, eigen/subspace processing, and radar gesture recognition. A contribution cannot be claimed merely by combining those generic pieces. Source: https://patents.google.com/patent/US20200242342A1/en

2. An IR-UWB moving-target paper describes estimating a low-rank clutter subspace and separating it from target returns, including online PCA and robust PCA variants. Its task is target detection/localization rather than hand-gesture classification, so it is adjacent signal-processing prior art, not an exact same-task reproduction. It further means that generic subspace-based clutter suppression is established radar processing. Source: https://doi.org/10.3390/s16091409

3. The X4 source corpus is itself a 12-gesture, eight-participant IR-UWB release and includes an empty class. The later same-corpus three-view paper applies MTI clutter suppression before classification. These sources do not establish the proposed empty-only learned projection, but they make a broad “empty gesture plus multi-radar recognition” framing insufficiently distinct. Sources: https://www.nature.com/articles/s41597-021-00876-0 and https://doi.org/10.3389/fnins.2026.1864448

## Decision

- Novelty gate: **not cleared** for the generic empty-class subspace-projection candidate.
- Classification experiment: **not run**. The positive energy screen does not establish classification benefit or novelty; doing the eight-fold fit now would test an uncleared method.
- No labels, tensor values, or checkpoints were read in this review. No model code, project data, or source snapshot was changed. No training process was started.
- This is a bounded literature screen, not a legal opinion and not an exhaustive review. The patent's legal status and claim scope are not assessed here.

A narrower contribution would need a technically material difference beyond replacing the patent's gesture-derived eigenbasis with an empty-only basis, plus a literature-backed explanation for why that difference addresses a measured IR-UWB failure mode. Current cached inputs lack raw complex echoes, per-view background captures, calibrated transfer data, and independent motion truth, which limits physically grounded alternatives. No parameter/rank search or classifier grid is authorized by this gate.

## Current execution conditions

The latest cloud preflight recorded the restored cache at `/workspace/uwb-restored-20260930`, no active training process, CPU-only PyTorch, 746,266,624 bytes free, and restricted egress. This review neither redownloaded the migration nor started compute. The research goal remains unvalidated.

## Reproduction and source pointers

- Eligibility diagnostic: `uwb_direction1/results/cloud_20261007_empty_subspace/RUN_REPORT.md`
- Frozen diagnostic protocol: `uwb_direction1/research/EMPTY_SUBSPACE_PROTOCOL.json`
- Current status and continuation memory: `uwb_direction1/STATUS.json`, `uwb_direction1/memory/CLOUD_CONTINUATION.md`, `uwb_direction1/memory/PROJECT_MEMORY.md`
