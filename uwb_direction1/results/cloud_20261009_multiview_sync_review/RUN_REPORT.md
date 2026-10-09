# Multi-view synchronization evidence review — 2026-10-09

## Scope and decision

Read the full UWB-Gestures data descriptor and the full-text 2026 MV-GLFNet paper sections describing acquisition, view inputs, and deployment constraints. This was a literature/protocol audit only: no local labels or radar arrays were read, no fitting occurred, and no training was launched.

## What the primary sources establish

The UWB-Gestures descriptor says three X4 radars operate independently in monostatic mode, that each view is stored separately, and that each gesture sample contains 90 slow-time rows (4.5 s). It does not describe a shared trigger, timestamps, or a cross-radar synchronization/calibration procedure. This is an absence in the published protocol description; it is not proof that the physical recordings were asynchronous.

The 2026 MV-GLFNet paper uses the same public corpus. Its method organizes the three equally preprocessed TR maps as multi-channel inputs because they share aligned TR dimensions. Its deployment discussion explicitly lists view alignment and possible synchronization/calibration as costs of three-radar setups. Thus matching tensor dimensions or sample indices must not be treated as evidence of time/phase synchronization, and current public metadata do not support coherent cross-radar phase fusion, physical-delay estimation, or calibrated 3-D trajectory claims.

This does not invalidate ordinary noncoherent multi-view classification: same-corpus single/dual/tri-view learning has already been published. Nor does it validate a new alignment method. Radar-camera temporal alignment and DTW-based synchronization already have adjacent prior art, and exact multi-radar gesture alignment novelty is not cleared.

## Current execution status

The restored UWB receipt is present at `/workspace/uwb-restored-20260930/RESTORE_VERIFIED.json`: 86 files, verified state, expected transfer-manifest SHA. Runtime is Python 3.12.14 / PyTorch 2.14.0+cpu, CUDA unavailable. Process inspection found only defunct Python entries, no active training/checkpoint writer. No labels or arrays were opened. No data restoration was needed.

## Gate

Do not fit another generic fusion/alignment variant. Before any physical cross-view timing/trajectory claim, require acquisition metadata or a defensible per-sample alignment truth and a frozen participant-disjoint evaluation. A future candidate would also need a distinct, source-reviewed mechanism beyond ordinary multi-view fusion and generic DTW alignment. No candidate currently clears novelty and execution gates; the requested research contribution remains unvalidated.

## Sources

- Ahmed et al., *UWB-gestures, a public dataset of dynamic hand gestures acquired using impulse radar sensors*, Scientific Data (2021), full text: https://www.nature.com/articles/s41597-021-00876-0
- Xue, Yang & Zhu, *Neuromorphic-inspired multi-view global-local fusion for IR-UWB radar dynamic gesture recognition*, Frontiers in Neuroscience (2026), full text: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1864448/full
- Liu et al., *A DTW-Based Spatio-Temporal Synchronization Method for Radar and Camera Fusion*, Sensors (2026): https://doi.org/10.3390/s26072108
- Targeted search for multi-radar synchronization/gesture alignment is bounded and does not establish exhaustive novelty clearance.
