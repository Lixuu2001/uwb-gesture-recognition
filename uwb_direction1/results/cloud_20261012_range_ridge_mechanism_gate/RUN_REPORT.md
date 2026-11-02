# Range-ridge and hand-area mechanism novelty gate

Date: 2026-10-12 14:20 UTC  
Branch head before review: `0932baf9b95ecc638653f222ae08802b31fdfec0`  
Review type: bounded primary-source screen of a possible single-view physical mechanism. No project labels, tensor values or checkpoints were read; no data fitting or training.

## Candidate

A plausible low-cost mechanism on time-range maps is to trace the hand-induced range ridge, estimate direction/displacement, and combine it with echo energy or apparent hand-area change. This is feasible in principle from a time-range map, so it was checked before opening an experiment.

## Primary-source evidence

- Ren, Quan & Cho, “Algorithm for Gesture Recognition Using an IR-UWB Radar Sensor” (2016; DOI 10.4236/jcc.2016.43015), directly describes recognition from hand moving direction, distance change, and frontal hand surface-area change. The method removes clutter, analyzes target location and energy changes, and recognizes six gesture types. This is direct prior art for generic range-ridge direction/displacement plus echo-energy/extent as the contribution. It does not establish the same classes, subject split, or accuracy as this project. Primary source: https://doi.org/10.4236/jcc.2016.43015
- Ahmed et al., “UWB-gestures, a public dataset…” (Scientific Data 2021; DOI 10.1038/s41597-021-00876-0), defines the released corpus as 90 slow-time by 189 fast-time samples per 4.5-second gesture, with three independent monostatic X4 views. This supports the physical interpretation of the axes and deployment scope, but the paper does not add independent trajectory ground truth. Primary source: https://www.nature.com/articles/s41597-021-00876-0
- Xue, Yang & Zhu, “Neuromorphic-inspired multi-view global-local fusion for IR-UWB radar dynamic gesture recognition” (Frontiers in Neuroscience 2026; DOI 10.3389/fnins.2026.1864448), uses this public 12-class UWB-Gestures benchmark. It explicitly constructs motion-enhanced time-range maps, applies slow-time MTI, and combines three views with parallel local/global temporal branches and adaptive fusion. This preempts presenting generic TR-map temporal modeling, multi-view combination, or a global/local network as the new mechanism. The local held-user protocol's comparability to the paper is not established; its headline accuracy is not compared with the local score. Primary source: https://doi.org/10.3389/fnins.2026.1864448

## Decision

Do not proceed with a standalone “range-ridge trajectory + echo energy/area” classifier or another generic temporal/multi-view network. The core physical observables and broad architectural variants are already represented in the cited IR-UWB literature and project archive. This is a bounded novelty screen, not a claim that every possible range-time signal mechanism has been exhausted.

No classification experiment or source-only data probe was run: available project records already document range moments, temporal extent, range-frequency, scattering, SSM, local matching and related screens. Repeating them under a new name would not identify a new contribution. A candidate should proceed only after a distinct physical observable is specified, prior art is checked, and the precise required input channel is confirmed to exist.

## Data and executor gate

The current STATUS and continuation memory report that the clutter-removed cache lacks raw complex echoes, per-view backgrounds, calibrated transfer data and synchronized motion truth. Therefore coherent phase/transfer-function claims, independently validated hand kinematics, and measured radar acquisition-power savings are not supported by the present cached tensors. The cloud still has the restored dataset at `/workspace/uwb-restored-20260930`; this turn did not download or restore it.

Fresh preflight: CPU-only, 5 cores, Python 3.12.14, PyTorch 2.14.0+cpu, CUDA unavailable, 712 MiB free on the 32 GiB workspace. No active training process was found. The only Python entries were defunct historical processes. No labels, tensor values, or checkpoints were read; no training started.

## Next gate

A new experiment needs a signal-specific mechanism that is not merely range trajectory, energy/extent, generic TR temporal modeling, multi-view fusion, or an archived screen. Before a run, document (1) the physically interpretable observable, (2) a primary-source distinction, (3) channel availability and preprocessing provenance, and (4) a frozen user-disjoint protocol with destructive controls and CPU/storage estimate. If a candidate needs raw echoes, empty-scene backgrounds, calibration, or motion truth, those exact measurements are currently absent from this cloud cache. Research contribution remains unvalidated.
