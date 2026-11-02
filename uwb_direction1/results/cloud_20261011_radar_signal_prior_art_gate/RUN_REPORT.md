# IR-UWB signal-level candidate gate

Date: 2026-10-11  
Branch: `codex/uwb-direction1-cloud-20260930`  
Parent observed before this update: `ba41925a3f50ddd8aabc1d8634eefeb891aba541`

## Question

Can the existing clutter-removed X4 range-time cache support a radar-specific mechanism that is distinct from the already-screened generic image, motion-feature, multi-view, augmentation, and sample-wise view-selection approaches?

## Bounded primary-source screen

A targeted search and source review covered the canonical UWB-Gestures dataset paper, two IR-UWB hand-gesture methods, an IR-UWB spiking/event-encoding paper, and the same-corpus multi-view paper:

- The original UWB-Gestures dataset defines fast time as range and slow time as frames, and documents XeThru X4 data and a fixed 1.2 m range window (Scientific Data 2021: [dataset paper](https://www.nature.com/articles/s41597-021-00876-0)). This supports treating the cached input as a range-time observation, but does not turn its magnitude map into phase, calibrated range truth, or an independent motion measurement.
- An IR-UWB Inception-classifier paper already maps clutter-removed radar returns to image-like patterns and classifies them with a CNN ([Sensors 2020](https://www.mdpi.com/1424-8220/20/2/564)). Generic “treat the RT map as an image and learn local/global patterns” is therefore not a cleared contribution.
- An earlier IR-UWB hand-gesture paper uses hand-region/area cues and time-of-arrival variance ([Sensors 2017](https://www.mdpi.com/1424-8220/17/4/833)); generic range-motion/hand-area descriptors are also not cleared.
- An IR-UWB gesture paper reports event coding and level-crossing sampling before an SNN ([IEEE AICAS 2022](https://ieeexplore.ieee.org/document/9870013/)). The reviewed record is the publisher abstract/metadata, not a full-text audit; it is sufficient to reject a broad claim that event-based radar gesture encoding itself is new, but not to assert equivalence of implementation or protocol.
- Same-corpus X4 multi-view local/global and gated/bilinear fusion is already described in Xue et al. 2026 ([full text](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1864448/full)). The project’s newer readout and view-selection reports also close generic head swaps, fusion, and confidence/cost gates.

This is a bounded screen, not an exhaustive novelty search. The cited works differ in corpus, sensor configuration, or evaluation protocol; their scores are not compared with the project’s eight-fold results.

## Feasibility finding

The only current model input is the restored clutter-removed cache at `/workspace/uwb-restored-20260930/cache/X_clutterremoved.npy` (1,958,123,288 bytes). The inventory and prior restore notes state that this subset has no raw complex echoes, per-view background captures, antenna/transfer calibration, synchronized frame clock, or independent hand-motion/pose ground truth. No radar acquisition device is attached in this executor.

Consequently, a claim about phase/group delay, calibrated 3-D motion, physical scattering compensation, true sensor acquisition cost, or measured view-to-view synchronization would be unsupported here. No new signal-level mechanism cleared both the prior-art and input-feasibility gates in this screen. **No experiment or training was authorized or run.** No labels, tensor values, or checkpoint contents were read; no data was downloaded or changed.

## Decision and concrete next step

Keep the novelty gate unresolved and do not launch another architecture/feature grid on this cache. The next genuinely discriminating experiment requires either (a) a new independent annotated cohort with compatible IR-UWB range-time data, or (b) a calibrated acquisition that records, per view:

1. raw complex/pulse-level echoes and a per-view empty-scene/background capture;
2. sensor serial/configuration, pose, range and angle, plus a calibration reflector/known-range target;
3. a shared trigger/frame timestamps for simultaneous views (or explicit repeated-trial timing if a single radar is moved);
4. gesture onset/offset and motion/pose reference synchronized to the radar;
5. per-view power, wake-up/acquisition latency and failure/missing-view conditions if the one-versus-three-radar budget question is retained;
6. new participants and sessions reserved as untouched external tests.

A radar-specific method can then be formulated against the actual observables and tested against locked single-view, conventional multi-view fusion, and signal-processing controls. The above is an acquisition requirement and experimental design, **not** a claim that a new method has been invented or validated.

## Runtime snapshot

- Cloud executor status: running; five CPUs; PyTorch 2.14.0+cpu; CUDA unavailable; no acquisition device.
- Restored data path exists; no active training process (only defunct historical Python PIDs 598, 607, 4719).
- Local source worktree is partial (31 files); the remote 789-file original source snapshot is left untouched.
- Current branch head checked before writing: `ba41925a3f50ddd8aabc1d8634eefeb891aba541`. No raw arrays or model weights are committed.
