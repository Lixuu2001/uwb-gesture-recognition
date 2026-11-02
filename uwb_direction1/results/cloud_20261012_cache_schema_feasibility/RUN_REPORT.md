# Restored-cache schema feasibility preflight

Date: 2026-10-12 16:31:39 UTC  
Branch head before this record: `b24dc330d3f1205c3ec04e4ba327930d927126e3`  
This was a file/schema-only preflight to confirm whether signal-level candidates can be evaluated from the current cloud cache.

## Observed files and headers

A recursive filename listing and read-only NumPy memory-mapped header inspection found these five files in `/workspace/uwb-restored-20260930/cache`:

| File | Header shape | dtype | Bytes |
|---|---:|---|---:|
| `X_clutterremoved.npy` | (9593, 3, 90, 189) | float32 | 1,958,123,288 |
| `y_clutterremoved.npy` | (9593,) | int64 | 76,872 |
| `subj_clutterremoved.npy` | (9593,) | int64 | 76,872 |
| `y_raw.npy` | (9593,) | int64 | 76,872 |
| `subj_raw.npy` | (9593,) | int64 | 76,872 |

The restored tree contains no `X_raw.npy` or complex-valued echo file. The code did not access array payloads: NumPy opened the files as memory maps and inspected only headers. It did not read label values, tensor values or checkpoint contents. No project data were copied, restored, or uploaded.

## Scientific consequence

The usable core input is a clutter-removed real-valued three-view time-range tensor. It can support experiments whose observables are identifiable from that representation. It cannot support coherent complex-phase, pulse-level event-coding, or raw-waveform reconstruction claims. The absence of raw echoes is now confirmed by the restored file inventory/header, rather than inferred solely from a prior report. This does not independently establish every other missing field; per-view empty-scene backgrounds, calibrated transfer functions, and synchronized motion truth remain listed as unavailable in the latest STATUS and project memory.

This feasibility result does not identify a novel mechanism. The latest primary-art screens already close generic range-ridge/distance-energy, generic TR-temporal/multi-view fusion and the prior archived candidate grids. No new classifier candidate was justified from file headers alone, and no fit/training was started.

## Runtime

The restored path was present. Current workspace had 712 MiB free, 5 CPUs and PyTorch 2.14.0+cpu with CUDA false. Process inspection found only historical defunct Python entries and no active training process. The raw 1.96 GB tensor was not materialized into memory.

## Next gate

Continue only with a narrowly defined mechanism that (a) operates on the real-valued clutter-removed range-time input, (b) is distinguished from primary literature and archived experiments, and (c) has a frozen user-disjoint evaluation with destructive controls. A method requiring complex echoes, pulse sequences, background references, calibration or independent motion labels cannot be run from the current restored cache. The Direction 1 goal remains unvalidated.
