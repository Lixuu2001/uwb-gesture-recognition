# Independent wearable FMCW gesture cohort gate

Date: 2026-10-15 01:16:57 UTC  
Branch head at review: `929e40c98b0b2cf2b0f81896ef9127ff20d0d8e2`  
Review type: bounded public-source search and dataset-card metadata review. No archive files were downloaded; no project arrays, labels, checkpoints, fitting, or training were accessed.

## Finding

A public, apparently separate wearable-hand-gesture dataset was identified in the maintainer repository and its linked Hugging Face dataset card. The maintainer README describes 3,300 recordings from 11 participants, ten classes (30 samples per class per participant), collected in 2025 using a wrist-mounted Infineon BGT60TR13C 60-GHz FMCW radar at 24 frames/s. Each sample is listed as a float32 array of shape (71, 3, 64, 64); the repository says the release is about 22 GB. The repository says its associated article is forthcoming and does not yet provide the paper link/method details.

This is a separate cohort lead, but it does not currently support independent validation of the X4 IR-UWB 12-class task. The wearable classes (pinch/release, palm/fist transitions, hand signs, etc.) do not provide an evident one-to-one label map to X4's directional swipes, rotations, inward push, and empty class. The sensing setup also differs substantially: wrist-mounted FMCW raw tensors versus free-space X4 range-time maps. Broad cross-sensor transfer is already prior art in adjacent radar tasks; the exact gesture-specific method and a defensible shared-task protocol have not been cleared. Therefore this dataset is not treated as an external test set or as a validated contribution.

## Execution feasibility

Current cloud inspection found 746,266,624 bytes free, CPU-only PyTorch 2.14.0 with CUDA unavailable, and no active training process. The published archive estimate is about 22 GB, so it cannot be staged in the current workspace. The managed network policy has no configured custom egress hosts; no dataset download was attempted. No Drive objects or source snapshot were changed.

## Primary sources

- Maintainer dataset README: https://github.com/otsol/wearable-mmwave-radar-dataset
- Linked Hugging Face dataset card: https://huggingface.co/datasets/otsol/wearable-mmwave-radar-dataset
- For the X4 task being continued: Ahmed et al., *UWB-gestures, a public dataset of dynamic hand gestures acquired using impulse radar sensors*, Scientific Data 8, 102 (2021), https://doi.org/10.1038/s41597-021-00876-0

## Decision

Record this as a new independent-cohort lead, not a method result. Do not download or train on it in this executor. A later cross-sensor study would first need the actual method paper, a justified class mapping or a deliberately reframed label-free transfer question, adequate storage/network, and a frozen protocol. The original direction-one contribution remains unvalidated.
