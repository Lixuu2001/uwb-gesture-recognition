# View/label reflection-equivariance candidate gate

**Date:** 2026-10-26 04:10 UTC  
**Decision:** a geometry-derived left/right sensor-swap and gesture-label permutation is not cleared for implementation or training on the current package. No experiment ran.

## Candidate

For an in-plane mirror of the gesture, the published class names imply an involution: LR↔RL, diagonal LR↔RL pairs, CW↔CCW, while UD/DU, inward push, and empty stay fixed. Under a symmetric three-sensor layout, a physical mirror might also exchange Left and Right radar roles while leaving Top unchanged. A narrow model could enforce that joint sensor-role / label transformation as equivariance.

This is distinct in formulation from generic feature fusion, but its physical validity depends on exact acquisition geometry, shared timing, view registration, and a known mapping between mirrored hand paths and measured maps. The current files do not establish that mapping. Applying an image flip or swapping two view channels without validating this relation would impose an unsupported synthetic label rule.

## Literature review

- Ahmed et al., *UWB-Gestures: a public dataset of dynamic hand gestures acquired using impulse radar sensors*, Scientific Data 8, 102 (2021), DOI [10.1038/s41597-021-00876-0](https://doi.org/10.1038/s41597-021-00876-0): publisher full text reviewed. It documents three monostatic radar sensors operating independently, with each sensor's data saved separately; the dataset permits single- or multi-radar evaluation and defines the directional gesture labels. It does not provide a validated joint reflection-equivariance transform.
- Xue et al., *Neuromorphic-inspired multi-view global-local fusion for IR-UWB radar dynamic gesture recognition*, Frontiers in Neuroscience (2026), DOI [10.3389/fnins.2026.1864448](https://doi.org/10.3389/fnins.2026.1864448): full text reviewed for setup and prior art. It uses the same corpus, describes the three-view arrangement and performs generic tri-view fusion. This preempts generic multi-view recognition/fusion, although it does not describe the proposed joint reflection group.
- Cohen & Welling, *Group Equivariant Convolutional Networks*, ICML/PMLR 48 (2016): primary paper page reviewed. It establishes group-convolution machinery for rotations/reflections; a generic G-CNN application is not itself novel.
- Kern et al., *Data Augmentation in Time and Doppler Frequency Domain for Radar-based Gesture Recognition*, EuRAD (2022), DOI [10.23919/EuRAD50154.2022.9784553](https://doi.org/10.23919/EuRAD50154.2022.9784553): IEEE record reviewed. Radar-gesture augmentation is prior art.
- Fusco et al., *Physically-Interpretable Data Augmentation for Multi-Range Hand Gesture Recognition Using FMCW Radar Time Series*, IEEE Transactions on Radar Systems 1 (2023), DOI [10.1109/TRS.2023.3320869](https://doi.org/10.1109/TRS.2023.3320869): the university repository abstract was reviewed. It manipulates physically interpretable range/angle time-series to model range, orientation, and interference variations. This makes generic “physical symmetry augmentation” an insufficient novelty claim.

This was a bounded screen, not an exhaustive search. The exact same-corpus sensor-swap / gesture-permutation constraint was not found in the reviewed records; that does not prove novelty. The candidate still fails feasibility because the required relation is not validated by the available synchronized/calibrated inputs.

## Input and evaluation limits

The present package contains clutter-removed real-valued Left/Top/Right range-time maps and participant/class metadata. It lacks a shared frame-trigger record, calibrated sensor extrinsics and a ground-truth paired mirrored hand trajectory. The dataset article says the sensors operate independently and save their data separately. Consequently, per-record cross-view tensors do not by themselves establish simultaneous or geometrically paired measurements. No test-label fitting is relevant here: no feature calculation, fitting, or training was done. All eight participants have prior project-development exposure.

## Runtime check

Managed executor revision 1401 is running with restricted/enforced egress, five CPU cores, and no configured capabilities, secrets, or outbound identities. The restore and 789-file original source subtree are present. PyTorch is absent. Process inspection found no active UWB/training process, and a workspace scan through depth 6 found no `.pt`, `.pth`, or `.ckpt` files. This screen accessed repository records and literature only; no radar arrays or label values were read and no download occurred.

## Disposition

Do not implement or train this candidate unless a pre-existing synchronized/calibrated geometry source is found within the repository. Do not fabricate the symmetry by flipping the processed map or swapping view channels. The novel-method goal remains unresolved. Continue only with a materially distinct, input-observable signal mechanism; if none clears, the remaining defensible study option is a reproducibility/protocol audit, which is not a new method or independent validation.
