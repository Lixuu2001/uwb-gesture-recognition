# 2026-10-29 — dynamic-mode feature candidate gate

## Candidate screened
Apply standard Dynamic Mode Decomposition (DMD) to each existing clutter-removed, real-valued range-time map and use modal/eigenvalue summaries as a single-view gesture descriptor. This is distinct from the recent transfer-entropy and adjacent-bin covariance candidates in mathematical form, but it uses no new signal measurement.

## Bounded primary-source review
- Sharma & Panda, *Signal, Image and Video Processing* 18 (2024), DOI 10.1007/s11760-024-03038-y, already applies DMD to spatiotemporal hand-gesture imagery and feeds the resulting features to a gesture classifier. This is a vision modality, not radar, but it establishes direct DMD-for-gesture prior art.
- The same-corpus Xue et al. paper, *Frontiers in Neuroscience* (2026), DOI 10.3389/fnins.2026.1864448, already models local and long-range temporal structure on UWB-Gestures time-range maps. A DMD summary of those maps would be a different readout, but is not by itself a distinct sensing mechanism.
- DMD has also been used in radar motion/target processing, including DMD-CFAR target detection (IEEE Sensors 2025, DOI 10.1109/SENSORS59705.2025.11331185). The full paper was not available in this bounded screen; this source is used only as evidence that DMD is not a new radar processing primitive.

A targeted exact-term search found no direct DMD classifier on this specific IR-UWB gesture cache. That gap alone does not establish method novelty. Applying an existing spatiotemporal decomposition to an existing map and attaching a classifier is standard feature transfer. The maps are already clutter-removed and contain only real-valued magnitudes; no raw coherent echoes or independent hand-motion truth are available to define/validate a new radar-specific dynamical model. The proposed use therefore has no concrete algorithmic distinction or observable physical claim that separates it from the prior work.

## Decision
**Do not implement or fit this candidate as a research contribution.** It fails the novelty gate before data-value/label access. This is a bounded literature/feasibility no-go, not a measured accuracy result and not proof that every current-data method is impossible. No arrays, labels, predictions, or test-user outcomes were read; no model fitting/training or parameter search ran. The current process/checkpoint scan found no UWB run or model checkpoint.

Runtime/data check (no restore repeated): existing transfer receipt remains verified (86 payload files); original source snapshot remains 789 files and untouched. Runtime: Python 3.12.14, 5 CPUs, NumPy/SciPy/scikit-learn installed, PyTorch absent; network is restricted/enforced. No raw arrays were committed.

## Next
Keep the method-first gate. Do not relabel standard DMD-on-map feature engineering as a new method. Reopen only if a materially distinct, physically interpretable observable or new primary-source evidence appears; otherwise report the remaining work as a reproducibility/engineering scope without claiming the original research goal.
