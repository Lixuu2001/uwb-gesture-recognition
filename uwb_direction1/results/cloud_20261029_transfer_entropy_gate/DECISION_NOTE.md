# 2026-10-29 — directed inter-range information-flow gate

## Candidate
A single-view feature that estimates lagged directed predictive information (transfer entropy) between neighboring range-bin envelope traces in the existing clutter-removed range-time maps. This differs from the just-screened zero-lag/second-order co-modulation statistic, but the mere change to a standard nonlinear dependence estimator does not establish a new radar method.

## Bounded primary-source review
- Schreiber introduced transfer entropy as a general measure of asymmetric predictive information flow; it is not radar-specific: [Physical Review Letters 85 (2000), 461–464](https://doi.org/10.1103/PhysRevLett.85.461).
- Ritchie and Jones already represent gesture motion at multiple range bins, generating a spectrogram per bin to retain time, Doppler, range, and polarization structure: [2019 IEEE Radar Conference, DOI 10.1109/RADAR.2019.8835782](https://doi.org/10.1109/RADAR.2019.8835782).
- Xue et al. model local and long-range temporal structure on time-range maps for the same UWB-Gestures corpus, with three Left/Top/Right views: [Frontiers in Neuroscience (2026), DOI 10.3389/fnins.2026.1864448](https://doi.org/10.3389/fnins.2026.1864448).

Two targeted searches for transfer entropy/directed information with radar gesture range-time or UWB gesture terms returned no exact direct application. That bounded absence is not novelty clearance. Here, applying a standard TE estimator to adjacent range cells remains generic feature engineering and overlaps established multi-range temporal radar representations. Only 90 time samples per clip are available in the restored map; estimator embeddings/discretization and finite-sample bias would be consequential. The maps also lack synchronized hand-kinematic truth. A score change would not establish that the estimated direction is physical hand-motion information flow rather than common latent gesture dynamics or preprocessing effects.

## Decision
**Do not implement or fit this candidate.** It fails the method-novelty and physical-verification gates before any data-value/label read. This is a bounded literature/feasibility no-go for this candidate, not a predictive-utility result and not a proof that all methods on current data are impossible. No labels, map values, predictions, or test users were read; no fitting/training or parameter search ran. Current process scan found no UWB job, and the workspace scan found no model checkpoint.

Cloud availability was checked without restoring again: `/workspace/uwb-restored-20261020/RESTORE_VERIFIED.json` confirms the existing 86-file transfer SHA; the untouched original project subtree still has 789 files. Runtime is Python 3.12.14 with NumPy/SciPy/scikit-learn, 5 CPUs, and no PyTorch; network policy is restricted/enforced. No raw arrays were added to Git.

## Next
Keep the method-first no-go. Only reopen candidate screening if a materially different observable mechanism or new primary-source evidence appears. In the absence of that, the feasible deliverable is a clearly scoped reproducibility/engineering audit; it must not be presented as the requested original method contribution.
