# Preregistration: range-time echo Betti-curve source feasibility gate

**Date:** 2026-10-28 UTC  
**Data read:** none before this preregistration.  
**Run type:** one source-only diagnostic with a frozen gate; not a claimed method result.

## Candidate and prior-art distinction

The candidate is a single-view representation of the real-valued clutter-removed range-time map: per-sample normalized upper-level-set Betti curves (4-connected components and enclosed holes) across eight fixed upper-tail quantiles. The intended observable is the persistence of connected echo-pattern geometry across amplitude thresholds. It is not a hand trajectory estimate, calibrated motion field, or new persistent-homology algorithm.

A full text review of [Maršálek et al., “Persistent Homology Approach for Human Presence Detection from 60 GHz OTFS Transmissions,” Sensors 23(4), 2224 (2023)](https://doi.org/10.3390/s23042224) found persistent homology used on OTFS delay–Doppler maps to find/score target peaks and compare detection against CFAR. That is adjacent radar precedent, not gesture classification. A full publisher article by [Xue, Yang, and Zhu (2026)](https://doi.org/10.3389/fnins.2026.1864448) evaluates the same 12-class, eight-subject, three-X4-view UWB-Gestures corpus and proposes early fusion plus local/global networks and adaptive bilinear feature fusion; no topological descriptor is reported in the reviewed method. The bounded search found no direct UWB dynamic-gesture classification paper using range-time Betti curves, but it was not exhaustive. Accordingly, this gate tests whether the map topology is worth a separate validation design; it does not establish novelty or support a contribution claim.

This descriptor differs from the already closed local patch correspondence, range-ridge/trajectory/energy, recurrence/RQA, phase-motion/reliability, temporal reversal, and generic multi-view routes. Its direct controls are a spatial-layout-free amplitude-quantile baseline and a fixed spatial-shuffle topology null that preserves each map’s intensity multiset.

## Frozen data boundary and evaluation

Use only participants 1–6, with a six-fold internal leave-one-source-participant-out split; participants 7–8 gesture labels are not read. The quality mask is fixed from the archived source audit: participant 2 zero-based classes 10/11 and participant 6 zero-based class 3 are excluded; expected retained source counts are 1,200 / 1,000 / 1,200 / 1,200 / 1,200 / 1,107. The executor has the restored cache and original source snapshot, no active UWB training process, and no common serialized checkpoint; current stack has NumPy/SciPy/scikit-learn, five CPUs, and no PyTorch.

For each of Left, Top, and Right independently, compare the Betti descriptor against the amplitude and spatial-shuffle controls using the same train-only StandardScaler and fixed RidgeClassifier(alpha=1.0). No radar view is selected, no views are fused, no parameter grid or p-value is used, and no CNN is trained. Scores are macro recall per held-out source participant, reported for all six source folds and all three fixed views.

The predeclared GO condition is: overall topology-minus-amplitude gain at least 0.010 absolute macro recall; at least four of six held-source users improve in at least two of three views; no view mean is worse than −0.010; and topology exceeds the spatial-shuffle null by at least 0.010 overall. GO allows designing a separate frozen outer-user study only. A failed gate closes this exact descriptor; no threshold, connectivity, classifier, seed, or view rescue is allowed. Because source users have prior project-development exposure, this is exploratory feasibility, not independent validation.

## Reproduction and resource receipt

- Protocol SHA-256: 04668d3cf2a16514132e88a60dd43c0a1be031acb6da9dd5a1085e2af6ac6290
- Runner SHA-256: ddd58ac798cba14a01d458882c5ec9e30714c25e27c1f5313c8fdf5a6d3350ae
- Hard budget: 900 CPU seconds, independent 900-second wall timeout, one process, one BLAS thread.
- Synthetic descriptor smoke check passed; no restored arrays or labels were opened for this check.
- Data artifacts stay in the cloud workspace and will not be committed to Git.

The experiment remains unstarted at this preregistration commit.
