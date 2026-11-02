# Range-trajectory representation prior-art gate

Date: 2026-10-11  
Branch: `codex/uwb-direction1-cloud-20260930`

## Candidate examined

A possible next candidate was to derive gesture trajectories from the cached range-time maps (for example, a dominant range ridge or time-varying range centroid) and classify the trajectory under held-user folds. This is signal-compatible in principle with the existing range-time input, so it was screened before any implementation.

## Prior art and decision

The exact-corpus 2026 MV-GLFNet article describes range-time energy trajectories as encoding motion direction, amplitude, and temporal pattern; it already models motion-enhanced range-time maps across three X4 viewpoints. Earlier IR-UWB work also uses time-of-arrival variation as a hand-motion range feature, and describes gesture-region localization from fast-time variance. These precedents mean that generic trajectory/range-displacement extraction from an IR-UWB gesture map is not a defensible standalone novelty claim. The exact extraction choices in the older papers were not fully reviewed here; this is a bounded prior-art gate, not an exhaustive review. [Xue et al. 2026 full text](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1864448/full), [Ren et al. IR-UWB gesture paper](https://doi.org/10.4236/jcc.2016.43015), [Khan et al. 2017 paper](https://doi.org/10.3390/s17040833).

No distinct physical measurement is available to turn a range ridge into verified hand kinematics: the restored cache is clutter-removed and has no synchronized motion ground truth or calibrated physical coordinates beyond its documented range/time axes. A new ridge tracker or a held-user evaluation would be an implementation/protocol change, not evidence of a research contribution. **Do not start a trajectory-feature/classifier grid.**

## Runtime and data safety

The restored dataset remains at `/workspace/uwb-restored-20260930`. Fresh process inspection found no active training process (only defunct historical Python PIDs 598, 607, and 4719). Runtime: Python 3.12.14, PyTorch 2.14.0+cpu, no CUDA. Historical queue locks were left untouched. No labels, radar-array values, or checkpoint contents were read; no fitting, training, or checkpoint mutation occurred.

## Decision

This candidate is closed as generic trajectory extraction. The project still has no validated method. Continue only with a distinct signal-level mechanism whose prior-art distinction can be stated and whose validation is feasible with current inputs; otherwise the actual blocker is lack of a novelty-cleared candidate and/or new calibrated, annotated measurements.
