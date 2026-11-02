# 2026-10-24 — Physics-normalization prior-art and input-feasibility gate

Date: 2026-10-24 UTC  
Remote branch head used as parent: `997e97d29513920ba53c981d8036e09f9c5fb393`  
Scope: bounded primary-source literature check plus read-only cloud preflight. No classifier, fitting, or training was run.

## Literature checked

1. **Sluyters et al., “An Electromagnetic Model and Inversion-Based Approach for Radar Gesture Recognition,” IEEE Transactions on Instrumentation and Measurement 75 (2026), article 6514420, DOI [10.1109/TIM.2026.3729019](https://doi.org/10.1109/TIM.2026.3729019).** The publisher's IEEE Xplore abstract says the method uses a full-wave electromagnetic model and inversion to normalize radar measurements into a Green's-function representation intended to be independent of radar hardware and background scene. It reports a two-radar same-scene comparison, distance/angle sensitivity analysis, and evaluations over 20 gestures. This is an abstract-level review only; the full paper was not accessed. It is sufficient to close generic “physics-based radar/background invariant normalization” as a novelty claim, but does not establish the same cohort or a comparable held-user protocol. [IEEE Xplore record](https://ieeexplore.ieee.org/abstract/document/11672853/).

2. **Kim et al., “Robust UWB Radar Gesture Recognition Addressing Speed-Induced Scale Variations via Multi-Scale Feature Extraction,” IEEE Access 14 (2026), 40034–40041, DOI [10.1109/ACCESS.2026.3671243](https://doi.org/10.1109/ACCESS.2026.3671243).** The official DGIST repository record describes first-layer range-Doppler multi-scale features for speed-induced scale variation on UWB-Gestures and reports 98.12%. The record/abstract was checked; the full article was not opened. This corroborates the project's existing speed/multi-scale prior-art closure; its headline score is not comparable to the project's eight-user macro protocol. [DGIST repository record](https://scholar.dgist.ac.kr/handle/20.500.11750/60259).

The Oct 20 report already documents the same-corpus Xue et al. global/local and multi-view fusion prior art. No score across papers was compared in this run.

## Fit to the current cloud inputs

The restored input present is the existing clutter-removed, real-valued tensor `X_clutterremoved.npy` (recorded schema `[9593, 3, 90, 189]`, float32). The current prepared cache does not include the raw complex pulse echoes or measured per-view background/system calibration needed to reproduce a coherent full-wave inversion or claim hardware-invariant Green's-function measurements. Turning the cached envelope map into an inferred “physical” normalization would not reproduce that measurement procedure.

The electromagnetic-inversion route is therefore both prior-art-covered at the generic-method level and infeasible to validate from the current cache. A future study of that mechanism needs the measured raw/calibrated channels and a frozen user-disjoint protocol; no such acquisition hardware or extra measurements are available in this cloud. No new candidate passed the novelty + observability gates.

## Cloud preflight and decision

- Managed runtime status: connected/running, spec revision 1299; network policy restricted/enforced, no custom allowed hosts, capabilities, secrets, or outbound identities configured.
- Runtime: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0; PyTorch absent; 5 CPU cores; CUDA unavailable; 17,651,268 kB available RAM; 25,847,521,280 bytes free in `/workspace`.
- `/workspace/uwb-restored-20261020` is present. This is a path-presence check; the migration was not repeated and Drive permissions were not changed.
- Process-name check found no active training process. A workspace filename scan found no serialized `.pt`, `.pth`, or `.ckpt` model/optimizer checkpoint; checkpoint contents were not read.
- No project label values, tensor values, model weights, or checkpoint contents were read. No data were downloaded; no experiment or training was started.

Decision: preserve the unresolved status and stop before another model/feature grid. The research goal is still **not validated**. Continue only after a technically distinct, literature-cleared mechanism fits the available measured input, or after the required raw/calibrated acquisition becomes available.

## Reproduction and integrity scope

This report adds no model artifact. The source snapshot under `uwb_direction1/original/project/` was not edited. Historical migration and cached LTR replay remain documented in the Oct 20 restore report; this run did not re-run their array-level audits.
