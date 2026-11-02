# 2026-10-08 — multiradar hand-trajectory novelty gate

## Result

The new-capture candidate was reviewed against direct prior art. Generic single-view versus multiview hand trajectory/observability and generic view-count/coverage are already represented by a 2020 X4 impulse-radar continuous-writing paper; the UWB-Gestures dataset paper also identifies hand tracking/trajectory prediction as a multiradar use, and a 2026 paper reports three-view fusion on the same dataset. The current broad candidate therefore fails its novelty gate **as written**. This is a research-design result, not an empirical model result.

The narrower proposed endpoint (synchronized external hand truth plus line-of-sight ambiguity under participant/session/placement holdout) is not shown by the reviewed evidence to have been answered exactly. It remains unvalidated and is not yet a defensible contribution. It needs a materially sharper question and another direct prior-art review before any capture.

## Evidence and scope

- Khan, Leem & Cho (2020), “In-Air Continuous Writing Using UWB Impulse Radar Sensors,” IEEE Access 8:99302–99311, DOI [10.1109/ACCESS.2020.2994281](https://doi.org/10.1109/ACCESS.2020.2994281): four X4 radars; TOA/peak-magnitude range observations; trilateration plus EKF hand-path reconstruction; tangent/time character representation; paper discusses why three views were inadequate for its approximately 60-degree beam coverage and uses four. This blocks a generic X4 multiradar trajectory or 3-vs-4 coverage novelty claim. The continuous-writing task differs from UWB-Gestures, so this is not a claim that all gesture-specific external-truth endpoints are preempted.
- Ahmed et al. (2021), DOI [10.1038/s41597-021-00876-0](https://doi.org/10.1038/s41597-021-00876-0): three independent X4 views in UWB-Gestures and a stated multiradar use for hand tracking/trajectory prediction. The restored cloud package has no per-frame common clock or external hand-motion truth.
- Xue, Yang & Zhu (2026), DOI [10.3389/fnins.2026.1864448](https://doi.org/10.3389/fnins.2026.1864448): same-dataset three-view fusion/view-ablation precedent. Its split/reporting limitations are not evidence of leakage; scores are not directly comparable to the project's eight-fold evaluation.
- Novelda's public X4 driver exposes manual/X4/MCU sweep-trigger modes and frame-period controls. This corrects any categorical claim that X4 has no trigger control; actual shared-trigger capability and residual timing remain hardware/firmware-specific and must be tested on the exact units.

## Execution and data audit

- Read the latest remote branch state before review; current working parent was `dfc181ad9367030475fec277bf1ef3c1687910f0`.
- Confirmed the existing cloud restore path remains `/workspace/uwb-restored-20260930`; did not redownload or alter it.
- Current restored-package filename inventory found no synchronized video/motion-capture assets. This is limited to the package inspected; no assertion is made about files elsewhere.
- Cloud runtime has no attached X4 capture interface, optical-motion-capture stream, or new participant/session/placement acquisition. No new data, labels, radar arrays, model fits, or training runs were used in this review.
- No background training was started. The research goal remains unvalidated.

## Disposition

Do not proceed with the broad “one versus three views resolve trajectory ambiguity” novelty claim. Keep the acquisition design archived as a hypothesis only. A future candidate must specify a materially distinct endpoint and truth source, pass primary-paper review, and pass exact-device trigger/timing preflight before collection or fitting. No alternate method currently clears that gate from the evidence available in this cloud session.

Details are recorded in [`research/HAND_TRAJECTORY_PRIOR_ART_REVIEW_20261008.md`](../../research/HAND_TRAJECTORY_PRIOR_ART_REVIEW_20261008.md). The existing 789-file source snapshot and raw arrays are unchanged.
