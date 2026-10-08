# Hand trajectory and multiradar view-count prior-art review — 2026-10-08

## Decision

The archived proposal in [`NEW_CAPTURE_PROTOCOL_20261008.md`](NEW_CAPTURE_PROTOCOL_20261008.md) does **not** clear its novelty gate as currently phrased. It asks whether one X4 view is ambiguous for hand-motion/gesture trajectories and whether multiple views resolve the ambiguity. That broad combination of X4 multiradar trajectory reconstruction and view-count/coverage reasoning has a direct prior in Khan, Leem & Cho (2020), and the UWB-Gestures dataset paper itself describes tracking and trajectory prediction as uses of multiple radars. Therefore, generic “single versus three views for hand trajectory observability” is not an eligible novelty claim.

This is a bounded novelty decision, not a claim that every possible participant-held-out, placement-transfer, uncertainty-calibration, or gesture-specific question has been answered. The specific additions proposed in 2026—synchronized external hand truth, a line-of-sight ambiguity endpoint, and participant/session/placement holdouts—are not established by the evidence reviewed here as having been tested in Khan et al. They nevertheless need their own direct literature review and a concrete difference in scientific question or endpoint before any acquisition. Do not describe the current protocol as a novel method or start a capture/training queue on that basis.

## Direct evidence

| Source | Relevant evidence | Implication and limit |
|---|---|---|
| Khan, Leem & Cho, “In-Air Continuous Writing Using UWB Impulse Radar Sensors,” *IEEE Access* 8 (2020), 99302–99311, [doi:10.1109/ACCESS.2020.2994281](https://doi.org/10.1109/ACCESS.2020.2994281) | Uses four X4 IR-UWB radars in a rectangular arrangement. Per-radar time-of-arrival and peak-magnitude observations are combined by trilateration; an EKF with a position/velocity model reconstructs a hand path, which is represented by tangent/time features for continuous character recognition. The paper discusses that its approximately 60-degree beam coverage made three viewpoints inadequate for its setup and selected four. | Directly preempts presenting X4 multiradar hand-trajectory reconstruction or a generic 3-versus-4 view-coverage comparison as new. The task is continuous writing and the reported setup/endpoint differs from the proposed UWB-Gestures capture; this is not evidence that every gesture-specific external-truth or domain-transfer question is preempted. |
| Ahmed et al., “UWB-Gestures: A public dataset of dynamic hand gestures acquired using impulse radar,” *Scientific Data* 8 (2021), [doi:10.1038/s41597-021-00876-0](https://doi.org/10.1038/s41597-021-00876-0) | Describes three independently operated monostatic X4 views, separately saved, and notes multiradar use can support hand tracking and trajectory prediction. The cloud-restored tensor is not accompanied by a per-frame common clock or external hand-motion truth. | Tracking/trajectory as a multiradar motivation is already part of the dataset context. The paper does not establish synchronized trajectory ground truth in the restored cloud package. |
| Xue, Yang & Zhu (2026), *Frontiers in Neuroscience*, [doi:10.3389/fnins.2026.1864448](https://doi.org/10.3389/fnins.2026.1864448) | Reports three-view fusion and view ablations on UWB-Gestures. | Generic three-view gesture fusion is already a same-dataset precedent. Do not infer leakage from its reporting limitations, and do not compare its split-specific score directly with this project's eight-fold results. See [`results/cloud_20261008_direct_prior_art/RUN_REPORT.md`](../results/cloud_20261008_direct_prior_art/RUN_REPORT.md). |

The resulting overlap is enough to reject the current broad novelty wording, but not enough to assert that the exact new physical endpoint is already solved. A narrower question could proceed only after comparing its exact observable, truth signal, target population/conditions, and endpoint against primary full texts—not by changing the name of the same trajectory/fusion idea.

## Hardware implication

The Novelda Legacy-SW X4 driver source exposes sweep-trigger control (`MANUAL`, `X4`, and `MCU` modes), a sweep-trigger callback, and frame-trigger-period control. This means it would be inaccurate to say that X4 has no trigger mechanism. The public driver code alone does **not** prove that the exact X4 boards, firmware, and acquisition stack available for a study can share a trigger or meet a cross-device timing tolerance. Any future hardware preflight must identify board revision/firmware and verify measured trigger/timestamp residuals on the actual units before recruiting or fitting. Official source: [Novelda Legacy-SW `x4driver.h`](https://github.com/novelda/Legacy-SW/blob/master/Examples/X4Driver_RaspberryPi/src/XDriver/x4driver.h) and [`taskRadar.cpp`](https://github.com/novelda/Legacy-SW/blob/master/Examples/X4Driver_RaspberryPi/src/taskRadar.cpp).

## Cloud evidence boundary

The restored package contains the already verified radar data and cached arrays; a local file inventory found no synchronized video or motion-capture files. This is a statement about the current cloud package only, not about whether any video is hosted elsewhere. No X4 acquisition interface, optical-motion-capture stream, or new participant/session/placement capture is attached to this execution environment. The static arrays cannot supply independent hand-motion truth that was not recorded with them.

No labels or radar arrays were accessed for this review, and no fitting/training was run. The prior-art conclusion is based on the cited papers, prior archived review, official driver source, and a filename-only inventory of the restored package.

## Next gate

Before another physical-observability proposal:

1. State a task-specific endpoint and external truth source that are materially different from generic UWB trajectory reconstruction, view-count coverage, and same-dataset fusion.
2. Review the relevant primary full texts for that exact endpoint and disclose any unresolved coverage.
3. Establish that the exact X4 hardware can produce the required synchronized observations; identify the actual board/firmware and verify timing empirically.
4. Obtain new, consented measurements with new participants/conditions if the question concerns population/session/placement generalization. Freeze participant roles, endpoint, exclusions, controls, and stopping rule before model fitting.

Until all four gates have evidence, keep the candidate at design stage and do not start training. The overall project goal remains unmet.
