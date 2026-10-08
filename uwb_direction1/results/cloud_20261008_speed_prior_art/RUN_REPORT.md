# 2026-10-08 — direct prior art for speed-scale robustness

## Finding

A full-text review found a direct March 2026 prior using the same UWB-Gestures dataset for UWB hand-gesture recognition robust to speed-induced scale variation. It proposes first-layer parallel 3×3/5×5/7×7 multi-scale feature extraction on range-Doppler views. Therefore generic “speed-robust multiscale UWB gesture recognition” is not an eligible novelty claim for Direction 1.

The prior paper's exact model is not the same as the project's already completed range-time dilation screen. That screen remains a negative fixed-hypothesis result: time dilation was +0.1301 percentage points over range dilation, won 3/6 users, and failed its preregistered +0.25 pp/4-of-6 gate. It did not measure gesture speed. Neither the prior paper nor that old score validates a new method here.

## Primary source reviewed

Kim et al., “Robust UWB Radar Gesture Recognition Addressing Speed-Induced Scale Variations via Multi-Scale Feature Extraction,” *IEEE Access* 14 (2026), 40034–40041, DOI [10.1109/ACCESS.2026.3671243](https://doi.org/10.1109/ACCESS.2026.3671243). Read the complete eight-page publication version hosted by [DGIST Scholar](https://scholar.dgist.ac.kr/bitstream/20.500.11750/60259/2/Robust_UWB_Radar_Gesture_Recognition_Addressing_Speed_Induced_Scale_Variations_via_Multi_Scale_Featu.pdf) and the [institutional metadata record](https://scholar.dgist.ac.kr/handle/20.500.11750/60259).

The paper describes all 9,600 UWB-Gestures records (8 participants × 12 classes × 100 repetitions), three X4 views, IIR clutter removal, range-Doppler preprocessing, and per-view MSFE branches. It reports 98.12% accuracy and RTX 4080 runtime/complexity values. It says its test data were not used for training, but the reviewed evaluation section does not state participant-disjoint roles or held-out IDs. The reported score is not directly comparable to this project's participant-held-out eight-fold score; absent split detail is not proof of leakage. The paper uses 9,600 records while this project freezes the pre-existing quality mask at 9,300.

## Cloud execution check

- Remote branch head was `77d36aedbd6bb1ab1c3753cdcd60615d1f4a1a44`; required STATUS/memory/reports were read from GitHub before this review.
- The verified restore remains at `/workspace/uwb-restored-20260930`; it was not re-downloaded.
- Runtime remains connected but reports no attached device capabilities; no X4 or optical capture interface is present. Process inspection found no active UWB training run.
- No labels or radar arrays were read for this literature review. No fitting or training was run, and no test scores were produced.

## Disposition

Close the speed-scale/multiscale novelty route; do not restart the prior dilation screen. The project goal remains unmet. A future speed-specific research contribution needs a distinct endpoint and measured per-sample speed or new synchronized, participant/session-separated measurements. No current candidate clears the novelty and measurement gates.

Full details: [`research/UWB_SPEED_SCALE_PRIOR_ART_REVIEW_20261008.md`](../../research/UWB_SPEED_SCALE_PRIOR_ART_REVIEW_20261008.md). Existing 789-file original source snapshot is unchanged.
