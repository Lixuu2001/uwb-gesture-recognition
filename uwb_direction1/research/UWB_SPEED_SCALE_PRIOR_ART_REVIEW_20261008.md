# UWB speed-scale and multiscale prior-art update — 2026-10-08

## Decision

The broad claim “UWB hand gestures vary in speed, so radar-specific multiscale temporal features improve robustness” is already directly claimed and tested on UWB-Gestures in a March 2026 *IEEE Access* paper. It is not an eligible novelty claim for this project. The prior paper uses a different exact representation and network than this project's previously screened dilation intervention, but that distinction is not enough to repackage the same physical hypothesis as a new contribution.

## Full-text source reviewed

Kim, Bong-Seok; Choi, Rock-Hyun; Jang, Seonghyun; Kim, Sangdong. “Robust UWB Radar Gesture Recognition Addressing Speed-Induced Scale Variations via Multi-Scale Feature Extraction.” *IEEE Access* 14 (March 2026), 40034–40041. DOI [10.1109/ACCESS.2026.3671243](https://doi.org/10.1109/ACCESS.2026.3671243). The full eight-page publication PDF is hosted by DGIST Scholar: [publisher-version PDF](https://scholar.dgist.ac.kr/bitstream/20.500.11750/60259/2/Robust_UWB_Radar_Gesture_Recognition_Addressing_Speed_Induced_Scale_Variations_via_Multi_Scale_Featu.pdf). Metadata record: [DGIST Scholar article page](https://scholar.dgist.ac.kr/handle/20.500.11750/60259).

## What the paper establishes

- It uses the same UWB-Gestures source dataset and describes eight participants, twelve hand gestures, 100 repetitions per participant/gesture (9,600 total), and three X4 radars in left/top/right positions.
- It transforms slow-time radar data into a range-Doppler representation and applies IIR high-pass clutter removal.
- Its first CNN layer uses parallel 3×3, 5×5 and 7×7 filters per radar view; view-specific features are processed separately and later fused.
- It explicitly motivates the design as handling speed-induced time-frequency scale mismatch, reports 98.12% accuracy, a roughly 1.3% parameter increase and 43% more computation, and reports 17.72 ms inference on RTX 4080.
- The paper says the test data were not used for training. The reviewed evaluation description does not specify whether the test split is participant-disjoint or name held-out participants. The number is therefore not directly comparable to this project's eight-fold leave-one-participant-out results; this missing detail is not evidence of leakage. The paper reports the full 9,600 sample count, whereas this project freezes a quality filter retaining 9,300.

## Relationship to the prior project experiment

The project ran a distinct, bounded experiment on 2026-10-04: a single dilation-2 intervention on one stride-1 convolution axis of the existing small range-time CNN, with a distance-axis dilation control and matched parameter count. Its fixed-tail six-user screen was 93.5000% for time dilation, 93.3699% for range dilation, and 92.7222% for ordinary CNN. Time dilation beat the ordinary CNN by 0.7778 percentage points but beat the matched range dilation by only 0.1301 pp, won on 3/6 users, and missed the preregistered +0.25 pp/4-of-6 joint gate. That experiment did not measure true execution speed per gesture. Its negative gate remains final for that fixed hypothesis; do not rerun or alter the controls/thresholds.

The paper's parallel-kernel range-Doppler MSFE is not the same computational graph as this project's range-time dilation. Nonetheless, it directly preempts the broad physical idea of speed-robust multiscale UWB gesture recognition on this dataset. The prior paper does not prove that the project's dilation result was caused by real speed variation; no such per-sample motion-speed truth is available in the restored data.

## Consequence

Close multiscale/speed robustness as an originality route. Do not reimplement the paper as a “new method,” run another dilation/kernel-size sweep, or compare the paper's 98.12% against this project's 8-fold score as though the protocols matched. A future speed-specific study requires a materially distinct endpoint and measured per-sample speed or a new, participant/session-separated capture. Until a direct-prior-art-cleared candidate and its required truth data exist, no training is justified.

This review used the primary full-text PDF and the publisher's institutional repository record. No current radar arrays or labels were accessed, and no model fitting/training was performed.
