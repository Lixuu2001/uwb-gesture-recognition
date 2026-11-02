# Same-corpus speed-robustness and fusion prior-art review

Date: 2026-10-11 (UTC)

## Question and decision

Can generic multi-scale/speed-robust feature extraction, CNN-LSTM attention, early multi-view fusion, or synthetic radar-noise robustness be claimed as a new contribution on this project’s UWB-Gestures corpus?

**Decision: no.** Two papers now establish same-corpus prior art for these broad claims. Neither published accuracy is directly comparable to the project's fixed eight-fold held-user evaluation because the protocols are not established as identical.

## Primary-source findings

1. **Kim et al., “Robust UWB Radar Gesture Recognition Addressing Speed-Induced Scale Variations via Multi-Scale Feature Extraction,” IEEE Access 14 (2026), 40034–40041, DOI [10.1109/ACCESS.2026.3671243](https://doi.org/10.1109/ACCESS.2026.3671243).** The paper explicitly says it evaluates on UWB-Gestures: 12 gestures, eight participants, three left/top/right X4 radars, with 100 repetitions per participant/gesture. Its MSFE uses parallel 3×3, 5×5, and 7×7 kernels in the first convolution layer, motivated by speed-induced range-Doppler scale variation; it reports 98.12% accuracy. The accessible full text says a test set is held out, but does not specify a participant-disjoint split or identify a cross-validation protocol. Thus the reported accuracy cannot be compared to our eight held-user folds, and the novelty of generic speed/multiscale processing on this corpus is closed. Source: [author-uploaded full text](https://www.researchgate.net/publication/401674921_Robust_UWB_Radar_Gesture_Recognition_Addressing_Speed-Induced_Scale_Variations_via_Multi-Scale_Feature_Extraction); institutional record: [DGIST Scholar](https://scholar.dgist.ac.kr/handle/20.500.11750/60259).

2. **Khan & Kwon, “Radar-Based Hand Gesture Recognition With Feature Fusion Using Robust CNN-LSTM and Attention Architecture,” IEEE Access 13 (2025), 69281–69291, DOI [10.1109/ACCESS.2025.3558293](https://doi.org/10.1109/ACCESS.2025.3558293).** The authors use the same 12-class, eight-person, three-X4 UWB-Gestures corpus. They report early spatial-temporal fusion with CNN-LSTM-attention and robustness tests for Gaussian noise, signal inversion, and multipath; abstract reports 98.33%. The full text specifies a 20% test split with the remainder for training/validation, but does not state that participants are held out. Treat this as sample-level/test-holdout prior art, not evidence for held-user generalization. Source: [full text](https://www.researchgate.net/publication/390568482_Radar-Based_Hand_Gesture_Recognition_With_Feature_Fusion_Using_Robust_CNN-LSTM_and_Attention_Architecture); institutional record: [KNU Scholar](https://research.knu.ac.kr/en/publications/radar-based-hand-gesture-recognition-with-feature-fusion-using-ro/).

## Protocol relevance

Our project’s primary evaluation is eight folds with one participant held out per fold, and macro accuracy averaged across held users. Neither paper's headline score is established to use this exact split and metric. The papers therefore close broad method novelty on this same corpus, but their percentages must not be used as head-to-head results.

The 2026 paper's fixed first-layer multi-scale kernels and speed-robust framing are enough to reject a generic MSFE proposal, even though its held-user protocol and per-speed condition labels are not documented in the accessible text. The 2025 paper is direct prior art for generic CNN/LSTM/attention, early multi-view fusion, and synthetic robustness tests on this corpus.

## Cloud execution audit

- Managed cloud observed running, spec revision 640; CPU-only PyTorch 2.14.0, CUDA unavailable; five CPUs.
- `/workspace/uwb-restored-20260930` is present (2.0 GB). Workspace filesystem has 712 MB free of 32 GB.
- Process scan found no active training job (only historical defunct Python processes).
- No data labels, tensor values, or checkpoint contents were read. No archive was fetched. No fitting or training was started.
- No original source snapshot or raw data was modified.

## Next decision

Do not repackage generic multiscale kernels, speed augmentation, CNN-LSTM attention, early fusion, or generic noise robustness as the contribution. The method goal remains unvalidated. Continue only with a signal-specific mechanism that is both distinguishable from this same-corpus prior art and testable with the currently available processed inputs; if it requires pulse-level echoes, synchronized motion labels, or calibrated view geometry, obtain those through new authorized acquisition before proposing a training run. This bounded literature review is not an exhaustive novelty search and makes no legal conclusion.