# Focused SWAD paper/source cross-check

Source: Cha et al., [SWAD: Domain Generalization by Seeking Flat Minima](https://arxiv.org/pdf/2102.08604), arXiv v4, 3 November 2021, NeurIPS 2021. Reviewed theory assumptions/statements, sections3.1–3.2,4.3,5 and AppendixB.1/B.4. Algorithm1 was also inspected in the PDF page16 image. Full appendix proof/result verification is incomplete; the full-paper-review flag remains false. No paper text or PDF is copied into Git.

## Paper facts (focused paraphrase)

Section 3.2 and Algorithm 1 average densely sampled weights over a validation-loss-defined interval. Algorithm 1 sets the threshold to r times the mean over the convergence window, and ends collection when the tolerance window stays above it. Appendix B.1 reports Ns=3, Ne=6, r=1.3 except VLCS (1.2), with ImageNet-pretrained ResNet-50 and frozen BN statistics. These values were validation-selected in the source study. Section 2 assumes bounded loss and a common labeling function. Section 5 describes SWAD as a heuristic, without a theorem guaranteeing it finds flat minima.

## Frozen code cross-check and implications for this experiment

Pinned author source252190eb881169e736beccf51af5caaa5c5cf666, domainbed/swad.py lines102–103, computes mean convergence loss times(1+tolerance_ratio). The frozen tolerance_ratio0.3 therefore means multiplier1.3, not multiplier0.3. This confirms the existing mapping; no configuration change follows this review. The original queue/finalization boundary behavior remains copied and tested, including segment-merge counts that need not equal unique optimizer-step counts.

The active UWB study is a component transfer under the matched GAP617612/globalBCE/AdamWLookahead/B64/64/seed20261029 protocol. Epoch-level monitoring on a separate validation person, native trainableBN, and one source-only recalibration pass are registered adaptations. Their effects require the predeclared native-reference, BN-only and fixed-tail controls. The observed averaging range is an output of the frozen rule; it is not replaced with a test-preferred window.

Our BCE objective is unbounded as incorrect logits grow; it is not itself a bounded loss directly covered by the stated theorem. No flatness or source-target divergence measurement has been performed in this run. Weight averaging, exact replay, or a future accuracy improvement therefore cannot alone establish a causal flatness explanation or a new theoretical contribution. Checkpoint cadence and validation population may affect window endpoints, which reinforces documenting these conditions without retuning this active cohort.

All scientific files/configs, validation selection, seed, native optimizer, sourceBN rule, sixfold gate and private-before-one-score requirement remain frozen. Users7–8 remain outside new method selection; historical eight-person development is exploratory.
