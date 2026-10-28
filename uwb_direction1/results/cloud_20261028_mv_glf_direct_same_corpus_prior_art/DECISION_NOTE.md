# Direct same-corpus prior-art gate: MV-GLFNet

**Date:** 2026-10-28 UTC  
**Type:** literature-only reconciliation; no data analysis or experiment.

## Reviewed source and scope

Read the full open HTML article by Xue, Yang, and Zhu, “Neuromorphic-inspired multi-view global-local fusion for IR-UWB radar dynamic gesture recognition,” *Frontiers in Neuroscience* 20 (2026), article 1864448, DOI [10.3389/fnins.2026.1864448](https://doi.org/10.3389/fnins.2026.1864448). The review covered the signal/preprocessing description, method, dataset/protocol, ablations, and results sections of the publisher’s full text. The article itself describes “neuromorphic-inspired” as an algorithm-level design, not a spiking implementation or special hardware.

## Direct overlap established by the paper

The article evaluates the public UWB-Gestures corpus and describes 12 classes, eight subjects, three XeThru X4 impulse-radar views (left/top/right), and MTI-processed time-range maps with the same 90×189 dimensions as the project’s released maps. It resizes maps to 64×128 and reports 9,600 samples. Its protocol trains on six subjects and tests on two, with no subject overlap.

Its proposed MV-GLFNet early-fuses the three physical views, uses a ConvMixer local branch and CNN–Transformer global branch, then combines them through gated first-order and bilinear second-order fusion. It reports 98.29% accuracy. Its view ablation also reports single-view, dual-view, and tri-view results. The article therefore directly preempts claims that, on this corpus, the contribution is simply (a) trying another generic single-view backbone, (b) combining local and global map encoders, (c) generic adaptive/second-order feature fusion, or (d) comparing one, two, and three physical radar views.

## Comparability limits

The article’s six-train/two-test subject split is not the project’s eight-fold leave-one-user-out protocol. The published 98.29% must not be compared as if it were a matched result to the archived 93.4417% replay or the exploratory phase-motion figures. Its reported 9,600 samples also differs from the restored quality-filtered count of 9,300; this review did not infer a mask equivalence or sample-level split overlap. The article does not establish that its test users are independent of all prior development in this project, and it is not external-corpus validation.

## Decision

Close the generic architecture/fusion/view-count route as prior art. This strengthens the existing novelty gate but does **not** rule out every possible signal mechanism on the current maps. Any future candidate still needs a distinct physical observable in the real-valued clutter-removed maps, exact prior-art separation, a frozen train-only development plan, and an evaluation that does not use test labels for selection. No such candidate was cleared in this review.

No dataset values, labels, checkpoints, or predictions were read. No training, fitting, download, or Drive operation occurred. No research contribution was validated; the user’s original method-contribution goal remains unresolved.

## Reproduction pointers

- Publisher full text: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1864448/full
- DOI: https://doi.org/10.3389/fnins.2026.1864448
