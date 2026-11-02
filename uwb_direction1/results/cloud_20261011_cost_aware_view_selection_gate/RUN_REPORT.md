# Cost-aware view-selection prior-art gate

Date: 2026-10-11  
Branch: `codex/uwb-direction1-cloud-20260930`

## Candidate screened

Because the project permits one or three radars, screened whether per-instance selection of a subset/view could form a new method: use one view for easy events, acquire/use more views when confidence is low, and optimize accuracy against sensor cost.

## Prior art and decision

The same-corpus Xue et al. 2026 paper already reports single-, dual-, and tri-view input comparisons and adaptive feature selection/fusion. Generic instance-wise dynamic sensor selection to trade classification loss against sensor count is also published for human-activity recognition (AAAI 2020). The project has additionally closed generic sensor dropout, view-subset fusion, and angular-diversity robustness in its prior-art review. Therefore, a per-event view gate or a cost/accuracy curve alone does not establish a new contribution. A truly new algorithm would need a radar-specific acquisition or observation-cost mechanism and evidence beyond generic confidence gating; the current stored corpus has only fixed Left/Top/Right captures, not power/latency/acquisition-cost measurements per view. No such distinction is established, so no view-selection experiment was run.

Sources: [same-corpus MV-GLFNet full text](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1864448/full), [Instance-Wise Dynamic Sensor Selection for Human Activity Recognition](https://ojs.aaai.org/index.php/AAAI/article/view/5461). This was a bounded check, not an exhaustive survey.

## Runtime and data safety

The restored dataset path `/workspace/uwb-restored-20260930` is present. Fresh process inspection found only defunct historical Python PIDs 598, 607, 4719; no active training process. No labels, array values, or checkpoint contents were read. No model fit, training, checkpoint change, or data download occurred.

## Decision

Do not run a generic per-sample view-selection/uncertainty gate. The project still has no validated original method. Further work must identify a radar-specific method with a clear distinction from same-corpus multi-view fusion and general sensor selection; otherwise, the missing input is a new calibrated acquisition with measured view cost and/or an independent annotated cohort.
