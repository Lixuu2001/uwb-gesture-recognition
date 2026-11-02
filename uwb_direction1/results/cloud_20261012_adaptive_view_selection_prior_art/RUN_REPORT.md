# Adaptive per-instance radar-view selection prior-art gate

Date: 2026-10-12 UTC  
Branch head before review: `c778daf30ce936cc5c7aac1724f041a6c6475f51`

## Candidate screened

A possible experiment was to select one or more of the three X4 views per gesture instance, trading classification loss against view/acquisition cost. The current data package contains three separately recorded views, but the executor has no radar hardware or calibrated per-view acquisition/energy/latency measurements. A benchmark based only on model FLOPs would not establish a sensor acquisition-cost contribution.

## Primary-source review

1. Yang et al., *Instance-Wise Dynamic Sensor Selection for Human Activity Recognition*, AAAI 2020, DOI 10.1609/aaai.v34i01.5461. The official paper page describes selecting a sparse sensor subset for each instance while jointly minimizing activity loss and sensor count; it uses an MDP policy, imitation learning, and Mutual DAgger, evaluated on three real HAR datasets. This is general HAR rather than radar gesture sensing, but it preempts a broad claim that per-instance sensor selection for accuracy-versus-sensor-cost is itself new. Source: https://ojs.aaai.org/index.php/AAAI/article/view/5461

2. Xue, Yang & Zhu, *Neuromorphic-inspired multi-view global-local fusion for IR-UWB radar dynamic gesture recognition*, Frontiers in Neuroscience 2026, DOI 10.3389/fnins.2026.1864448. The primary full text uses the public 12-class UWB-Gestures benchmark, combines three viewpoints, and claims adaptive fusion/feature selection. Its reported protocol is not verified as comparable to this project's held-user macro protocol. It directly preempts generic adaptive three-view feature fusion on this corpus. Source: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1864448/full

3. Yang et al., *Sparsity aware dynamic gesture recognition using radar sensors with angular diversity*, IET Radar, Sonar & Navigation 2018, DOI 10.1049/iet-rsn.2018.5206. The full article studies simultaneous two-radar angular-diversity gesture observations, sparse time-frequency features and feature fusion; it reports improved robustness when one viewpoint is poor and discusses orthogonal lines of sight. It is not the X4 corpus or the same sensor type, but closes generic angular-diversity fusion/poor-view robustness as a standalone claim. Source: https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-rsn.2018.5206

## Decision

- The broad per-instance view-selection candidate is **not cleared as a research contribution**. It overlaps established sensor-selection objectives and same-corpus adaptive multi-view feature selection.
- No classification experiment was run. No project labels, tensor values, or checkpoint contents were read; no model fit, data download, or training was performed.
- A narrower future study would need a technically material, radar-signal-specific selection rule and a meaningful measured cost/quality target, distinguished from the above methods. The current cache and CPU-only executor do not provide calibrated per-view acquisition cost, and the project's same-corpus multi-view novelty gate is already closed.
- This is a bounded literature screen, not an exhaustive review or legal opinion. The branch's 789-file original source snapshot remains untouched.

## Current executor preflight

At 2026-10-12 13:08 UTC, the managed executor was running with enforced restricted egress, no custom egress hosts, no outbound identity or additional capabilities. The restored path `/workspace/uwb-restored-20260930` exists. Runtime: Python 3.12.14, PyTorch 2.14.0+cpu, 5 CPUs, CUDA unavailable, 746,266,624 bytes free. Process scan found only three historical defunct Python entries and no active training. The restored cache was not re-downloaded.
