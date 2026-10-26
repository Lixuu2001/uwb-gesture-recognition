# Multi-view teacher to single-view student candidate gate

**Date:** 2026-10-26 09:43 UTC  
**Decision:** close the generic privileged multi-view distillation candidate before fitting. This is a bounded prior-art screen, not a scientific negative or exhaustive novelty search.

## Candidate screened

Train a three-view teacher on the Left/Top/Right clutter-removed maps, then distill it into a one-view student for deployment with one radar. The hypothesized practical value is using the extra sensors only during training while retaining a single-sensor inference path.

## Prior art reviewed

- Xue, Yang & Zhu, “Neuromorphic-inspired multi-view global-local fusion for IR-UWB radar dynamic gesture recognition,” *Frontiers in Neuroscience* (2026), DOI [10.3389/fnins.2026.1864448](https://doi.org/10.3389/fnins.2026.1864448). Publisher full text reviewed. It uses the same UWB-Gestures corpus, and Table 6 reports Left, Top, Right, every two-view combination, and all three views. It does not describe teacher-student distillation (a bounded full-text search for “distill” returned no matches), but it preempts claiming single/dual/tri-view accuracy comparisons as the contribution.
- Zhang et al., “Joint MobileViT and Knowledge Distillation Network for Hand Gesture Recognition via mmWave Radar,” *IET Signal Processing* (2026), DOI [10.1049/sil2/9971257](https://doi.org/10.1049/sil2/9971257). Wiley full text reviewed. It uses a radar gesture teacher/student KD pipeline over RTM/DTM inputs; its experiment uses a different FMCW dataset and a random 4:1 sample split. Generic radar-gesture distillation and compact-student claims are therefore already established, although its protocol is not comparable to this project’s held-user evaluation.
- Tian, Sun & Tang, “Multi-view Teacher–Student Network,” *Neural Networks* 146 (2022), 69–84, DOI [10.1016/j.neunet.2021.11.002](https://doi.org/10.1016/j.neunet.2021.11.002). Abstract and bibliographic record reviewed, not full text. It directly combines multi-view learning and teacher-student distillation.
- A targeted search also returned cross-modal multi-view action-recognition distillation (including View-aware Cross-modal Distillation) and radar-camera distillation. These are adjacent precedents; their full papers were not reviewed here.

The exact combination “three X4 views available only during training, one X4 view at inference, under held-participant evaluation” was not confirmed in the bounded search. That gap is a task setting, not yet a distinct learning mechanism. Simply combining known multi-view distillation with the same-corpus view subset comparison would not satisfy the project’s method-contribution bar.

## Feasibility and evaluation constraints

The verified cloud package contains clutter-removed real-valued maps with shape [9593, 3, 90, 189], plus participant/class/repetition metadata. The loader supports same participant/gesture/repetition grouping across views under soft synchronization. It does not establish shared frame timing, phase calibration, or synchronized motion truth. Output-level distillation may not require frame alignment, but paired samples are still repeated recordings and generic view-level distillation does not itself explain a physical mechanism.

A valid effectiveness study would need a frozen participant-disjoint protocol: all teacher/student fitting on the six source users in each outer fold, validation on the predetermined next user, and the held user evaluated once after all choices. It would require matched controls for a one-view supervised student and the multi-view teacher. This report does not authorize that experiment because the method-level novelty gate did not clear.

## Runtime and data handling

The cloud executor is running at revision 1415: five CPU cores, PyTorch unavailable, restricted/enforced network, and no additional configured capabilities. The restored package has 86 payload files plus its receipt; the original-source subtree has 789 files. No active UWB/training process or serialized checkpoint was found. No sample arrays, label values, logits, or checkpoints were read; no data were downloaded and no fitting/training ran.

## Disposition

Do not implement or train generic multi-view-to-single-view KD on this corpus as a claimed new contribution. Keep the research goal unresolved and method-first. Reopen only if a materially different, physically justified transfer mechanism is supported by current inputs and clears a new literature review; otherwise do not relabel this task setting or a reproducibility audit as a new method.
