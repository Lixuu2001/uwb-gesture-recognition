# Source-archive candidate reconciliation

Date: 2026-10-12 13:55 UTC  
Branch head before this record: `b0a78ec8e48dfaef047e719f906b83c950de7ae7`  
Purpose: reconcile the original 2026-09-28 route proposal against later archived experiments and the current continuation gate. This is a read-only source-record audit; it does not claim a new method or efficacy result.

## Finding

The original proposal `original/project/proposal_v2/31_第一部分新路线_用户校准与多雷达预算识别_调研方案.md` proposed two candidates as not yet tested **at the time it was written**: (A) few-shot user calibration from three demonstrated classes, and (B) fixed multi-radar processing-budget recognition. The proposal's historical status is not the current status. Later primary records already ran and closed both candidate families:

- **A, low-rank few-shot user calibration:** the later independent audit at `original/project/experiments/p1_20260928/calibration_s2026/INDEPENDENT_AUDIT.md` reproduced the frozen stop decision. Rank-2 calibration was **3.799774 percentage points below** the validation-selected simple calibration on uncalibrated classes; only 1/8 users improved. All five preregistered GO conditions failed. The audit limits this conclusion to this implemented rank-2 mechanism, eight-user protocol and one encoder seed; it does not refute all personalization.
- **B, multi-radar budget/value policy:** the later pilot report at `original/project/experiments/p1_20260928/report_s2026_release_v2/REPORT.md` already evaluated fixed subsets, cascades and a value predictor. The ExtraTrees value policy scored **90.863258%** at **2.007873** mean processed views, versus **91.920833%** at **2.004616** views for the main confidence/entropy cascade; it was **2.578409 pp below** fixed three-view processing. It failed the STOP_VALUE_POLICY_MECHANISM gate. This is not just an unrun idea awaiting a classifier.
- The fixed-view family also has same-corpus adaptive multi-view fusion prior art in the 2026-10-12 adaptive-view review. Cost claims are additionally limited: the current package has no calibrated per-view acquisition/energy measurements, so offline processing count cannot establish radar hardware power savings.

These archived values are reported as recorded, not recomputed from sample arrays in this review. Do not reopen the rank, support-count, regularization, threshold or value-predictor grids, and do not rename a stopped route as a new method.

## Source chain and integrity

The source documents were read from the current private branch and kept unmodified:

| Record | Blob SHA |
|---|---|
| Proposal 31 (historical candidate plan) | `846e6e33b6449cbec1af2e7f493a714c345a7101` |
| Mechanism gap map (2026-10-08) | `91de32e1552814f61ec9094e2643cfe49c37e4e9` |
| Next-experiment gate (2026-10-09) | `9929497e703cac21311bd55f8021443a91a0ba1e` |
| Radar phase-neighbor inventory | `6c1287274214276e764579ac08ee0cf0134329ec` |
| Calibration independent audit | `49cb21e812254e118924a39631a985056a706459` |
| Calibration/budget pilot report | `68f5d016e8a5bb9a2d426b2d9f8c789968e5caee` |
| Adaptive-view prior-art report | `3df4db5bb9758a044988c1224f01c9ae9be8b373` |

The reconciliation matches the current STATUS and memory entries: candidate novelty/evaluation remain unresolved; no contribution has been validated. It does not alter any of the 789 original source-snapshot files.

## Execution boundary

A fresh cloud preflight found the restored cache present at `/workspace/uwb-restored-20260930`, 5 CPU cores, Python 3.12.14, PyTorch 2.14.0+cpu, no CUDA, about 712 MiB free on the 32 GiB workspace, and no active training process (only historical defunct Python entries). No restore/download was needed. This review read project reports and metadata only: no labels, tensor values, checkpoint contents or raw arrays were read; no fitting or training was run.

The current continuation record says the processed cache lacks raw complex echoes, per-view backgrounds, calibrated transfer data and synchronized motion truth. Thus coherent-phase/transfer-function mechanisms, physical trajectory claims, and measured radar acquisition-power savings are not testable from the present package.

## Decision and next gate

The archived proposal does not supply an untried route. Keep the research goal open, but do not launch another classifier/readout grid from these two candidates. A future run requires, before any fit:

1. one sharply specified signal-level observable/mechanism;
2. a direct primary-source distinction from the existing IR-UWB, radar-gesture and same-corpus work;
3. proof that the required input exists in the restored cache, or a documented new acquisition with raw/calibrated/synchronized measurements;
4. a frozen user-disjoint protocol, destructive controls and bounded CPU/storage estimate.

If no candidate clears all four gates with current inputs, the real blocker is missing measurement/input evidence, not a need to repeat the completed migration or ask the user to move files manually. Contribution remains unvalidated.
