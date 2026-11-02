# Direction 1: mechanism-level archive review

Date: 2026-10-08. This is a source-archive inventory and design decision, not a literature-completeness claim or new experimental result. No labels, features, predictions, or model fitting were used for this review.

## What the source archive already covers

| Physical/statistical angle | Existing evidence in the project | Decision for continuation |
|---|---|---|
| One-, two-, and three-view recognition; view dropout, shared encoders, cross-view fusion and masked reconstruction | `original/project/proposal/开题报告.md` describes the planned multiview open-set architecture. Prior experiments and the archived 2026-09-28 report cover fixed subsets, dropout, calibration/reconstruction and budget policies. The report records `STOP_LOW_RANK_MECHANISM` and `STOP_VALUE_POLICY_MECHANISM`; it explicitly says neither failure rules out every adaptation/acquisition method. | Do not repackage ordinary fusion, missing-view completion, or the failed fixed value predictor as a new contribution. A changed acquisition condition would require new evidence, not just a new head. |
| Single-view few-shot user calibration | `original/project/experiments/p1_20260928/calibration_s2026/INDEPENDENT_AUDIT.md` independently replays the rank-2 user-factor calibration study: the proposed low-rank method lost 3.799774 pp versus the validation-selected simple method on uncalibrated classes and failed its fixed GO criteria. This is a narrow negative result, not evidence that all user calibration fails. | The generic “few labels adapt a new user” question is already tested. Do not reopen low-rank rank/regularization/support grids. |
| Signal transforms and hand-crafted temporal/range descriptors | The current screens cover range compression and range-frequency projections, range moments, temporal extent/reversal/scale, self-similarity, scattering, SSM descriptors, empty-gesture whitening/subspaces, source-selected subspace cancellation, and radar-exchange controls. Their frozen reports/gates are in `research/` and `results/`. | Stop adding another transform or pooled statistic to the same frozen readout without a different measured mechanism and a result that survives a destructive control. In particular, a descriptor score alone does not identify velocity, trajectory, or noise. |
| Loss/readout changes and source-training optimizer changes | The archived runs include ridge/logistic and semantic-code baselines, class-conditioned covariance, Huber, user weighting/deletion, gradient-variance/Fishr, SAM, SWAD, GroupDRO/regularization and response-pooling screens. The 2026-10-08 fixed loss/weight factorial also failed its preregistered primary gate. | Do not continue tuning readouts, losses, or optimizer variants as the main research route. These are known methods, not a physical contribution. |
| Geometric trajectory from multiple radar stations | `original/project/proposal_v2/29_导航化重切_定位跟踪辨识主线_GPT轨迹蒸馏方案判决与新三点.md` records Viterbi range tracking and spline reconstruction in the project, and cites three-radar gesture/air-writing trajectory prior art (Arsalan & Santra 2019; Leem et al. 2019). The dataset has no independently measured hand-trajectory ground truth. | Do not claim “three radars recover a hand trajectory” as new, and do not score geometry against a proxy as though it were ground truth. |

## Data boundary relevant to novelty

The restored collection contains 8 development users, 12 gesture labels, three fixed radar views, and 9,300 quality-retained instances. The original project notes that the capture has no independent 3-D hand-path truth and does not establish cross-day, arbitrary-placement, or natural-stream deployment. The same eight users have been repeatedly used for historical model selection. Thus another positive eightfold number can support reproducibility on this collection, but cannot by itself establish external-user or cross-condition contribution.

## Decision

The archive review found no untested, clearly identifiable physical measurement in the present records that both (a) has direct labels/ground truth available and (b) is not already represented by a prior proposal or an archived negative screen. No new experiment is justified from this inventory alone. The next candidate must start from a genuinely new observable or deployment condition, with a direct-paper check and controls specified before labels are inspected. Examples that would change the evidence are an independently measured trajectory/pose, a controlled radar-placement or obstruction condition, or a new sensor/subject cohort; they are not present in the current archive as verified labels.

If continuation remains limited to this fixed dataset, the next step is a targeted direct-prior-art review of a sharply stated physical observable, followed by a source-only feasibility test with no test-user scoring. If no falsifiable/identifiable signal survives that review, record the limitation rather than launching another broad feature sweep. No new training queue was started by this review.

## Sources checked

- Original project proposal: `original/project/proposal/开题报告.md` (multiview open-set, sensor failure and masked reconstruction scope).
- Original calibration/budget report: `original/project/experiments/p1_20260928/report_s2026_release_v2/REPORT.md` (fixed protocol, both STOP outcomes, timing/evaluation limits).
- Independent calibration audit: `original/project/experiments/p1_20260928/calibration_s2026/INDEPENDENT_AUDIT.md` (all fold/support IDs and stop result).
- Prior-art and scope note: `original/project/proposal_v2/29_导航化重切_定位跟踪辨识主线_GPT轨迹蒸馏方案判决与新三点.md` (trajectory pipeline and cited three-radar trajectory prior art).
- Dataset description cited by the source proposal: [UWB-Gestures, Scientific Data (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8041886/).

This is a bounded review of the preserved project materials. It does not assert that the entire radar-gesture literature has been searched or that all possible contributions are impossible.
