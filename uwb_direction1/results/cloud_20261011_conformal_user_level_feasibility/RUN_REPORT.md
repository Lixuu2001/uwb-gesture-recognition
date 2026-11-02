# User-level conformal prediction feasibility gate

Date: 2026-10-11 (UTC)

## Candidate screened

Candidate: add participant-disjoint, set-valued prediction to the fixed UWB-Gestures evaluation and claim finite-sample reliability for a new user.

Decision: **not cleared as a contribution and not feasible for a nontrivial 90% user-level coverage guarantee with this eight-user cohort. No fit or training was run.**

## Prior-art check

The same-corpus STGPL paper already reports UWB-Gestures open-set recognition and cross-subject evaluation, using graph prototypes and gradient-reversal subject disentanglement (Yang et al., 2026, DOI 10.1088/1361-6501/ae7fab). Adding a generic rejection/set-output wrapper would need a sharper distinction and value beyond that work.

Adjacent primary prior art applies label-conditional conformal prediction to subject-out wearable IMU human-activity recognition under orientation, gain, time-warp, missing-axis, and cross-placement shifts (Lamaakal et al., *Scientific Reports* 16, 26262 (2026), DOI 10.1038/s41598-026-57375-8). That is not radar, but it means generic “conformal uncertainty for subject-out activity recognition” is also not by itself a cleared contribution. The targeted search did not identify an exact IR-UWB radar-gesture conformal paper; this bounded result is not an exhaustive novelty search.

## Finite-sample feasibility

The unit relevant to the deployment claim is a new participant, not each correlated recording/window. Under ordinary split conformal, a calibration set of (n) exchangeable units uses rank (k=\lceil(n+1)(1-\alpha)\rceil), with the augmented (+\infty) score when (k>n).

For nominal 90% coverage ((\alpha=0.1)), at least nine calibration participants are needed before this rank can be finite: (\lceil 0.9(n+1)\rceil\le n\) first holds at (n=9). In an outer leave-one-participant-out fold here, only seven participants remain for development. Reserving one participant for calibration leaves (n=1), so the 90% threshold is (+\infty) and predictions are vacuous. Treating many windows from those same participants as independent calibration units would not justify a new-participant exchangeability claim.

This is a protocol-size limitation, not evidence that conformal prediction is ineffective. A within-user calibration protocol could answer a different question, but does not support unseen-user guarantees and would overlap the subject-calibration assumptions already explored in adjacent HAR work.

## Decision and next action

- Do not implement a generic conformal wrapper or tune a threshold on held-out users.
- Keep the eight held-user folds for closed-set recognition; do not reinterpret their test labels as calibration data.
- A meaningful new-user coverage study requires an independently sampled participant cohort large enough for calibration and an untouched participant test cohort, plus a method distinction beyond generic OSR and conformal prediction.
- No labels, radar-array values, or checkpoints were read; no training process was active; training was not started.

## Sources

- Same-corpus UWB open-set/subject-generalization prior art: https://doi.org/10.1088/1361-6501/ae7fab
- Subject-out IMU HAR with label-conditional conformal prediction: https://doi.org/10.1038/s41598-026-57375-8
- Publisher full text for the IMU-HAR study: https://www.nature.com/articles/s41598-026-57375-8
