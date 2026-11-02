# Run03 amendment — range-time echo Betti source feasibility

**Status:** preregistered, not started. This is an engineering repair before any feature extraction, fit, prediction, or score.

## Correction and preserved attempts

Run01 stopped because its expected user-6 count was transcribed as 1,107; the archived `experiments/p1_20260905/AUDIT.md` states HV6 retains 1,100 after excluding 93 quality-flagged G4 records. Run02 corrected that per-user count, but retained a stale literal total-count guard of 6,907, so it stopped at the same assertion despite observing the correct 6,900 total. Both attempts read only allowed source labels 1–6 to apply the fixed quality mask; neither read map pixels, extracted features, fitted models, or produced scores. Both receipts remain available and neither result informed the decision rule.

Run03 changes only the stale code guard from 6,907 to 6,900. Per-user counts are 1,200 / 1,000 / 1,200 / 1,200 / 1,200 / 1,100. The science is unchanged: same data mask, participants 1–6, held-source-user folds, views, fixed max-pooling/normalization/quantiles, Betti and amplitude/shuffle controls, train-only scaling, sample/user balanced RidgeClassifier(alpha=1), and frozen GO/STOP gate. Participants 7–8 gesture labels remain unopened.

The runner's CPU soft-stop is 885 seconds, hard ceiling 895 seconds, and independent wall timeout 895 seconds. Earlier preflight attempts each exited in under two seconds wall before map access; cumulative compute is therefore bounded below the original 900-second authorization. The signal handler persists an incomplete state and failure receipt.

- Protocol SHA-256: `aae558f7c50c94993f9d383709ba5fdf0111fb1b382cf913efe63bf9a471a4d8`
- Runner SHA-256: `d78df4577005c4339612deabec5aea8d9302fb48c6ab984c537c6c7f604ba00c`
- Code syntax check passed; prior synthetic descriptor and signal-receipt checks passed without data access.
- No results are available at registration. One run03 execution is planned. Any incomplete exit is not a scientific negative.
