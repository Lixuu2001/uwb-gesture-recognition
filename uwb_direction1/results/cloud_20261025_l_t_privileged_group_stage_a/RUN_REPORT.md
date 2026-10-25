# L/T-derived class-conditional subgroups for Right-only learning — Stage A design

**Run status:** not run. This file records a frozen diagnostic design and executable code, not an experimental result. Date: 2026-10-25 UTC.

## Current verified progress

- Latest private-branch head reviewed: `bf357a5be1dbd2df8761978cbaa305fbe82e98fc` (Oct 24, 2026); latest prior-art report is `results/cloud_20261024_physical_normalization_prior_art_gate/RUN_REPORT.md`.
- Dataset restore is still present at `/workspace/uwb-restored-20261020`, previously SHA-verified (86 files; 2,141,981,100 bytes). No migration replay or Drive access was performed this run.
- Process scan found no active UWB/training job. A recursive workspace filename scan found no model checkpoint. P1 multi-view prediction artifacts are present, but only their ZIP member names and NumPy headers were inspected. They contain `features (9300,3,256)` and `logits (9300,7,12)` for the seven view subsets; they are not out-of-source Right-only logits for users 7–8 from a model fitted only on users 3–6. Their payload values were not read.
- The exact L/T-derived, within-class subgroup transfer test is not recorded in the latest branch or run reports. Existing Group Gap/GroupDRO runs use user-level groups; P1 tests whole-view teacher logits/OOF soft targets, not L/T-derived group labels.
- The restored cache also contains the historical source-only/LOSO P1 analysis, but no suitable source-only Right CNN checkpoint or per-sample predictions for the proposed inner split are available. Stage A therefore uses one fixed linear Right-only probe and explicitly treats its errors as a proxy, not as the historical CNN failure mode.
- No real map values, label values, logits, or checkpoint weights were read. No fit or training was started.

## Frozen diagnostic

Use only source users 3–8. Seal outer test user 1 and validation user 2. Fit all group-discovery transforms and one fixed Right-only Ridge probe on users 3–6; apply them once to inner source audit users 7–8. Exclude only the established quality-conflict rows among the source split (HV6/class 3). Group candidates are classwise K=2 from fixed pooled magnitude maps: L/T combined versus matched Right-only. The candidate control also reports error information by audit-user identity and a count-preserving within `(class,user,R-group)` permutation null.

The one primary quantity is `I(Right-probe error; L/T group | class,user,R-group)`. The decision rule is frozen in `protocol.json`: minimum group support in 8/12 classes; same ≥5-pp high-error group direction in both source audit users in at least 8/12 classes; incremental conditional MI ≥0.005 bits/sample and one global permutation p≤0.05. All gates must pass. Failure closes the candidate. A pass only justifies a separately authorized, matched CNN Stage B; it is not a contribution and does not establish kinematics.

## Prior-art boundary

GEORGE, Domino, EIIL, and AGRO already establish hidden-subclass clustering, cross-modal error-slice discovery, inferred environments, and adversarial group discovery/robust optimization. LUPI already establishes training-only privileged information. Same-corpus Xue et al. 2026 already studies the L/T/R views and generic fusion. The new question is limited to the empirical, testable role of L/T-derived class-conditional groups as *training-only* supervision for Right-only deployment. None of the above algorithms or their simple combination is claimed as novel. Bounded prior-art search did not identify an exact same-corpus experiment, but novelty is **not cleared**.

Primary references and the design code are in `DESIGN.md`, `protocol.json`, and `diagnostic.py`.

## Budget and instruction boundary

Current executor: 5 CPUs, ~17.7 GB available RAM, ~25.8 GB workspace free, no GPU/PyTorch; restricted network. Stage A is capped at 900 CPU-seconds / 15 minutes wall, single process and one BLAS thread, with one pass over the source rows. The implementation requires `--confirm-stage-a`; no diagnostic command was run because the user requested this design be presented first and the diagnostic held for the next instruction. Only syntax compilation and CLI help were checked; no tests were added or run.

## Decision

Keep the candidate unresolved. It is feasible as a low-cost source-only screen and is materially different from prior user-group and whole-view-KD experiments, but the proxy cannot establish the historical CNN's Right-failure structure. Wait for an explicit next-session instruction before executing Stage A or training any CNN.
