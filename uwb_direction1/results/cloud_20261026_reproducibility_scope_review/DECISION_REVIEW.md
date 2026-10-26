# Research-scope decision review: method search vs reproducibility audit

**Date:** 2026-10-26 03:05 UTC  
**Decision needed:** whether to continue method-first screening for the original novel-method goal, or formally narrow the paper to an empirical reproducibility / protocol audit of the existing UWB-Gestures corpus. This review is not an experiment and does not claim either scope is a validated contribution.

## Verified evidence

The latest status and reports record the following:

- The original eight-fold cached LTR result replays at 93.4417% mean participant-level macro accuracy and 75.00% worst participant. This is a replay of saved historical predictions, not a fresh end-to-end training replication.
- The corpus contains 9,593 records, with 9,300 retained after the recorded quality mask. The documented exclusions are 200 HV2 class-10/11 records and 93 HV6 class-3 records. The source manifest and all 789 original project files remain available and unchanged.
- The circular-feature experiment's final scores were legacy 94.1843%, angle control 94.1581%, and circular 94.4492%, but its continuation gate failed. That result does not validate an original method.
- The L/T privileged subgroup Stage A passed its coverage-qualified direction criterion for only 1/12 classes against a frozen requirement of 8/12; the candidate was closed. The follow-on ontology-factorization and generic topological-map candidates did not clear novelty. No Stage B or CNN run is authorized.
- All eight participants have prior project-development exposure. A held-participant fold on this same cohort is therefore not an untouched independent validation. The Figshare derivative is the same corpus; other public radar cohorts currently lack a verified class map/access path for a matched external test.

## Scope A — continue searching for a distinct method on current data

This preserves the original goal, but the available inputs are limited to clutter-removed real-valued three-view range-time maps and participant/class metadata. The package does not contain phase-coherent raw echoes, per-view background captures, calibrated sensor geometry/cost, synchronized frame/hand-pose truth, or independent participants. The already screened routes include generic fusion, subgroup/group robustness, calibration/normalization, speed/trajectory, temporal reversal, label-ontology factorization, and generic topology summaries.

A new method candidate would need to state a concrete signal mechanism not reducible to another feature family, be observable in these exact inputs, and pass a prior-art gate before any fitting. No such candidate is currently on record. Continued broad ideation without that gate risks accumulating uninformative negative probes; this route is not presently ready for training.

## Scope B — reproducibility and protocol audit using existing artifacts

This is feasible without new data or model training. A publishable audit would need to be framed as an empirical assessment of what the available artifacts actually reproduce, not as a new recognition algorithm or untouched generalization study. Minimum deliverables:

1. Freeze and publish a manifest of the 9,300 retained records, the 293 quality exclusions, source-code revision, and the original 789-file snapshot hash.
2. Reconcile the source scripts, saved split IDs, cached predictions, and scoring implementation; independently recompute the eight participant-level confusion matrices and macro accuracies from saved predictions.
3. Separate “historical prediction replay” from “fresh baseline reproduction.” Only the former is currently complete. If fresh CPU-capable baselines are needed, use one frozen protocol with all selection confined to source/validation participants and no post-hoc grids.
4. Report every participant, worst-participant performance, class support, and uncertainty at participant level; do not treat correlated trials as independent replicates.
5. Put the Stage A, circular, and other preregistered negative gates in the same ledger. Do not promote a failed candidate by dropping controls or changing the aggregation.
6. Limit the conclusion to this repeatedly developed eight-participant corpus. No claim of independent-subject validation, physical mechanism, or novel model follows from the audit alone.

The existing LTR replay and prior failed-gate ledger are strong engineering/reproducibility assets. They do not by themselves establish a new scientific contribution, and a reproducibility audit may not satisfy the original novel-method objective.

## Current execution / integrity state

Managed executor revision 1394 is running with restricted/enforced egress, five CPU cores, no configured credentials/capabilities, and no active UWB/training process. The restored-data receipt is present; the original source subtree contains 789 files. A workspace scan found no serialized model checkpoint files through depth 6. This review accessed only repository metadata and historical reports; it read no radar arrays or label values, ran no fitting/training, and changed no source snapshot or dataset.

## Recommendation

If the user's priority remains a genuinely new method, keep the contribution gate open but do not launch another model experiment without a specific, literature-cleared, observable mechanism; current reports do not support one. If the user accepts a narrower engineering-science outcome, proceed with Scope B as a protocol/reproducibility audit, explicitly acknowledging that it is not a novel method and cannot establish independent generalization. No training is started pending this scope choice.
