# Signal-input inventory gate: Raw versus clutter-removed data

**Observed:** 2026-10-26 18:57 UTC  
**Remote parent:** 17a3a32e0bb3c18b658f40e798372bdded7716c9  
**Decision:** no new model experiment. The Raw/ClutterRemoved interaction is an existing project result, not a newly cleared method; the current restored package does not contain the Raw signal tensor needed to reproduce that comparison.

## Bounded inventory performed

This was a metadata-only check. No array values, labels, logits, or checkpoints were read and no fitting or training ran.

- The verified restore is present at `/workspace/uwb-restored-20261020`: 86 payload files plus the verification receipt.
- Header inspection of NumPy files reports `cache/X_clutterremoved.npy`, shape `[9593, 3, 90, 189]`, dtype float32. The accompanying raw-named files are participant and label index arrays, not signal tensors.
- The restore contains no `X_raw.npy` or `X_raw.npz`; the bounded workspace filename search also found no correspondingly named Raw signal tensor. The existing real-valued clutter-removed maps remain the only full signal tensor identified here.
- The 789-file source snapshot is present and was not edited.

## Existing project evidence and limits

The immutable project `docs/RESULTS.md` already reports:

- §16: three-view Raw versus ClutterRemoved was near-tied (OSCR 95.53 ± 1.15 versus 95.14 ± 0.81).
- §20.4: single-view Raw under a random split reported OSCR 86.74 ± 1.29.
- §30.4: single-view held-user case 0 reported Raw 57.10 versus ClutterRemoved 68.38 OSCR. The document interprets the split-dependent reversal as participant/environment leakage in static clutter.

These are historical source-document results, not rerun or independently audited in this pass. They do not form a complete matched multi-view × preprocessing × held-user factorial, and the Raw input is unavailable in the current restore. Consequently, reopening this comparison would neither be a fresh finding nor a reproducible experiment here. It is closed as a method-contribution route for this executor.

## Environment and action

The managed cloud executor is running at spec revision 1437. The verified data restore and 789-file source snapshot are present. Process and checkpoint scans found no UWB training job and no common serialized model checkpoint. The executor has five CPUs; PyTorch is absent. No Drive object was fetched and the Drive folder was not modified.

No new method has passed the novelty/observability gate. Continue method-first only if a distinct physical signal mechanism is directly observable in the existing clutter-removed maps and is not preempted by the cited literature. Do not claim the existing Raw/ClutterRemoved observation as a new method or contribution.
