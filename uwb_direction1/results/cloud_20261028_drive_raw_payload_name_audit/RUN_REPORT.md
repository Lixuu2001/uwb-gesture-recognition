# Drive metadata audit for a separate raw UWB signal archive

**Date:** 2026-10-28 UTC  
**Scope:** metadata only. No Drive file content, experiment arrays, labels, or checkpoints were fetched or read. No training, fitting, or permission change.

## Checks and observations

- Read metadata for the specified Drive folder `uwb-data-package-20260930` (ID omitted here): it is a folder with `shared=false` and owner-only permission metadata. No sharing change was made.
- Listed its 803 direct children as metadata. The visible names include the transfer chunks and prior experiment archives. No separately named UWB-Gestures/XeThru raw-complex echo archive was identifiable from those child names.
- Ran account-wide metadata-only document searches for `UWB raw signal`, `UWB complex radar`, `X_raw UWB`, `UWB-Gestures raw radar`, `XeThru X4 raw echo`, and `UWB gesture complex samples`. Results were unrelated UWB positioning papers or empty for gesture-specific terms; no candidate raw echo archive was identified.
- The verified restored directory remains present at `/workspace/uwb-restored-20261020`; the immutable original-source snapshot still has 789 files. Current process and common-checkpoint scans found no UWB/training process or checkpoint.

## Interpretation and limits

This closes only the bounded search for an **obviously named and indexed** separate raw-signal archive in the specified folder and the connected Drive search results. It does not prove that no differently named, unindexed, or inaccessible archive exists. The current verified 86-file payload remains the available study input; its transfer manifest identifies the real-valued clutter-removed maps and does not list a raw complex echo tensor.

No new method candidate follows from this metadata check. Existing method novelty/input-observability gates remain unresolved; no fit or training is authorized by this report. This is a data-availability audit, not a scientific negative or research contribution.
