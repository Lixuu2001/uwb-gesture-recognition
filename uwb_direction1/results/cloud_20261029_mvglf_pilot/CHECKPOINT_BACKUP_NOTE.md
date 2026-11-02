# Immutable partial checkpoint backup — 2026-10-29 UTC

The original MV-GLFNet-style source pilot continues unchanged in the cloud. At 12:14 UTC worker42028 and supervisor42027 were alive, full arm epoch13; the local CNN remains queued in the same worker. No duplicate worker, new seed, new fitting, diagnostic scoring, or scientific amendment was introduced.

An immutable full-arm epoch11 snapshot is now backed up in the existing private Drive folder `12AGMwT99yXCTYBQocQaX901Br-IelmOf`. The selected validation checkpoint in this snapshot is also epoch11. This is a recovery artifact, not an accuracy result. Both arms must finish before diagnostic comparison and the frozen continuation gate.

## Verified recovery evidence

- 38 archived members: exact frozen code/protocol/identity, source IDs manifest, role split, source input/prepared-map hashes, epoch log, initial/first-update/best/last checkpoints, optimizer and Python/NumPy/Torch RNG state, audit code and prior receipts.
- Compressed archive: 133,533,466 bytes; SHA-256 `da0e3870b7f84667e5f01f18871e79604c97d2de6835ad4c5c304056523ab342`.
- 19 chunks of at most 7 MiB. Every uploaded object was checked for size, target parent, and owner-only permission metadata, then fetched with `download_raw_file=true, include_base64=false`, immediately materialized via its Sediment file ID, and SHA-256 checked. Maximum concurrent pipelines:3.
- The materialized remote chunks were reassembled; the complete archive SHA and all38 member SHA/byte counts match. Receipt verification time: 2026-10-29T12:14:40.634677+00:00.
- A standalone protocol backup also passed authenticated readback and its original frozen SHA. No sharing permissions were changed.
- Raw radar arrays and prepared map arrays are excluded. Weights, optimizer states and sample IDs stay private in Drive/cloud, outside Git. Git contains only the recovery index, helper code, hashes and this note.
- The original789-file source subtree remains unchanged.

The checkpoint snapshot is fixed even as the worker atomically replaces its live files. The live STATE captured in the archive can be partway into the following epoch; recovery from epoch11 charges already-consumed partial work against the original cumulative budget. It must not claim that partial optimizer work was saved. Use `full/last.pt`, never `first_update.pt`, for an explicitly diagnosed resume, and never resume alongside a living original worker.

## Recovery and continuation

Persistent Drive IDs, chunk order, hashes, and all member hashes are in [DRIVE_BACKUP_RECEIPT.json](DRIVE_BACKUP_RECEIPT.json). Local immutable bundle: `/workspace/uwb-mvglf-20261029/backup_20261029_1210`. Fetch each ID through the authenticated connector after metadata checks, materialize immediately, verify chunk hashes, concatenate in numeric order, and verify the archive/member hashes before restoring to a fresh directory. Do not use expiring signed URLs. The separately verified original input package remains at `/workspace/uwb-restored-20261020`; the frozen runner can regenerate source maps from it. Retain the recorded source-manifest path or explicitly document a hash-preserving engineering path amendment.

Continue the existing full24epoch + local24epoch queue. When SUMMARY/STATEcomplete and both ARM_COMPLETE receipts exist, run the independent completed-run audit to replay validation selection and diagnostic predictions before applying the preregistered gate. Full replay is not yet executed. Diagnostic labels remain unread by this backup; users7–8 labels remain unused. Selection-side validation is not evidence of diagnostic improvement.

The latest user authorizes ongoing published-method accuracy experiments independently of originality claims. A completed pilot may justify a separately frozen source-fold comparison; no eightfold, independent-generalization or original-contribution claim is made here.
