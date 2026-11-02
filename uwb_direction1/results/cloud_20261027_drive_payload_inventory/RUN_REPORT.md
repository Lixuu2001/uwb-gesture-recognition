# Drive payload and current-input inventory

**Date:** 2026-10-27 07:48 UTC  
**Branch parent:** c35303faa0092037f596df45b9f333bbad50a5d9  
**Scope:** one private Drive transfer manifest, file metadata, and local NPY headers only. No experiment archives were fetched.

## Verified payload evidence

- The requested Drive folder metadata still reports owner-only sharing (`shared=false`); the transfer manifest file metadata also reports `shared=false`. No sharing or permission changes were made.
- The single fetched object was `TRANSFER_MANIFEST.json` (56,367 bytes). It was fetched with `download_raw_file=true`, `include_base64=false`, materialized immediately, and SHA-256 verified as `10516baf0b49a84e6a39169ef447fffcdcb29defa2a6cc79e301dc938ddceb4d`. This matches the SHA-256 recorded for the archived transfer manifest in `results/cloud_20260930/RUN_REPORT.md`.
- That archived manifest is a historical transfer artifact and is not byte-identical to the later-edited migration manifest in the current branch/source snapshot. The verification above is anchored to the hash recorded in the original restore report; it does not claim the two manifest versions are identical.
- The transfer manifest records a complete 86-file payload totaling 2,141,981,100 original bytes. Its array inventory includes `cache/X_clutterremoved.npy`, `cache/subj_raw.npy`, and `cache/y_raw.npy`; it contains no raw complex echo tensor entry.
- In the restored package, NPY headers (not values) confirm `X_clutterremoved` has shape `[9593,3,90,189]`, dtype `float32`; `subj_raw` and `y_raw` are length-9593 `int64` index/label arrays. Filename matches alone were not used to infer tensor contents.
- The 803-item Drive folder also contains opaque archive parts. Their contents were not fetched, so this audit does **not** rule out a separately stored raw-signal archive under an opaque name. No such archive was downloaded or restored in this run.

## Runtime and research disposition

- Current managed executor: spec revision 1437, running; 5 CPU cores; NumPy/SciPy/scikit-learn available; PyTorch unavailable.
- The restored dataset and immutable 789-file source snapshot are present. Process scan found no active UWB/training process; the inspected workspace scope contained no serialized model checkpoint.
- No array values, label values, model outputs, or checkpoint contents were read. No fitting, training, or model evaluation ran.
- The original method goal remains unresolved. Current restored inputs support work on clutter-removed real-valued range-time maps; methods requiring raw complex phase or calibrated synchronized kinematics are not executable from this 86-file payload. No failed candidate was reopened.

## Next bounded action

Continue method-first literature screening only for a signal mechanism directly observable in the current clutter-removed maps and materially distinct from the already closed routes. Do not fetch unrelated opaque archives or start training without a frozen candidate that clears the literature and observability gates.
