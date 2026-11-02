# Runtime restoration and same-corpus prior-art gate

Date: 2026-10-20 UTC  
Remote branch head reviewed: `5b811103daa2b3e546ee591c971547d94779bb31`  
Report scope: restore the already-migrated private dataset into the replacement execution environment, audit it, and review the directly relevant 2026 primary paper. No new model was fitted and no training was launched.

## Runtime replacement and restore

The current executor did not contain the previously used `/workspace/uwb-restored-20260930` directory or the project checkout. The branch snapshot was fetched from the current remote head into a separate detached worktree at `/workspace/uwb-cloud-current`; the existing checkout was left on its original `work` branch. The 789-file original source subtree remains the tracked `uwb_direction1/original/project` snapshot.

The private Google Drive folder `12AGMwT99yXCTYBQocQaX901Br-IelmOf` metadata reported `shared=false`, owner-only permission metadata. No sharing settings were changed. Recovery followed `uwb_direction1/migration/20260930/CLOUD_DOWNLOAD_INDEX.json`: 163 compressed data chunks and the Drive copy of `TRANSFER_MANIFEST.json` were fetched with `download_raw_file=true`, `include_base64=false`, at no more than three concurrent fetches. Each streamed chunk was immediately materialized, size checked, and SHA-256 checked against the persistent index.

- Drive transfer manifest SHA-256: `10516baf0b49a84e6a39169ef447fffcdcb29defa2a6cc79e301dc938ddceb4d`.
- Original-source manifest SHA-256: `8f19d50aacdcc2cfe3457e9dc4785a0cf440ef76c5777a751806b5575a9f2884`.
- `transfer.py verify` passed for the complete package.
- `transfer.py restore` completed to the new, non-overwriting path `/workspace/uwb-restored-20261020`: 86 files, 2,141,981,100 bytes, all original-file hashes passed.
- `audit_restored.py` independently passed source-file hashes and replayed all eight cached held-user LTR folds. The replayed historical result is mean user-macro accuracy 93.4417%, worst held user 75.00%. This is an integrity/replay check, not a new model result.

## Primary-source prior-art result

The full text of Xue, Yang & Zhu, “Neuromorphic-inspired multi-view global-local fusion for IR-UWB radar dynamic gesture recognition,” *Frontiers in Neuroscience* 20 (2026), DOI [10.3389/fnins.2026.1864448](https://doi.org/10.3389/fnins.2026.1864448), was reviewed directly.

It explicitly evaluates the public Ahmed et al. UWB-Gestures corpus with XeThru X4 IR-UWB radar, 12 gesture classes, Left/Top/Right views, 20-fps maps, and 90×189 TR preprocessing. It reports 9,600 samples from eight subjects and a subject-disjoint 6-subject train / 2-subject test split. Its method uses a local ConvMixer branch, a global CNN–Transformer branch, and gated first-order plus bilinear second-order branch fusion; the paper reports 98.29% for three-view fusion and studies single-/dual-/tri-view settings.

This directly preempts generic same-corpus claims around TR maps, global/local feature fusion, adaptive feature fusion, bilinear branch interactions, and the basic benefit of three-view fusion. Its published 6/2 split and 9,600-sample count do not match the current quality-filtered 9,300-sample, eight held-user-fold protocol, so the reported 98.29% is not a directly comparable score. It is not independent validation.

A separate primary metadata/abstract review also confirms that Eom, Jeon & Jeong (IEEE IoT Journal 12(13), 2025, DOI [10.1109/JIOT.2025.3556901](https://doi.org/10.1109/JIOT.2025.3556901)) already describe UWB impulse-radar open-set gesture recognition with temporal/spatial transformer features and one-versus-rest deep inspectors. Open-set rejection is therefore not a cleared standalone novelty claim either.

## Runtime capabilities and decision

Current executor: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, 5 CPUs, about 17.9 GB available RAM and 25.8 GB free filesystem space. PyTorch is not installed. No training process was found. The recovered dataset is the existing clutter-removed, real-valued three-view range-time tensor; no new complex raw echoes, calibrated per-view costs, acquisition hardware, or synchronized motion ground truth were added.

No candidate passed a new mechanism, prior-art, and input-feasibility gate in this run, so no experiment was started. The original contribution goal remains unvalidated. The next candidate must identify a signal-level physical mechanism not covered by the same-corpus global/local and adaptive-fusion prior art, be observable in the available real-valued maps, and have a frozen user-disjoint test protocol before any fitting. If its physical claim requires phase/coherent Doppler, calibration, or motion truth, it is infeasible with this cache and needs new acquisition; do not proxy those missing measurements with uncalibrated image features.

## Reproduction paths

- Current restored data: `/workspace/uwb-restored-20261020`
- Current project/source worktree: `/workspace/uwb-cloud-current`
- Package verification: `python /workspace/uwb-cloud-current/uwb_direction1/tools/transfer.py verify --package /workspace/uwb-transfer`
- Restore audit: `python /workspace/uwb-cloud-current/uwb_direction1/tools/audit_restored.py --data /workspace/uwb-restored-20261020 --package /workspace/uwb-transfer --out /workspace/uwb-audit-20261020`

No raw arrays, model weights, or private checkpoint artifacts were uploaded to Git.
