# Cloud-only continuation / restore

Local data: /workspace/uwb-restored-20260930. Prepared signals: /workspace/uwb-circular-prepared-20260930. Training root: /workspace/uwb-temporal-scale-runs-20261003. Before continuing inspect queue_status.json and /proc PID status (Z is exited). Never launch duplicate jobs.

Source/protocol hashes must match identity.json and COST.json exactly. Resume uses same train command in LAUNCH.json: each completed result is skipped only after identity/artifact checks; incomplete trajectory restores model, AdamW, scheduler, RNG, best validation state and dense tail, then starts the next epoch. Preserve all existing files; no overwrite from another configuration.

Core training inputs and old six baseline weights already have independent private receipts. Baselines: results/cloud_20261003_swad/DRIVE_BACKUP_RECEIPT.json (if named differently use RUN_REPORT), and results/cloud_20261003_view_sharing/BASELINE_STATE_DRIVE_RECEIPT.json. Newly completed trajectories are packed by archive_temporal_scale.py; exact relative paths preserved under empty restore root. source_orders.npz contains every training epoch's ordered sample IDs. No original large data arrays go to Git.

Only START_DRIVE_RECEIPT.json verifies an initial checkpoint snapshot; it does not imply full-run backup. Final archive receipt is local verification until each object is uploaded privately, authenticated-redownloaded and SHA checked, with durable DRIVE_BACKUP_RECEIPT.json. Never report an unverified backup as complete.
