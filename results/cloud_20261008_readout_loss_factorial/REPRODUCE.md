# Reproduction

The private archive contains `source_run/` with all 42 fold artifacts and their SHA manifest, plus `independent_audit/AUDIT.json`, the logs, cost screen, and frozen protocol. The repository contains the independent replay implementation at `uwb_direction1/tools/audit_loss_factorial.py` and its dependencies `audit_restored.py` and `second_order_core.py`.

From the repository root, with Python 3.12, NumPy 2.3.5, scikit-learn 1.8.0, SciPy 1.17.0 and threadpoolctl installed, and the protocol-pinned source data/feature references at their recorded cloud paths, run:

```sh
python uwb_direction1/tools/audit_loss_factorial.py \
  --run /path/to/unpacked/uwb-readout-factorial-backup-20261008/source_run \
  --out /path/to/new/audit-output
```

The output path must not already exist. The script checks the protocol, source-run files, frozen-feature manifests, sample IDs, fold roles and protocol-bound implementation hashes, then refits and compares all 96 candidate models. It writes `AUDIT.json`. Held diagnostic labels are used only after the fixed validation selections and predictions have been persisted by the source run.

The original source runner is preserved byte-for-byte by its SHA in `IDENTITY.json` and by Git commit `0f6f10a8659726fd2216210297d4cd6ad522ae01`. Its final summary step had a gate-key typo; the independent replay is the audited result. Do not change the protocol, retune, or rerun this closed candidate to pursue a better number.
