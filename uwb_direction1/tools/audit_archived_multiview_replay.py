#!/usr/bin/env python3
"""Read-only audit of an existing eight-fold prediction archive; no fitting."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


SOURCE = Path(
    "/workspace/uwb-restored-20261020/experiments/p1_20260928/multiview_s2026"
)
OUT = Path("/workspace/uwb-repro-audit-20261028")
FOLDS = range(1, 9)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    started = time.perf_counter()
    fold_rows = []
    checks = []
    artifact_hashes = {}
    all_heads = None

    for fold in FOLDS:
        root = SOURCE / f"fold{fold}"
        split_path = root / "split.json"
        result_path = root / "result.json"
        pred_path = root / "predictions.npz"
        split = load_json(split_path)
        result = load_json(result_path)
        for path in (split_path, result_path, pred_path):
            artifact_hashes[str(path.relative_to(SOURCE))] = sha256(path)

        # Access only IDs, role metadata, logits, and labels needed for replay.
        # The saved feature matrix is deliberately never loaded.
        with np.load(pred_path, allow_pickle=False) as z:
            ids = z["ids"]
            y = z["y"]
            subjects = z["subjects"]
            roles = z["role"]
            logits = z["logits"]
            train_ids = z["train_ids"]
            val_ids = z["val_ids"]
            test_ids = z["test_ids"]
            val_logits = z["val_logits"]
            val_y = z["val_y"]
            subset_masks = z["subset_masks"]

            heads = int(logits.shape[1])
            if all_heads is None:
                all_heads = heads
            elif heads != all_heads:
                checks.append({"fold": fold, "name": "head_count_constant", "pass": False})
            checks.append({
                "fold": fold,
                "name": "row_shapes",
                "pass": logits.shape[0] == len(ids) and y.shape == ids.shape and subjects.shape == ids.shape and roles.shape == ids.shape,
            })
            checks.append({"fold": fold, "name": "unique_ids", "pass": len(np.unique(ids)) == len(ids)})

            split_ids = split["ids"]
            groups = {
                "train": np.asarray(split_ids["train"], dtype=np.int64),
                "val": np.asarray(split_ids["val"], dtype=np.int64),
                "test": np.asarray(split_ids["test"], dtype=np.int64),
            }
            archive_groups = {
                "train": np.asarray(train_ids, dtype=np.int64),
                "val": np.asarray(val_ids, dtype=np.int64),
                "test": np.asarray(test_ids, dtype=np.int64),
            }
            group_masks = {k: np.isin(ids, v) for k, v in archive_groups.items()}

            for name in ("train", "val", "test"):
                same = set(groups[name].tolist()) == set(archive_groups[name].tolist())
                checks.append({"fold": fold, "name": f"{name}_ids_match_split", "pass": same})
                checks.append({
                    "fold": fold,
                    "name": f"{name}_count_matches_split_metadata",
                    "pass": len(archive_groups[name]) == len(groups[name]) and int(group_masks[name].sum()) == len(groups[name]),
                })

            disjoint = (
                not (set(archive_groups["train"]) & set(archive_groups["val"]))
                and not (set(archive_groups["train"]) & set(archive_groups["test"]))
                and not (set(archive_groups["val"]) & set(archive_groups["test"]))
            )
            checks.append({"fold": fold, "name": "split_disjoint", "pass": disjoint})
            checks.append({
                "fold": fold,
                "name": "split_union_matches_archive",
                "pass": sum(int(m.sum()) for m in group_masks.values()) == len(ids),
            })

            val_mask = group_masks["val"]
            test_mask = group_masks["test"]
            checks.append({
                "fold": fold,
                "name": "validation_logits_bitwise_replay",
                "pass": val_logits.shape == logits[val_mask].shape and np.array_equal(val_logits, logits[val_mask]),
            })
            checks.append({
                "fold": fold,
                "name": "validation_labels_match_archive",
                "pass": val_y.shape == y[val_mask].shape and np.array_equal(val_y, y[val_mask]),
            })

            test_user = int(result["test_subject"])
            val_user = int(result["validation_subject"])
            train_users = set(int(v) for v in result["training_subjects"])
            test_user_ok = set(np.unique(subjects[test_mask]).tolist()) == {test_user}
            val_user_ok = set(np.unique(subjects[val_mask]).tolist()) == {val_user}
            train_user_ok = set(np.unique(subjects[group_masks["train"]]).tolist()) == train_users
            checks.extend([
                {"fold": fold, "name": "test_subject_matches_split", "pass": test_user_ok},
                {"fold": fold, "name": "validation_subject_matches_split", "pass": val_user_ok},
                {"fold": fold, "name": "training_subjects_match_split", "pass": train_user_ok},
            ])

            # Role codes must be constant within each split and distinct across splits.
            role_by_group = {
                name: sorted(int(x) for x in np.unique(roles[mask]))
                for name, mask in group_masks.items()
            }
            role_integrity = all(len(v) == 1 for v in role_by_group.values()) and len({v[0] for v in role_by_group.values()}) == 3
            checks.append({"fold": fold, "name": "role_codes_partition_splits", "pass": role_integrity})

            pred = logits.argmax(axis=2)
            overall_acc = (pred[test_mask] == y[test_mask, None]).mean(axis=0)
            reported = np.asarray(result["test_accuracy_by_subset"], dtype=np.float64)
            metric_matches = reported.shape == overall_acc.shape and np.allclose(reported, overall_acc, rtol=0, atol=1e-12)
            checks.append({"fold": fold, "name": "reported_test_accuracy_replay", "pass": metric_matches})

            classes = np.unique(y[test_mask])
            class_acc = []
            for head in range(heads):
                per_class = []
                for cls in classes:
                    mask = test_mask & (y == cls)
                    per_class.append(float((pred[mask, head] == y[mask]).mean()))
                class_acc.append(float(np.mean(per_class)))

            fold_rows.append({
                "fold": fold,
                "test_subject": test_user,
                "validation_subject": val_user,
                "training_subjects": sorted(train_users),
                "sample_counts_by_split": {k: int(len(v)) for k, v in archive_groups.items()},
                "role_codes_by_split": role_by_group,
                "subset_masks": np.asarray(subset_masks).astype(int).tolist(),
                "test_accuracy_by_head": overall_acc.astype(float).tolist(),
                "test_macro_accuracy_by_head": class_acc,
                "test_class_count": int(len(classes)),
            })

    # Aggregate each fixed head over the eight held users, without choosing a head.
    macro_matrix = np.asarray([x["test_macro_accuracy_by_head"] for x in fold_rows])
    overall_matrix = np.asarray([x["test_accuracy_by_head"] for x in fold_rows])
    summary = {
        "audit_id": "archived_p1_20260928_multiview_outputs_20261028",
        "source_root": str(SOURCE),
        "folds": 8,
        "heads": int(all_heads or 0),
        "all_heads_reported": True,
        "no_fitting_or_selection": True,
        "fold_count": len(fold_rows),
        "checks_passed": sum(bool(x["pass"]) for x in checks),
        "checks_total": len(checks),
        "checks_failed": [x for x in checks if not x["pass"]],
        "mean_user_macro_accuracy_by_head": macro_matrix.mean(axis=0).tolist(),
        "mean_user_overall_accuracy_by_head": overall_matrix.mean(axis=0).tolist(),
        "per_user": fold_rows,
        "artifact_sha256": artifact_hashes,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": "Read-only replay of historical saved predictions; not a new model result, independent cohort, or novel-method validation.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "AUDIT.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "audit_id", "fold_count", "heads", "checks_passed", "checks_total",
        "checks_failed", "mean_user_macro_accuracy_by_head", "mean_user_overall_accuracy_by_head",
        "elapsed_seconds", "interpretation"
    )}, indent=2))
    return 0 if not summary["checks_failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
