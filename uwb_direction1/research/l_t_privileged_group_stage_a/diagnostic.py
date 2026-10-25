#!/usr/bin/env python3
"""Frozen Stage-A screen for L/T-derived within-class strata.

This is an exploratory diagnostic, not a classifier contribution. It uses no
CNN, no outer-test/validation rows, and never treats the discovered groups as
kinematic labels. See protocol.json and DESIGN.md before execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


SEED = 20261025
SOURCE_USERS = (3, 4, 5, 6, 7, 8)
FIT_USERS = (3, 4, 5, 6)
AUDIT_USERS = (7, 8)
OUTER_TEST_USER = 1
OUTER_VALIDATION_USER = 2
MIN_GROUP_N = 10
MIN_SUPPORTED_CLASSES = 8
MIN_ERROR_GAP = 0.05
MIN_DIRECTION_CLASSES = 8
MIN_INCREMENTAL_CMI_BITS = 0.005
N_PERMUTATIONS = 1999
HARD_BUDGET_SECONDS = 900


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_feature(sample: np.ndarray) -> np.ndarray:
    """Exact fixed 10x21 average pooling used by the archived P1 proxy.

    Input is one (view,time,range) sample with 3 x 90 x 189 geometry. Each view
    is independently magnitude-normalized, then pooled to 210 values.
    """
    if sample.shape != (3, 90, 189):
        raise ValueError(f"expected (3,90,189), got {sample.shape}")
    z = np.abs(sample.astype(np.float32, copy=False))
    mu = z.mean(axis=(-2, -1), keepdims=True)
    sd = z.std(axis=(-2, -1), keepdims=True)
    z = (z - mu) / (sd + 1e-6)
    return z.reshape(3, 10, 9, 21, 9).mean(axis=(2, 4)).reshape(3, 210)


def cmi_bits(a: np.ndarray, b: np.ndarray, strata: list[np.ndarray]) -> float:
    """Empirical I(A;B|strata), in bits per audit example."""
    n = len(a)
    if n == 0:
        return 0.0
    total = 0.0
    keys = np.stack(strata, axis=1)
    for key in np.unique(keys, axis=0):
        m = np.all(keys == key, axis=1)
        na = a[m]
        nb = b[m]
        if len(na) == 0:
            continue
        _, ai = np.unique(na, return_inverse=True)
        _, bi = np.unique(nb, return_inverse=True)
        table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
        np.add.at(table, (ai, bi), 1)
        p = table / table.sum()
        pa = p.sum(axis=1, keepdims=True)
        pb = p.sum(axis=0, keepdims=True)
        den = pa @ pb
        nz = p > 0
        mi = float(np.sum(p[nz] * np.log2(p[nz] / den[nz])))
        total += len(na) / n * mi
    return total


def permute_within_strata(x: np.ndarray, strata: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    out = x.copy()
    keys = np.stack(strata, axis=1)
    for key in np.unique(keys, axis=0):
        idx = np.flatnonzero(np.all(keys == key, axis=1))
        out[idx] = x[idx][rng.permutation(len(idx))]
    return out


def classwise_groups(features: np.ndarray, y: np.ndarray, users: np.ndarray,
                     fit_mask: np.ndarray, audit_mask: np.ndarray, mode: str):
    """Fit classwise K=2 groups on fit users; transfer frozen centers to audit.

    Group IDs are canonicalized by the lower-to-higher first train-PCA
    coordinate. This establishes identity only within this fixed split/class.
    No groups are aligned across classes, users, or outer folds.
    """
    g_fit = np.full(len(y), -1, dtype=np.int8)
    g_audit = np.full(len(y), -1, dtype=np.int8)
    models = {}
    labels = np.unique(y[fit_mask])
    for cls in labels:
        tr = np.flatnonzero(fit_mask & (y == cls))
        te = np.flatnonzero(audit_mask & (y == cls))
        if len(tr) < 32 or len(te) == 0:
            raise ValueError(f"class {cls}: insufficient fit/audit rows ({len(tr)}/{len(te)})")

        if mode == "lt":
            raw = np.concatenate([features[:, 0], features[:, 1]], axis=1)
        elif mode == "right":
            raw = features[:, 2]
        else:
            raise ValueError(mode)

        scaler = StandardScaler().fit(raw[tr])
        z_tr = scaler.transform(raw[tr])
        z_te = scaler.transform(raw[te])
        pca = PCA(n_components=16, svd_solver="full")
        q_tr = pca.fit_transform(z_tr)
        # Canonicalize PCA sign using train-only loadings before ordering centers.
        for j in range(pca.components_.shape[0]):
            k = int(np.argmax(np.abs(pca.components_[j])))
            if pca.components_[j, k] < 0:
                pca.components_[j] *= -1
                q_tr[:, j] *= -1
        q_te = pca.transform(z_te)
        km = KMeans(n_clusters=2, init="k-means++", n_init=10,
                    random_state=SEED, algorithm="lloyd").fit(q_tr)
        raw_train_group = km.labels_.astype(np.int8)
        centers = km.cluster_centers_
        order = np.argsort(centers[:, 0], kind="stable")
        remap = np.empty(2, dtype=np.int8)
        remap[order] = np.array([0, 1], dtype=np.int8)
        g_fit[tr] = remap[raw_train_group]
        g_audit[te] = remap[km.predict(q_te).astype(np.int8)]
        models[int(cls)] = {"scaler": scaler, "pca": pca, "kmeans": km, "remap": remap}
    return g_fit, g_audit, models


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    cache = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = Path(__file__).with_name("protocol.json")

    x_path = cache / "X_clutterremoved.npy"
    y_path = cache / "y_clutterremoved.npy"
    u_path = cache / "subj_clutterremoved.npy"
    for p in (x_path, y_path, u_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    x = np.load(x_path, mmap_mode="r")
    y_mm = np.load(y_path, mmap_mode="r")
    u_mm = np.load(u_path, mmap_mode="r")
    if x.shape != (len(y_mm), 3, 90, 189) or len(u_mm) != len(y_mm):
        raise ValueError(f"cache schema mismatch: X={x.shape}, y={y_mm.shape}, subj={u_mm.shape}")

    # Read only participant metadata globally, then gesture labels only for source
    # users 3..8. No y/X rows for outer validation user 2 or test user 1 are touched.
    src_rows = np.flatnonzero(np.isin(np.asarray(u_mm), SOURCE_USERS))
    src_y_all = np.asarray(y_mm[src_rows], dtype=np.int64)
    src_u_all = np.asarray(u_mm[src_rows], dtype=np.int64)
    keep = ~((src_u_all == 6) & (src_y_all == 3))  # audited source quality exclusion
    rows = src_rows[keep]
    y = src_y_all[keep]
    users = src_u_all[keep]
    fit_mask = np.isin(users, FIT_USERS)
    audit_mask = np.isin(users, AUDIT_USERS)
    if set(np.unique(users)) != set(SOURCE_USERS):
        raise ValueError("expected exactly source users 3..8")
    if set(np.unique(y[fit_mask])) != set(range(12)):
        raise ValueError("fit users do not cover all 12 gesture classes")
    if set(np.unique(y[audit_mask])) != set(range(12)):
        raise ValueError("audit users do not cover all 12 gesture classes")

    # Only map rows for the six source users are read; outer test/validation data
    # are never materialized.
    features = np.empty((len(rows), 3, 210), dtype=np.float32)
    for j, row_id in enumerate(rows):
        features[j] = sample_feature(np.asarray(x[int(row_id)]))
        if (j + 1) % 256 == 0 and time.monotonic() - started > HARD_BUDGET_SECONDS:
            raise TimeoutError("Stage-A 900-second budget reached during source feature pass")

    # Fixed Right-only linear diagnostic probe (not the historical CNN).
    r_scaler = StandardScaler().fit(features[fit_mask, 2])
    r_fit = r_scaler.transform(features[fit_mask, 2])
    r_audit = r_scaler.transform(features[audit_mask, 2])
    probe = RidgeClassifier(alpha=1.0, class_weight="balanced")
    probe.fit(r_fit, y[fit_mask])
    right_pred = probe.predict(r_audit)
    e = (right_pred != y[audit_mask]).astype(np.int8)
    y_a = y[audit_mask]
    u_a = users[audit_mask]
    idx_a = np.flatnonzero(audit_mask)

    # Discover L/T and Right-only partitions with identical fixed machinery.
    g_lt_fit, g_lt_a_all, _ = classwise_groups(features, y, users, fit_mask, audit_mask, "lt")
    g_r_fit, g_r_a_all, _ = classwise_groups(features, y, users, fit_mask, audit_mask, "right")
    g_lt = g_lt_a_all[idx_a]
    g_r = g_r_a_all[idx_a]

    cmi_lt_yu = cmi_bits(e, g_lt, [y_a, u_a])
    cmi_r_yu = cmi_bits(e, g_r, [y_a, u_a])
    cmi_lt_incremental = cmi_bits(e, g_lt, [y_a, u_a, g_r])
    cmi_user_y = cmi_bits(e, u_a, [y_a])

    rng = np.random.default_rng(SEED)
    perm_strata = [y_a, u_a, g_r]
    null = np.empty(N_PERMUTATIONS, dtype=np.float64)
    for b in range(N_PERMUTATIONS):
        gp = permute_within_strata(g_lt, perm_strata, rng)
        null[b] = cmi_bits(e, gp, perm_strata)
        if (b + 1) % 100 == 0 and time.monotonic() - started > HARD_BUDGET_SECONDS:
            raise TimeoutError("Stage-A 900-second budget reached during frozen permutations")
    p_perm = float((1 + np.count_nonzero(null >= cmi_lt_incremental)) / (N_PERMUTATIONS + 1))

    coverage = []
    directions = []
    for cls in range(12):
        fit_counts = {str(u): [int(np.sum((g_lt_fit == group) & fit_mask & (y == cls) & (users == u)))
                               for group in (0, 1)] for u in FIT_USERS}
        audit_counts = {str(u): [int(np.sum((g_lt_a_all == group) & audit_mask & (y == cls) & (users == u)))
                                 for group in (0, 1)] for u in AUDIT_USERS}
        fit_rep = sum(all(v >= MIN_GROUP_N for v in fit_counts[str(u)]) for u in FIT_USERS)
        audit_ok = all(all(v >= MIN_GROUP_N for v in audit_counts[str(u)]) for u in AUDIT_USERS)
        supported = fit_rep >= 3 and audit_ok
        coverage.append({"class": cls, "fit_group_counts_by_user": fit_counts,
                         "audit_group_counts_by_user": audit_counts,
                         "fit_users_with_both_groups_n_ge_10": fit_rep,
                         "supported": supported})
        if supported:
            diffs = []
            for user in AUDIT_USERS:
                rates = []
                for group in (0, 1):
                    m = (y_a == cls) & (u_a == user) & (g_lt == group)
                    rates.append(float(e[m].mean()))
                diffs.append(rates[1] - rates[0])
            if all(abs(d) >= MIN_ERROR_GAP for d in diffs) and np.sign(diffs[0]) == np.sign(diffs[1]):
                directions.append({"class": cls, "group1_minus_group0_error_rate": diffs,
                                   "same_direction": True})

    n_supported = sum(r["supported"] for r in coverage)
    n_direction = len(directions)
    gate = bool(n_supported >= MIN_SUPPORTED_CLASSES
                and n_direction >= MIN_DIRECTION_CLASSES
                and cmi_lt_incremental >= MIN_INCREMENTAL_CMI_BITS
                and p_perm <= 0.05)

    result = {
        "status": "stage_a_diagnostic_completed_not_contribution_validation",
        "candidate": "L/T-derived within-class strata as training-only labels for Right-only deployment",
        "selected_source_data_sha256": hashlib.sha256(
            np.asarray(rows, dtype=np.int64).tobytes()
            + np.asarray(y, dtype=np.int64).tobytes()
            + np.asarray(users, dtype=np.int64).tobytes()
            + features.tobytes()
        ).hexdigest(),
        "source_code_sha256": sha256(Path(__file__).resolve()),
        "protocol_sha256": sha256(protocol_path) if protocol_path.is_file() else None,
        "split": {"outer_test_user_sealed": OUTER_TEST_USER,
                  "outer_validation_user_sealed": OUTER_VALIDATION_USER,
                  "source_users": list(SOURCE_USERS), "group_fit_users": list(FIT_USERS),
                  "inner_source_audit_users": list(AUDIT_USERS)},
        "counts": {"source_rows_after_quality_mask": int(len(rows)),
                   "fit_rows": int(fit_mask.sum()), "audit_rows": int(audit_mask.sum()),
                   "fit_classes": sorted(map(int, np.unique(y[fit_mask]))),
                   "audit_classes": sorted(map(int, np.unique(y[audit_mask])))},
        "fixed_method": {"pooled_shape": [10, 21], "within_class_k": 2,
                         "pca_components": 16, "kmeans_n_init": 10,
                         "right_error_probe": "StandardScaler + RidgeClassifier(alpha=1.0,class_weight=balanced)",
                         "cnn_training": False, "parameter_grid": False,
                         "permutations": N_PERMUTATIONS, "seed": SEED},
        "metrics": {"conditional_mi_lt_error_given_class_user_bits": cmi_lt_yu,
                    "conditional_mi_right_error_given_class_user_bits_bits": cmi_r_yu,
                    "incremental_mi_lt_error_given_class_user_right_group_bits": cmi_lt_incremental,
                    "user_only_mi_error_given_class_bits": cmi_user_y,
                    "lt_incremental_permutation_p": p_perm,
                    "permutation_null_95pct": float(np.quantile(null, 0.95)),
                    "supported_classes": int(n_supported),
                    "same_error_direction_classes": int(n_direction)},
        "coverage_by_class": coverage,
        "direction_consistency": directions,
        "stage_a_gate_passed": gate,
        "interpretation_limit": "Groups are statistical view-derived strata, not verified motion/kinematic subgroups. Right error is from the fixed linear probe, not the historical CNN. Stage A is a screen only.",
        "sample_row_ids_sha256": hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest(),
        "audit_rows": [
            {"row_id": int(rows[i]), "subject": int(users[i]), "class": int(y[i]),
             "right_probe_prediction": int(right_pred[j]), "right_probe_error": int(e[j]),
             "lt_group": int(g_lt[j]), "right_group": int(g_r[j])}
            for j, i in enumerate(idx_a)
        ],
        "elapsed_seconds": float(time.monotonic() - started),
    }
    (out_dir / "diagnostic.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="/workspace/uwb-restored-20261020/cache")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--confirm-stage-a", action="store_true",
                    help="required explicit switch; this reads source-user arrays and labels")
    args = ap.parse_args()
    if not args.confirm_stage_a:
        ap.error("No arrays are read by default. Review the frozen design and pass --confirm-stage-a only when authorized.")
    if args.permutations != N_PERMUTATIONS:
        ap.error(f"permutation count is frozen at {N_PERMUTATIONS}; do not tune it")
    with threadpool_limits(limits=1):
        result = run(args)
    print(json.dumps({k: result[k] for k in ("status", "counts", "metrics", "stage_a_gate_passed", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
