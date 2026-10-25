#!/usr/bin/env python3
"""Frozen Stage-A screen for L/T-derived within-class strata.

This is an exploratory diagnostic, not a classifier contribution. It uses no
CNN, no outer-test/validation rows, and never treats the discovered groups as
kinematic labels. See protocol.json and DESIGN.md before execution.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits


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
WALL_TIMEOUT_SECONDS = 900
POLL_SECONDS = 0.05
_ACTIVE_STAGE = "worker_start"
_ACTIVE_FEATURES = None
_FEATURE_ROWS_COMPLETED = 0
_ACTIVE_NULL = None


class BudgetExceeded(BaseException):
    """Raised by the worker's SIGXCPU handler; never a scientific outcome."""


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _partial_artifacts(out_dir: Path) -> list[str]:
    excluded = {"run_status.json", "run_status.json.tmp", "worker.stdout.log", "worker.stderr.log"}
    return sorted(p.name for p in out_dir.iterdir() if p.is_file() and p.name not in excluded)


def write_run_status(out_dir: Path, state: str, phase: str, *, reason: str | None = None,
                     exit_code: int | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    previous: dict[str, Any] = {}
    previous_path = out_dir / "run_status.json"
    if previous_path.is_file():
        try:
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    record = {
        "state": state,
        "phase": phase,
        "last_worker_phase": previous.get("phase"),
        "last_progress": {k: v for k, v in previous.items()
                          if k.endswith("_completed") or k.endswith("_total")},
        "reason": reason,
        "exit_code": exit_code,
        "scientific_result_recorded": False,
        "is_scientific_negative": False,
        "cpu_limit_seconds": HARD_BUDGET_SECONDS,
        "wall_timeout_seconds": WALL_TIMEOUT_SECONDS,
        "partial_artifacts": _partial_artifacts(out_dir),
        "updated_unix": time.time(),
    }
    manifest_path = out_dir / "execution_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record["initial_code_sha256"] = manifest.get("initial_code_sha256")
            record["protocol_sha256"] = manifest.get("frozen_protocol_sha256")
        except (OSError, json.JSONDecodeError):
            pass
    if extra:
        record.update(extra)
    atomic_json(out_dir / "run_status.json", record)
    return record


def _kill_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        # Wall timeout is a hard deadline. Do not add a grace interval that
        # would let a worker exceed its declared wall budget; checkpoints are
        # flushed incrementally before this point.
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run_supervised(command: list[str], out_dir: Path, *, cpu_seconds: int = HARD_BUDGET_SECONDS,
                   wall_seconds: float = WALL_TIMEOUT_SECONDS, poll_seconds: float = POLL_SECONDS) -> dict[str, Any]:
    """Run one worker with kernel CPU RLIMIT and an independent parent wall timer."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    if cpu_seconds <= 0 or wall_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("CPU, wall, and poll limits must be positive")
    write_run_status(out_dir, "running", "worker_start", extra={
        "cpu_limit_seconds": cpu_seconds, "wall_timeout_seconds": wall_seconds,
    })
    stdout_path, stderr_path = out_dir / "worker.stdout.log", out_dir / "worker.stderr.log"
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    child_env = os.environ.copy()
    child_env.update({
        "UWB_STAGE_A_OUTPUT_DIR": str(out_dir.resolve()),
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    })
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            limit_launcher = Path(__file__).with_name("budget_exec.py").resolve()
            launch_command = [sys.executable, str(limit_launcher), str(cpu_seconds), *command]
            proc = subprocess.Popen(launch_command, stdout=stdout, stderr=stderr, env=child_env,
                                    start_new_session=True)
        except BaseException as exc:
            return write_run_status(out_dir, "failed_exception", "supervisor_spawn",
                                    reason=f"worker could not be started: {type(exc).__name__}: {exc}",
                                    extra={"cpu_limit_seconds": cpu_seconds,
                                           "wall_timeout_seconds": wall_seconds})
        timed_out = False
        while proc.poll() is None:
            if time.monotonic() - start >= wall_seconds:
                timed_out = True
                _kill_process_group(proc)
                break
            time.sleep(poll_seconds)
        rc = proc.wait()
    elapsed = time.monotonic() - start
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_used = ((usage_after.ru_utime + usage_after.ru_stime)
                - (usage_before.ru_utime + usage_before.ru_stime))
    if timed_out:
        return write_run_status(out_dir, "aborted_wall_timeout", "parent_supervisor",
                                reason="independent wall-clock limit reached; worker process group terminated",
                                exit_code=rc, extra={"elapsed_seconds": elapsed, "child_cpu_seconds": cpu_used,
                                                     "cpu_limit_seconds": cpu_seconds,
                                                     "wall_timeout_seconds": wall_seconds})

    status_path = out_dir / "run_status.json"
    existing = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    cpu_signal = (rc == -signal.SIGXCPU or
                  (rc == -signal.SIGKILL and cpu_used >= max(0.0, cpu_seconds - 0.15)) or
                  cpu_used >= cpu_seconds)
    if cpu_signal and existing.get("state") not in ("aborted_cpu_limit", "completed", "failed_exception"):
        return write_run_status(out_dir, "aborted_cpu_limit", existing.get("phase", "worker"),
                                reason="kernel process CPU limit reached; scientific result not recorded",
                                exit_code=rc, extra={"elapsed_seconds": elapsed, "child_cpu_seconds": cpu_used,
                                                     "cpu_limit_seconds": cpu_seconds,
                                                     "wall_timeout_seconds": wall_seconds})
    if rc != 0 and existing.get("state") != "aborted_cpu_limit":
        return write_run_status(out_dir, "failed_exception", existing.get("phase", "worker"),
                                reason="worker exited nonzero; inspect stderr; not a scientific negative",
                                exit_code=rc, extra={"elapsed_seconds": elapsed, "child_cpu_seconds": cpu_used,
                                                     "cpu_limit_seconds": cpu_seconds,
                                                     "wall_timeout_seconds": wall_seconds})
    if rc == 0 and existing.get("state") == "completed":
        existing.update({"elapsed_seconds": elapsed, "child_cpu_seconds": cpu_used,
                         "cpu_limit_seconds": cpu_seconds, "wall_timeout_seconds": wall_seconds,
                         "partial_artifacts": _partial_artifacts(out_dir)})
        atomic_json(status_path, existing)
        return existing
    return write_run_status(out_dir, existing.get("state", "failed_exception"), existing.get("phase", "worker"),
                            reason=existing.get("reason") or "worker ended without a completed diagnostic",
                            exit_code=rc, extra={"elapsed_seconds": elapsed, "child_cpu_seconds": cpu_used,
                                                 "cpu_limit_seconds": cpu_seconds,
                                                 "wall_timeout_seconds": wall_seconds})


def _on_sigxcpu(_signum, _frame) -> None:
    # With equal soft/hard limits the kernel may SIGKILL before this handler
    # runs. If delivered, flush both partial memmaps and write a non-science
    # terminal state; the parent remains the authoritative finalizer.
    _flush_active_partial_outputs()
    out = Path(os.environ["UWB_STAGE_A_OUTPUT_DIR"])
    write_run_status(out, "aborted_cpu_limit", _ACTIVE_STAGE,
                     reason="SIGXCPU received at the process-level CPU limit")
    raise BudgetExceeded()


def _flush_active_partial_outputs() -> None:
    for arr in (_ACTIVE_FEATURES, _ACTIVE_NULL):
        if arr is not None:
            try:
                arr.flush()
            except Exception:
                pass


def select_source_rows(subjects: np.ndarray) -> np.ndarray:
    """Use participant metadata to select source rows; no labels or maps."""
    subjects = np.asarray(subjects)
    if subjects.ndim != 1:
        raise ValueError("subject metadata must be one-dimensional")
    return np.flatnonzero(np.isin(subjects, SOURCE_USERS))


def read_selected_source_labels(label_array: Any, source_rows: np.ndarray) -> np.ndarray:
    """Read labels only at rows selected from source-user metadata."""
    rows = np.asarray(source_rows, dtype=np.int64)
    if rows.ndim != 1 or np.any(rows < 0):
        raise ValueError("source row indices must be a nonnegative vector")
    return np.asarray(label_array[rows], dtype=np.int64)


def iter_source_features(map_array: Any, source_rows: np.ndarray):
    """Yield features only for explicit source rows; never probe sealed rows."""
    for row_id in np.asarray(source_rows, dtype=np.int64):
        yield sample_feature(np.asarray(map_array[int(row_id)]))


def fit_audit_masks(users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    users = np.asarray(users)
    fit = np.isin(users, FIT_USERS)
    audit = np.isin(users, AUDIT_USERS)
    if np.any(fit & audit) or not fit.any() or not audit.any():
        raise ValueError("fit/audit users must be nonempty and disjoint")
    if set(np.unique(users[fit])).difference(FIT_USERS) or set(np.unique(users[audit])).difference(AUDIT_USERS):
        raise ValueError("unexpected participant in fit/audit mask")
    return fit, audit


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
                     fit_mask: np.ndarray, audit_mask: np.ndarray, mode: str,
                     model_callback=None):
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
        models[int(cls)] = {"scaler": scaler, "pca": pca, "kmeans": km, "remap": remap,
                            "fit_indices": tr.copy(), "fit_users": np.unique(users[tr]).copy()}
        if model_callback is not None:
            model_callback(mode, models)
    return g_fit, g_audit, models


def serialize_model_state(models: dict) -> dict[str, Any]:
    """Small, deterministic numerical state needed to replay group assignment."""
    out: dict[str, Any] = {}
    for cls, model in sorted(models.items()):
        scaler, pca, km, remap = (model[k] for k in ("scaler", "pca", "kmeans", "remap"))
        out[str(cls)] = {
            "fit_local_indices": model["fit_indices"].astype(int).tolist(),
            "fit_users": model["fit_users"].astype(int).tolist(),
            "scaler_mean": scaler.mean_.astype(float).tolist(),
            "scaler_scale": scaler.scale_.astype(float).tolist(),
            "pca_mean": pca.mean_.astype(float).tolist(),
            "pca_components": pca.components_.astype(float).tolist(),
            "pca_explained_variance": pca.explained_variance_.astype(float).tolist(),
            "kmeans_centers": km.cluster_centers_.astype(float).tolist(),
            "cluster_remap": remap.astype(int).tolist(),
        }
    return out


def environment_manifest() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    def pkg(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    pools = threadpool_info()
    if any(int(pool.get("num_threads", 1)) > 1 for pool in pools):
        raise RuntimeError("threadpoolctl failed to enforce the frozen one-thread numerical budget")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": pkg("scipy"),
        "scikit_learn": pkg("scikit-learn"),
        "threadpoolctl": pkg("threadpoolctl"),
        "cpu_count": os.cpu_count(),
        "threadpools": pools,
        "thread_limit_verified": True,
    }


def run(args: argparse.Namespace) -> dict:
    global _ACTIVE_STAGE, _ACTIVE_FEATURES, _ACTIVE_NULL, _FEATURE_ROWS_COMPLETED
    started = time.monotonic()
    cache = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # parent supervisor owns this immutable run directory
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    code_path = Path(__file__).resolve()
    execution_manifest = {
        "frozen_protocol_sha256": sha256(protocol_path),
        "initial_code_sha256": sha256(code_path),
        "environment": environment_manifest(),
        "cpu_limit_seconds": HARD_BUDGET_SECONDS,
        "wall_timeout_seconds": WALL_TIMEOUT_SECONDS,
        "blas_thread_limit": 1,
        "scientific_protocol_changed": False,
        "source_exchangeability": protocol["source_exchangeability"],
    }
    atomic_json(out_dir / "execution_manifest.json", execution_manifest)
    atomic_json(out_dir / "progress.json", {"phase": "input_open", "rows_completed": 0, "rows_total": None})
    _ACTIVE_STAGE = "input_open"

    # The proposed row-level permutation is not executed unless the design
    # record has an explicit exchangeability clearance. A timeout or this
    # block is never counted as a scientific negative.
    if not execution_manifest["source_exchangeability"]["run_authorized"]:
        write_run_status(out_dir, "blocked_exchangeability_review", "protocol_preflight",
                         reason="sample-level conditional permutation is not justified by available trial metadata; resolve the pre-run exchangeability plan",
                         extra={"source_exchangeability": execution_manifest["source_exchangeability"],
                                "initial_code_sha256": execution_manifest["initial_code_sha256"],
                                "protocol_sha256": execution_manifest["frozen_protocol_sha256"],
                                "cpu_limit_seconds": HARD_BUDGET_SECONDS,
                                "wall_timeout_seconds": WALL_TIMEOUT_SECONDS})
        return {"status": "blocked_exchangeability_review"}

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
    _ACTIVE_STAGE = "source_metadata_selection"
    source_subjects = np.asarray(u_mm)  # participant metadata only; no map/label values
    src_rows = select_source_rows(source_subjects)
    src_y_all = read_selected_source_labels(y_mm, src_rows)
    src_u_all = source_subjects[src_rows].astype(np.int64, copy=False)
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
    # Persist source-only metadata and a low-dimensional partial feature cache
    # locally. These contain no outer-test or outer-validation examples.
    metadata_partial = out_dir / "source_metadata.partial.npz"
    np.savez_compressed(metadata_partial, row_ids=rows.astype(np.int32),
                        labels=y.astype(np.int8), subjects=users.astype(np.int8))
    os.replace(metadata_partial, out_dir / "source_metadata.npz")
    from numpy.lib.format import open_memmap
    feature_partial_path = out_dir / "source_features.partial.npy"
    features = open_memmap(feature_partial_path, mode="w+", dtype=np.float32,
                           shape=(len(rows), 3, 210))
    features[:] = np.nan
    features.flush()
    _ACTIVE_FEATURES = features
    _FEATURE_ROWS_COMPLETED = 0
    _ACTIVE_STAGE = "feature_extraction"
    atomic_json(out_dir / "progress.json", {"phase": _ACTIVE_STAGE, "rows_completed": 0,
                                             "rows_total": int(len(rows))})
    for j, feature in enumerate(iter_source_features(x, rows)):
        features[j] = feature
        _FEATURE_ROWS_COMPLETED = j + 1
        if (j + 1) % 32 == 0 or j + 1 == len(rows):
            features.flush()
            atomic_json(out_dir / "progress.json", {"phase": _ACTIVE_STAGE,
                                                     "rows_completed": j + 1,
                                                     "rows_total": int(len(rows))})
            write_run_status(out_dir, "running", _ACTIVE_STAGE,
                             extra={"rows_completed": j + 1, "rows_total": int(len(rows)),
                                    "cpu_limit_seconds": HARD_BUDGET_SECONDS,
                                    "wall_timeout_seconds": WALL_TIMEOUT_SECONDS})
    features.flush()
    del features
    _ACTIVE_FEATURES = None
    feature_complete_path = out_dir / "source_features.npy"
    os.replace(feature_partial_path, feature_complete_path)
    features = np.load(feature_complete_path, mmap_mode="r", allow_pickle=False)

    # Fixed Right-only linear diagnostic probe (not the historical CNN).
    _ACTIVE_STAGE = "right_probe_fit"
    r_scaler = StandardScaler().fit(features[fit_mask, 2])
    r_fit = r_scaler.transform(features[fit_mask, 2])
    r_audit = r_scaler.transform(features[audit_mask, 2])
    probe = RidgeClassifier(alpha=1.0, class_weight="balanced")
    probe.fit(r_fit, y[fit_mask])
    atomic_json(out_dir / "right_probe_state.json", {
        "scaler_mean": r_scaler.mean_.astype(float).tolist(),
        "scaler_scale": r_scaler.scale_.astype(float).tolist(),
        "classes": probe.classes_.astype(int).tolist(),
        "coef": probe.coef_.astype(float).tolist(),
        "intercept": probe.intercept_.astype(float).tolist(),
        "fit_source_row_ids": rows[fit_mask].astype(int).tolist(),
        "fit_subjects": sorted(map(int, np.unique(users[fit_mask]))),
        "alpha": 1.0,
        "class_weight": "balanced",
    })
    right_pred = probe.predict(r_audit)
    e = (right_pred != y[audit_mask]).astype(np.int8)
    y_a = y[audit_mask]
    u_a = users[audit_mask]
    idx_a = np.flatnonzero(audit_mask)

    # Discover L/T and Right-only partitions with identical fixed machinery.
    model_state: dict[str, Any] = {"protocol_sha256": execution_manifest["frozen_protocol_sha256"],
                                   "initial_code_sha256": execution_manifest["initial_code_sha256"],
                                   "models": {"lt": {}, "right": {}}}
    def persist_models(mode: str, models: dict) -> None:
        model_state["models"][mode] = serialize_model_state(models)
        atomic_json(out_dir / "fitted_models.partial.json", model_state)
        write_run_status(out_dir, "running", _ACTIVE_STAGE,
                         extra={"completed_classes": len(models), "group_mode": mode,
                                "cpu_limit_seconds": HARD_BUDGET_SECONDS,
                                "wall_timeout_seconds": WALL_TIMEOUT_SECONDS})

    _ACTIVE_STAGE = "lt_group_fit"
    write_run_status(out_dir, "running", _ACTIVE_STAGE)
    g_lt_fit, g_lt_a_all, lt_models = classwise_groups(
        features, y, users, fit_mask, audit_mask, "lt", model_callback=persist_models)
    model_state["models"]["lt"] = serialize_model_state(lt_models)
    persist_models("lt", lt_models)
    _ACTIVE_STAGE = "right_group_fit"
    write_run_status(out_dir, "running", _ACTIVE_STAGE)
    g_r_fit, g_r_a_all, r_models = classwise_groups(
        features, y, users, fit_mask, audit_mask, "right", model_callback=persist_models)
    model_state["models"]["right"] = serialize_model_state(r_models)
    persist_models("right", r_models)
    os.replace(out_dir / "fitted_models.partial.json", out_dir / "fitted_models.json")
    g_lt = g_lt_a_all[idx_a]
    g_r = g_r_a_all[idx_a]

    cmi_lt_yu = cmi_bits(e, g_lt, [y_a, u_a])
    cmi_r_yu = cmi_bits(e, g_r, [y_a, u_a])
    cmi_lt_incremental = cmi_bits(e, g_lt, [y_a, u_a, g_r])
    cmi_user_y = cmi_bits(e, u_a, [y_a])

    rng = np.random.default_rng(SEED)
    perm_strata = [y_a, u_a, g_r]
    null_partial_path = out_dir / "null_statistics.partial.npy"
    null_store = open_memmap(null_partial_path, mode="w+", dtype=np.float64, shape=(N_PERMUTATIONS,))
    null_store[:] = np.nan
    null_store.flush()
    _ACTIVE_NULL = null_store
    null = null_store
    _ACTIVE_STAGE = "conditional_permutation"
    atomic_json(out_dir / "progress.json", {"phase": _ACTIVE_STAGE,
                                             "permutations_completed": 0,
                                             "permutations_total": N_PERMUTATIONS})
    for b in range(N_PERMUTATIONS):
        gp = permute_within_strata(g_lt, perm_strata, rng)
        null[b] = cmi_bits(e, gp, perm_strata)
        if (b + 1) % 50 == 0 or b + 1 == N_PERMUTATIONS:
            null_store.flush()
            atomic_json(out_dir / "progress.json", {"phase": _ACTIVE_STAGE,
                                                     "permutations_completed": b + 1,
                                                     "permutations_total": N_PERMUTATIONS})
            write_run_status(out_dir, "running", _ACTIVE_STAGE,
                             extra={"permutations_completed": b + 1,
                                    "permutations_total": N_PERMUTATIONS,
                                    "cpu_limit_seconds": HARD_BUDGET_SECONDS,
                                    "wall_timeout_seconds": WALL_TIMEOUT_SECONDS})
    p_perm = float((1 + np.count_nonzero(null >= cmi_lt_incremental)) / (N_PERMUTATIONS + 1))

    # Keep the complete fixed null distribution and its completed-prefix state.
    null_store.flush()
    del null_store
    _ACTIVE_NULL = None
    os.replace(null_partial_path, out_dir / "null_statistics.npy")
    _ACTIVE_NULL = None

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
        "source_exchangeability": execution_manifest["source_exchangeability"],
        "execution_manifest_sha256": sha256(out_dir / "execution_manifest.json"),
    }
    atomic_json(out_dir / "diagnostic.json", result)
    artifact_files = [p for p in out_dir.iterdir() if p.is_file() and p.name not in {
        "artifact_hashes.json", "run_status.json", "progress.json", "worker.stdout.log", "worker.stderr.log"}]
    atomic_json(out_dir / "artifact_hashes.json", {p.name: sha256(p) for p in sorted(artifact_files)})
    return result


def worker_main(args: argparse.Namespace) -> int:
    signal.signal(signal.SIGXCPU, _on_sigxcpu)
    try:
        with threadpool_limits(limits=1):
            result = run(args)
        if result.get("status") == "blocked_exchangeability_review":
            return 0
        write_run_status(Path(args.output_dir), "completed", "diagnostic_complete",
                         reason="diagnostic completed; this state is not a pass/fail research contribution",
                         extra={"scientific_result_recorded": True,
                                "is_scientific_negative": False,
                                "stage_a_gate_passed": bool(result["stage_a_gate_passed"])})
        return 0
    except BudgetExceeded:
        _flush_active_partial_outputs()
        return 86
    except BaseException as exc:
        _flush_active_partial_outputs()
        import traceback
        traceback.print_exc()
        out_dir = Path(args.output_dir)
        if out_dir.exists():
            write_run_status(out_dir, "failed_exception", _ACTIVE_STAGE,
                             reason=f"worker exception: {type(exc).__name__}: {exc}")
        return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="/workspace/uwb-restored-20261020/cache")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--confirm-stage-a", action="store_true",
                    help="required explicit switch; this reads source-user arrays and labels")
    args = ap.parse_args()
    if not args.confirm_stage_a:
        ap.error("No arrays are read by default. Review the frozen design and pass --confirm-stage-a only when authorized.")
    if args.permutations != N_PERMUTATIONS:
        ap.error(f"permutation count is frozen at {N_PERMUTATIONS}; do not tune it")
    if args._worker:
        return worker_main(args)
    command = [sys.executable, str(Path(__file__).resolve()), "--_worker",
               "--cache-dir", str(Path(args.cache_dir).resolve()),
               "--output-dir", str(Path(args.output_dir).resolve()),
               "--permutations", str(args.permutations), "--confirm-stage-a"]
    status = run_supervised(command, Path(args.output_dir),
                            cpu_seconds=HARD_BUDGET_SECONDS,
                            wall_seconds=WALL_TIMEOUT_SECONDS)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if status.get("state") in ("completed", "blocked_exchangeability_review") else 2


if __name__ == "__main__":
    raise SystemExit(main())
