#!/usr/bin/env python3
"""Frozen source-only feasibility screen for range-time Betti-curve features."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import signal
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import ndimage
import sklearn
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path("/workspace/uwb-restored-20261020")
RUN_ID = "run03"
OUT = Path("/workspace/uwb-range-time-topology-20261028/results/run03")
PROTOCOL = Path("/workspace/uwb-range-time-topology-20261028/protocol/STAGE_A_RUN03.json")
SCRIPT = Path(__file__).resolve()
USERS = (1, 2, 3, 4, 5, 6)
VIEWS = ("Left", "Top", "Right")
QS = np.asarray([0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95], dtype=np.float64)
AMP_QS = np.asarray([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90], dtype=np.float64)
STRUCT4 = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
STRUCT8 = np.ones((3, 3), dtype=np.uint8)
SEED = 20261028
ACTIVE_STATE: dict | None = None
RUN_CPU_START = 0.0
RUN_WALL_START = 0.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def descriptor(image: np.ndarray, permutation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Envelope geometry only; no raw complex phase is available in this release.
    z = np.abs(np.asarray(image, dtype=np.float32))
    # Frozen 3x3 max pooling: 90x189 -> 30x63.
    z = z.reshape(30, 3, 63, 3).max(axis=(1, 3)).astype(np.float32, copy=False)
    z /= np.float32(z.mean() + 1e-8)
    amp = np.quantile(z, AMP_QS).astype(np.float32)
    top = np.empty(2 * len(QS), dtype=np.float32)
    null = np.empty_like(top)
    shuffled = z.reshape(-1)[permutation].reshape(z.shape)
    for j, q in enumerate(QS):
        threshold = np.quantile(z, q)
        mask = z >= threshold
        _, b0 = ndimage.label(mask, structure=STRUCT4)
        filled = ndimage.binary_fill_holes(mask, structure=STRUCT8)
        holes = filled & ~mask
        _, b1 = ndimage.label(holes, structure=STRUCT4)
        top[2 * j:2 * j + 2] = np.log1p((b0, b1))

        smask = shuffled >= threshold
        _, sb0 = ndimage.label(smask, structure=STRUCT4)
        sfilled = ndimage.binary_fill_holes(smask, structure=STRUCT8)
        sholes = sfilled & ~smask
        _, sb1 = ndimage.label(sholes, structure=STRUCT4)
        null[2 * j:2 * j + 2] = np.log1p((sb0, sb1))
    return top, amp, null


def balanced_weights(y: np.ndarray, subject: np.ndarray) -> np.ndarray:
    w = np.zeros(len(y), dtype=np.float64)
    us = np.unique(subject)
    for u in us:
        ui = subject == u
        cs = np.unique(y[ui])
        for c in cs:
            ci = ui & (y == c)
            w[ci] = 1.0 / (len(us) * len(cs) * int(ci.sum()))
    return w / w.mean()


def fit_score(X: np.ndarray, y: np.ndarray, subject: np.ndarray, ids: np.ndarray,
              held: int, tag: str, view: str) -> tuple[dict, dict]:
    tr = subject != held
    te = ~tr
    scaler = StandardScaler()
    xtr = scaler.fit_transform(X[tr])
    xte = scaler.transform(X[te])
    model = RidgeClassifier(alpha=1.0)
    sw = balanced_weights(y[tr], subject[tr])
    model.fit(xtr, y[tr], sample_weight=sw)
    pred = model.predict(xte)
    labels = np.unique(y[te])
    score = float(recall_score(y[te], pred, labels=labels, average="macro", zero_division=0))
    detail = {
        "method": tag, "view": view, "held_source_user": held,
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "classes_test": labels.astype(int).tolist(), "macro_recall": score,
        "test_sample_ids": ids[te].astype(int).tolist(),
        "test_y_true": y[te].astype(int).tolist(),
        "test_y_pred": pred.astype(int).tolist(),
    }
    model_file = OUT / f"model_{tag}_{view}_held{held}.npz"
    np.savez_compressed(model_file, train_ids=ids[tr], test_ids=ids[te], classes=model.classes_,
                        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                        coef=model.coef_, intercept=model.intercept_)
    return detail, {"model_file": str(model_file), "model_sha256": sha256(model_file)}


def budget_signal(signum, frame) -> None:
    # Persist an explicit incomplete terminal state if the independent wall timer
    # or the preemptive CPU soft limit fires. The hard CPU ceiling remains 900 s.
    raise TimeoutError(f"budget signal {signum}")


def mark_incomplete(exc: BaseException) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state_path = OUT / "state.json"
    state = ACTIVE_STATE
    if state is None and state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except Exception:
            state = {}
    if state is None:
        state = {}
    state.update({
        "status": "incomplete", "completed_unix": time.time(),
        "cpu_seconds": time.process_time() - RUN_CPU_START if RUN_CPU_START else None,
        "wall_seconds": time.monotonic() - RUN_WALL_START if RUN_WALL_START else None,
        "failure_type": type(exc).__name__, "failure": str(exc),
        "partial_model_files": [str(p) for p in sorted(OUT.glob("model_*.npz"))],
        "scientific_result": "none_incomplete_not_a_negative"
    })
    try:
        write_json(state_path, state)
    except Exception:
        pass
    fail = {"run_id": RUN_ID, "status": "incomplete", "error_type": type(exc).__name__,
            "error": str(exc), "note": "engineering failure or budget interruption is not a scientific negative"}
    try:
        write_json(OUT / "failure.json", fail)
    except Exception:
        pass


def main() -> int:
    global ACTIVE_STATE, RUN_CPU_START, RUN_WALL_START
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_sha = sha256(PROTOCOL)
    code_sha = sha256(SCRIPT)
    cpu_start = time.process_time()
    wall_start = time.monotonic()
    RUN_CPU_START, RUN_WALL_START = cpu_start, wall_start
    signal.signal(signal.SIGXCPU, budget_signal)
    signal.signal(signal.SIGTERM, budget_signal)
    state = {
        "run_id": RUN_ID, "status": "running", "started_unix": time.time(), "protocol_sha256": protocol_sha,
        "code_sha256": code_sha, "cpu_limit_seconds": 900, "wall_limit_seconds": 900,
        "participants_read_for_labels": list(USERS), "audit_labels_read": False,
        "completed_feature_records": 0, "completed_models": 0, "results": []
    }
    ACTIVE_STATE = state
    write_json(OUT / "state.json", state)
    # SIGXCPU at 890 s persists an incomplete receipt; 900 s is the hard CPU ceiling.
    # Wall time is independently enforced by the launcher at 900 s.
    resource.setrlimit(resource.RLIMIT_CPU, (885, 895))
    with threadpool_limits(limits=1):
        x = np.load(ROOT / "cache/X_clutterremoved.npy", mmap_mode="r")
        s_all = np.load(ROOT / "cache/subj_clutterremoved.npy", mmap_mode="r")
        y_all = np.load(ROOT / "cache/y_clutterremoved.npy", mmap_mode="r")
        if x.shape != (9593, 3, 90, 189) or x.dtype != np.float32:
            raise RuntimeError(f"unexpected X metadata: {x.shape}, {x.dtype}")
        if s_all.shape != (9593,) or y_all.shape != (9593,):
            raise RuntimeError("unexpected source metadata vector shape")
        # Only user IDs are read globally to select source records. Gesture labels for users 7-8
        # are never indexed or loaded into a Python array.
        s_source_all = np.asarray(s_all)
        source_idx = np.flatnonzero(np.isin(s_source_all, USERS))
        source_subject = s_source_all[source_idx].astype(np.int16)
        y_source_all = np.asarray(y_all[source_idx]).astype(np.int16)
        quality = ~(((source_subject == 2) & np.isin(y_source_all, [10, 11])) |
                    ((source_subject == 6) & (y_source_all == 3)))
        ids = source_idx[quality]
        subject = source_subject[quality]
        y = y_source_all[quality]
        expected = {1: 1200, 2: 1000, 3: 1200, 4: 1200, 5: 1200, 6: 1100}
        got = {u: int(np.sum(subject == u)) for u in USERS}
        if got != expected or len(ids) != 6900:
            raise RuntimeError(f"quality-mask retained counts mismatch: {got}, total={len(ids)}")
        classes = np.unique(y)
        if len(classes) != 12:
            raise RuntimeError(f"expected 12 source classes after mask, found {classes.tolist()}")
        rng = np.random.default_rng(SEED)
        permutation = rng.permutation(30 * 63)
        features = {k: np.empty((len(ids), 3, d), dtype=np.float32)
                    for k, d in (("topology", 16), ("amplitude", 9), ("spatial_null", 16))}
        for row, sample_idx in enumerate(ids):
            for v in range(3):
                t, a, n = descriptor(x[int(sample_idx), v], permutation)
                features["topology"][row, v] = t
                features["amplitude"][row, v] = a
                features["spatial_null"][row, v] = n
            if (row + 1) % 250 == 0:
                state["completed_feature_records"] = row + 1
                state["cpu_seconds"] = time.process_time() - cpu_start
                state["wall_seconds"] = time.monotonic() - wall_start
                write_json(OUT / "state.json", state)
                print(f"features {row+1}/{len(ids)} cpu={state['cpu_seconds']:.1f}s", flush=True)
        state["completed_feature_records"] = len(ids)
        np.savez_compressed(OUT / "source_features.npz", **features, labels=y,
                            subjects=subject, sample_ids=ids, classes=classes)
        state["features_sha256"] = sha256(OUT / "source_features.npz")
        for v, view in enumerate(VIEWS):
            for method in ("topology", "amplitude", "spatial_null"):
                for held in USERS:
                    detail, model = fit_score(features[method][:, v], y, subject, ids,
                                              held, method, view)
                    detail.update(model)
                    state["results"].append(detail)
                    state["completed_models"] += 1
                    state["cpu_seconds"] = time.process_time() - cpu_start
                    state["wall_seconds"] = time.monotonic() - wall_start
                    write_json(OUT / "state.json", state)
                    print(f"model {state['completed_models']}/54 {method} {view} held={held} "
                          f"macro={detail['macro_recall']:.6f}", flush=True)
        # Frozen gate summary; no significance tests and no post-hoc view selection.
        lookup = {(r["method"], r["view"], r["held_source_user"]): r["macro_recall"]
                  for r in state["results"]}
        diffs = {}
        view_means = {}
        view_passes = {}
        for view in VIEWS:
            ds = [lookup[("topology", view, u)] - lookup[("amplitude", view, u)] for u in USERS]
            dn = [lookup[("topology", view, u)] - lookup[("spatial_null", view, u)] for u in USERS]
            diffs[view] = {str(u): ds[i] for i, u in enumerate(USERS)}
            view_means[view] = {"vs_amplitude": float(np.mean(ds)), "vs_spatial_null": float(np.mean(dn))}
            view_passes[view] = sum(d >= 0 for d in ds) >= 4
        all_da = [lookup[("topology", v, u)] - lookup[("amplitude", v, u)]
                  for v in VIEWS for u in USERS]
        all_dn = [lookup[("topology", v, u)] - lookup[("spatial_null", v, u)]
                  for v in VIEWS for u in USERS]
        go = (float(np.mean(all_da)) >= 0.010 and sum(view_passes.values()) >= 2 and
              all(view_means[v]["vs_amplitude"] >= -0.010 for v in VIEWS) and
              float(np.mean(all_dn)) >= 0.010)
        state.update({
            "status": "complete", "completed_unix": time.time(),
            "cpu_seconds": time.process_time() - cpu_start,
            "wall_seconds": time.monotonic() - wall_start,
            "source_participant_retained_counts": got,
            "per_user_view_topology_minus_amplitude": diffs,
            "view_means": view_means, "view_passes_4_of_6": view_passes,
            "overall_mean_topology_minus_amplitude": float(np.mean(all_da)),
            "overall_mean_topology_minus_spatial_null": float(np.mean(all_dn)),
            "frozen_stage_a_gate": "GO_TO_PROTOCOL_DESIGN_ONLY" if go else "STOP_CANDIDATE",
            "scientific_contribution_validated": False,
            "outer_audit_users_7_8_used": False,
            "p_values_computed": False
        })
        write_json(OUT / "state.json", state)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        mark_incomplete(exc)
        raise
