#!/usr/bin/env python3
"""Independent, bounded audit of paired_time_alignment_diagnostic outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def independent_corr(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    n = len(a)
    if lag > 0:
        aa, bb = a[: n - lag], b[lag:]
    elif lag < 0:
        aa, bb = a[-lag:], b[: n + lag]
    else:
        aa, bb = a, b
    return float(np.corrcoef(aa.astype(np.float64), bb.astype(np.float64))[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/workspace/uwb-restored-20260930"))
    ap.add_argument("--results", type=Path, default=Path("/workspace/uwb-pairwise-time-alignment-20261008"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.results / "SUMMARY.json").read_text(encoding="utf-8"))
    saved = np.load(args.results / "PAIRED_VIEW_READINGS.npz", allow_pickle=False)
    x = np.load(args.data / "cache/X_clutterremoved.npy", mmap_mode="r", allow_pickle=False)
    subj = np.load(args.data / "cache/subj_clutterremoved.npy", mmap_mode="r", allow_pickle=False)
    manifest = json.loads((args.data / "experiments/p1_20260905/results/data_manifest.json").read_text(encoding="utf-8"))

    dup = manifest["clutterremoved"].get("exact_duplicate_units", {})
    duplicate_equalities = []
    for a, b in sorted((int(k), int(v)) for k, v in dup.items()):
        duplicate_equalities.append(bool(np.array_equal(x[a], x[b])))
    ids = saved["sample_ids"].astype(np.int64)
    if not np.array_equal(saved["subjects"], np.asarray(subj[ids], dtype=np.int64)):
        raise SystemExit("saved subject IDs do not align with original cache row IDs")
    seed = int(summary["seed"])
    rng = np.random.default_rng(seed ^ 0x5A17)
    selection = np.sort(rng.choice(len(ids), size=min(128, len(ids)), replace=False))
    pair_list = ((0, 1), (0, 2), (1, 2))
    lags = np.arange(-4, 5)
    max_abs_corr_error = 0.0
    exact_lag_matches = 0
    total_lag_checks = 0
    for pos in selection:
        sample_id = int(ids[pos])
        block = np.asarray(x[sample_id], dtype=np.float32)
        env = np.sqrt(np.mean(np.square(np.diff(block, axis=1), dtype=np.float32), axis=-1, dtype=np.float32))
        env = (env - env.mean(axis=-1, keepdims=True)) / np.maximum(env.std(axis=-1, keepdims=True), 1e-8)
        for j, (v_a, v_b) in enumerate(pair_list):
            ref = np.asarray([independent_corr(env[v_a], env[v_b], int(lag)) for lag in lags])
            observed = np.asarray([
                independent_corr(env[v_a], env[v_b], int(lag)) for lag in lags
            ])
            # Compare against stored zero, maximum, and argmax from a separately coded reference.
            stored_zero = float(saved["matched_zero"][pos, j])
            stored_best = float(saved["matched_best"][pos, j])
            stored_lag = int(saved["matched_lag"][pos, j])
            max_abs_corr_error = max(max_abs_corr_error, abs(float(ref[4]) - stored_zero), abs(float(observed.max()) - stored_best))
            exact_lag_matches += int(int(lags[int(np.argmax(observed))]) == stored_lag)
            total_lag_checks += 1

    recovered = saved["injected_recovered"].astype(np.int64)
    injected = saved["injected_lags"].astype(np.int64)
    recovered_fraction = np.mean(recovered == injected[None, None, :], axis=(0, 2))
    pair_keys = ["0-1", "0-2", "1-2"]
    pairs_corr = [
        key for key in pair_keys
        if summary["pair_summary"][key]["median_lag_zero_correlation"] >= 0.30
        and summary["pair_summary"][key]["matched_minus_same_user_shuffle_median_best_correlation"] >= 0.10
    ]
    pairs_stable = [
        key for key in pair_keys
        if summary["pair_summary"][key]["users_with_median_abs_lag_le_1"] >= 6
        and summary["pair_summary"][key]["boundary_peak_fraction"] < 0.05
    ]
    recomputed_gate = bool(
        summary["valid_fraction_of_unique"] >= 0.95
        and len(set(pairs_corr) & set(pairs_stable)) >= 2
        and np.all(recovered_fraction >= 0.90)
    )
    if max_abs_corr_error > 1e-5 or exact_lag_matches != total_lag_checks:
        raise SystemExit("stored pairwise correlations/lag differ from independent reference")
    if not all(duplicate_equalities) or not np.all(recovered_fraction >= 0.90):
        raise SystemExit("duplicate or injected-shift control audit failed")
    if recomputed_gate != summary["gate"]["continue_to_separate_application_preregistration"]:
        raise SystemExit("frozen gate arithmetic differs")

    audit = {
        "independent_reference": "NumPy corrcoef per overlap; no import of runner code",
        "n_randomly_replayed_records": int(len(selection)),
        "n_pair_lag_vectors_replayed": int(total_lag_checks),
        "max_abs_correlation_error": max_abs_corr_error,
        "exact_best_lag_match_fraction": float(exact_lag_matches / total_lag_checks),
        "duplicate_triplets_rechecked_equal": bool(all(duplicate_equalities)),
        "n_duplicate_triplets": len(duplicate_equalities),
        "injected_shift_exact_recovery_fraction_by_pair": recovered_fraction.astype(float).tolist(),
        "correlation_pairs_passing_fixed_rule": pairs_corr,
        "stable_pairs_passing_fixed_rule": pairs_stable,
        "recomputed_continue_gate": recomputed_gate,
        "summary_sha256": sha256(args.results / "SUMMARY.json"),
        "readings_sha256": sha256(args.results / "PAIRED_VIEW_READINGS.npz"),
        "labels_accessed": False,
        "model_fitting": False,
    }
    (args.out / "AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
