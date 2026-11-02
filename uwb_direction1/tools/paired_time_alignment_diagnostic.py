#!/usr/bin/env python3
"""Label-free feasibility screen for sample-paired UWB view timing.

The script intentionally never opens gesture labels, model outputs, or trained
features. It reports envelope correspondence only; it cannot validate clock
offset without timestamps or synchronized reference video.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

LAGS = np.arange(-4, 5, dtype=np.int64)
N_NULL = 20
N_SHIFT = 512
SEED = 20261008


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def corr_at_lag(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    if lag >= 0:
        aa, bb = a[: len(a) - lag or None], b[lag:]
    else:
        aa, bb = a[-lag:], b[: len(b) + lag]
    if len(aa) < 8:
        return float("nan")
    aa = aa.astype(np.float64, copy=False)
    bb = bb.astype(np.float64, copy=False)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    den = np.sqrt(np.dot(aa, aa) * np.dot(bb, bb))
    return float(np.dot(aa, bb) / den) if den > 0 else float("nan")


def correlations(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    values = np.asarray([corr_at_lag(a, b, int(lag)) for lag in LAGS])
    if not np.isfinite(values).all():
        return float("nan"), float("nan"), 0
    best = int(np.argmax(values))
    return float(values[LAGS.tolist().index(0)]), float(values[best]), int(LAGS[best])


def shifted(x: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros_like(x)
    if lag > 0:
        out[lag:] = x[:-lag]
    elif lag < 0:
        out[:lag] = x[-lag:]
    else:
        out[:] = x
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/workspace/uwb-restored-20260930"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    cache = args.data / "cache"
    x_path = cache / "X_clutterremoved.npy"
    subj_path = cache / "subj_clutterremoved.npy"
    manifest_path = args.data / "experiments/p1_20260905/results/data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    x = np.load(x_path, mmap_mode="r", allow_pickle=False)
    subj = np.load(subj_path, mmap_mode="r", allow_pickle=False)
    expected_x = manifest["hashes"][r"D:\uwb_gestures\cache\X_clutterremoved.npy"]
    expected_subj = manifest["hashes"][r"D:\uwb_gestures\cache\subj_clutterremoved.npy"]
    actual_x = sha256(x_path)
    actual_subj = sha256(subj_path)
    if actual_x != expected_x or actual_subj != expected_subj:
        raise SystemExit("restored source arrays differ from the preserved data manifest")
    if x.shape != (9593, 3, 90, 189) or x.dtype != np.float32:
        raise SystemExit(f"unexpected X schema: {x.shape}, {x.dtype}")
    if subj.shape != (9593,) or not np.issubdtype(subj.dtype, np.integer):
        raise SystemExit(f"unexpected subject schema: {subj.shape}, {subj.dtype}")

    duplicates = manifest["clutterremoved"].get("exact_duplicate_units", {})
    duplicate_ids: list[int] = []
    duplicate_checks: list[dict] = []
    for duplicate, original in sorted((int(k), int(v)) for k, v in duplicates.items()):
        equal = bool(np.array_equal(x[duplicate], x[original]))
        duplicate_checks.append({"duplicate_id": duplicate, "original_id": original, "all_view_samples_equal": equal})
        if not equal:
            raise SystemExit(f"manifest duplicate mismatch: {duplicate} vs {original}")
        duplicate_ids.append(duplicate)
    ids = np.setdiff1d(np.arange(len(x), dtype=np.int64), np.asarray(duplicate_ids, dtype=np.int64))

    # A bounded block read limits working memory to roughly 105 MiB.
    traces = np.empty((len(ids), 3, 89), dtype=np.float32)
    finite = np.ones(len(ids), dtype=bool)
    for start in range(0, len(ids), 256):
        block_ids = ids[start : start + 256]
        block = np.asarray(x[block_ids], dtype=np.float32)
        if not np.isfinite(block).all():
            finite[start : start + len(block_ids)] = np.isfinite(block).all(axis=(1, 2, 3))
        delta = np.diff(block, axis=2)
        env = np.sqrt(np.mean(np.square(delta, dtype=np.float32), axis=-1, dtype=np.float32))
        mu = env.mean(axis=-1, keepdims=True)
        sd = env.std(axis=-1, keepdims=True)
        valid_block = np.isfinite(env).all(axis=(1, 2)) & (sd[:, :, 0] > 1e-8).all(axis=1)
        finite[start : start + len(block_ids)] &= valid_block
        z = (env - mu) / np.maximum(sd, 1e-8)
        traces[start : start + len(block_ids)] = z
    kept_ids = ids[finite]
    traces = traces[finite]
    subjects = np.asarray(subj[kept_ids], dtype=np.int64)

    rng = np.random.default_rng(SEED)
    pairs = ((0, 1), (0, 2), (1, 2))
    matched_zero = np.full((len(kept_ids), len(pairs)), np.nan, dtype=np.float32)
    matched_best = np.full_like(matched_zero, np.nan)
    matched_lag = np.zeros((len(kept_ids), len(pairs)), dtype=np.int8)
    for n in range(len(kept_ids)):
        for j, (a_view, b_view) in enumerate(pairs):
            zero, best, lag = correlations(traces[n, a_view], traces[n, b_view])
            matched_zero[n, j], matched_best[n, j], matched_lag[n, j] = zero, best, lag

    # Same-subject rematching uses IDs only (never gesture labels).
    chosen = np.sort(rng.choice(len(kept_ids), size=min(N_SHIFT, len(kept_ids)), replace=False))
    null_best = np.full((N_NULL, len(chosen), len(pairs)), np.nan, dtype=np.float32)
    subject_to_positions = {s: np.flatnonzero(subjects == s) for s in np.unique(subjects)}
    for repeat in range(N_NULL):
        permuted_index = np.arange(len(kept_ids), dtype=np.int64)
        for positions in subject_to_positions.values():
            permuted_index[positions] = rng.permutation(positions)
        for q, n in enumerate(chosen):
            p = int(permuted_index[n])
            for j, (a_view, b_view) in enumerate(pairs):
                _, best, _ = correlations(traces[n, a_view], traces[p, b_view])
                null_best[repeat, q, j] = best

    # Inject known shifts into one trace of same-event pairs. Zero padding avoids wraparound.
    injection_lags = np.asarray([-4, -3, -2, -1, 1, 2, 3, 4], dtype=np.int8)
    recovered = np.zeros((len(chosen), len(pairs), len(injection_lags)), dtype=np.int8)
    for q, n in enumerate(chosen):
        for j, (a_view, b_view) in enumerate(pairs):
            for k, injected in enumerate(injection_lags):
                # Self-pair before injecting the offset so the expected lag is known exactly.
                source_trace = traces[n, a_view]
                _, _, estimated = correlations(source_trace, shifted(source_trace, int(injected)))
                recovered[q, j, k] = estimated

    per_user: dict[str, dict] = {}
    for s in np.unique(subjects):
        mask = subjects == s
        per_user[str(int(s))] = {
            "n": int(mask.sum()),
            "median_best_lag_by_pair": np.median(matched_lag[mask], axis=0).astype(float).tolist(),
            "fraction_abs_lag_le_1_by_pair": np.mean(np.abs(matched_lag[mask]) <= 1, axis=0).astype(float).tolist(),
        }

    summary = {
        "protocol": "PAIRWISE_TIME_ALIGNMENT_FEASIBILITY_PROTOCOL_20261008.md",
        "seed": SEED,
        "n_original": int(len(x)),
        "n_exact_duplicate_triplets_verified_and_excluded": len(duplicate_ids),
        "n_unique": int(len(ids)),
        "n_valid_nonconstant": int(len(kept_ids)),
        "valid_fraction_of_unique": float(len(kept_ids) / len(ids)),
        "duplicate_checks": duplicate_checks,
        "view_pairs": [list(p) for p in pairs],
        "lag_range_frames": [-4, 4],
        "pair_summary": {},
        "per_user": per_user,
        "same_user_permutation": {
            "n_permutations": N_NULL,
            "n_records_per_permutation": int(len(chosen)),
            "median_best_correlation_by_pair": np.nanmedian(null_best, axis=(0, 1)).astype(float).tolist(),
            "q05_best_correlation_by_pair": np.nanquantile(null_best, 0.05, axis=(0, 1)).astype(float).tolist(),
            "q95_best_correlation_by_pair": np.nanquantile(null_best, 0.95, axis=(0, 1)).astype(float).tolist(),
        },
        "injected_shift_control": {
            "n_records": int(len(chosen)),
            "injected_lags_frames": injection_lags.astype(int).tolist(),
            "exact_recovery_fraction_by_pair": np.mean(recovered == injection_lags[None, None, :], axis=(0, 2)).astype(float).tolist(),
            "recovered_lags_sha256": hashlib.sha256(recovered.tobytes()).hexdigest(),
        },
        "input_hashes": {"X_clutterremoved.npy": actual_x, "subj_clutterremoved.npy": actual_subj},
        "label_accessed": False,
        "classifier_or_model_fitting": False,
        "clock_offset_ground_truth_available": False,
        "interpretation_limit": "Sample-paired envelope correspondence only; estimated lag is not a verified sensor-clock offset.",
    }
    for j, pair in enumerate(pairs):
        pair_key = "-".join(map(str, pair))
        user_medians = []
        boundary_by_user = []
        for s in np.unique(subjects):
            user_lags = matched_lag[subjects == s, j]
            user_medians.append(float(np.median(user_lags)))
            boundary_by_user.append(float(np.mean(np.abs(user_lags) == 4)))
        summary["pair_summary"][pair_key] = {
            "median_lag_zero_correlation": float(np.nanmedian(matched_zero[:, j])),
            "median_best_lag_correlation": float(np.nanmedian(matched_best[:, j])),
            "median_best_lag_frames": float(np.median(matched_lag[:, j])),
            "median_abs_best_lag_frames": float(np.median(np.abs(matched_lag[:, j]))),
            "boundary_peak_fraction": float(np.mean(np.abs(matched_lag[:, j]) == 4)),
            "per_user_median_abs_lag_frames": [float(np.median(np.abs(matched_lag[subjects == s, j]))) for s in np.unique(subjects)],
            "users_with_median_abs_lag_le_1": int(sum(np.median(np.abs(matched_lag[subjects == s, j])) <= 1 for s in np.unique(subjects))),
            "user_median_best_lags_frames": user_medians,
            "user_boundary_fractions": boundary_by_user,
            "matched_minus_same_user_shuffle_median_best_correlation": float(np.nanmedian(matched_best[:, j]) - np.nanmedian(null_best[:, :, j])),
        }
    summary["gate"] = {
        "valid_fraction_ge_0_95": bool(summary["valid_fraction_of_unique"] >= 0.95),
        "pairs_meeting_correlation_and_permutation_delta": [
            "-".join(map(str, pairs[j]))
            for j in range(len(pairs))
            if summary["pair_summary"]["-".join(map(str, pairs[j]))]["median_lag_zero_correlation"] >= 0.30
            and summary["pair_summary"]["-".join(map(str, pairs[j]))]["matched_minus_same_user_shuffle_median_best_correlation"] >= 0.10
        ],
        "pairs_meeting_lag_stability_and_boundary": [
            "-".join(map(str, pairs[j]))
            for j in range(len(pairs))
            if summary["pair_summary"]["-".join(map(str, pairs[j]))]["users_with_median_abs_lag_le_1"] >= 6
            and summary["pair_summary"]["-".join(map(str, pairs[j]))]["boundary_peak_fraction"] < 0.05
        ],
        "synthetic_shift_exact_recovery_ge_0_90_for_each_pair": bool(np.all(np.asarray(summary["injected_shift_control"]["exact_recovery_fraction_by_pair"]) >= 0.90)),
    }
    summary["gate"]["continue_to_separate_application_preregistration"] = bool(
        summary["gate"]["valid_fraction_ge_0_95"]
        and len(set(summary["gate"]["pairs_meeting_correlation_and_permutation_delta"]) & set(summary["gate"]["pairs_meeting_lag_stability_and_boundary"])) >= 2
        and summary["gate"]["synthetic_shift_exact_recovery_ge_0_90_for_each_pair"]
    )
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.out / "PAIRED_VIEW_READINGS.npz",
        sample_ids=kept_ids.astype(np.int32),
        subjects=subjects.astype(np.int8),
        matched_zero=matched_zero,
        matched_best=matched_best,
        matched_lag=matched_lag,
        sampled_ids=kept_ids[chosen].astype(np.int32),
        null_best=null_best,
        injected_lags=injection_lags,
        injected_recovered=recovered,
    )
    (args.out / "SOURCE_HASHES.json").write_text(json.dumps({
        "script": sha256(Path(__file__)),
        "protocol": sha256(Path(__file__).resolve().parents[1] / "research/PAIRWISE_TIME_ALIGNMENT_FEASIBILITY_PROTOCOL_20261008.md"),
        "summary": sha256(args.out / "SUMMARY.json"),
        "readings": sha256(args.out / "PAIRED_VIEW_READINGS.npz"),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(args.out / "SUMMARY.json"), "gate": summary["gate"], "pair_summary": summary["pair_summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
