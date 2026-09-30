"""Render a traceable three-view CIR triplet directly from UWB-Gestures CSV files.

The drawing path is intentionally simple and auditable:
  raw CSV -> repetition slice (90 x 189) -> abs -> per-view sample z-score.
No interpolation, smoothing, denoising, or synthetic texture is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VIEW_NAMES = ("Left", "Top", "Right")
VIEW_LABELS = ("L", "T", "R")
T, M = 90, 189


def find_source_files(root: Path, subject: int, gesture: int) -> list[Path]:
    paths = []
    for view in VIEW_NAMES:
        pattern = f"**/HV{subject}_G{gesture}_Radar{view}_ClutterRemoved_100samples.csv"
        matches = list(root.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one file for {view}, found {len(matches)}: {matches}")
        paths.append(matches[0])
    return paths


def load_triplets(paths: list[Path]) -> np.ndarray:
    views = []
    for path in paths:
        data = pd.read_csv(path, header=None).to_numpy(dtype=np.float32)
        if data.shape[1] != M or data.shape[0] % T:
            raise ValueError(f"Unexpected CSV shape {data.shape}: {path}")
        views.append(data.reshape(-1, T, M))
    n = min(v.shape[0] for v in views)
    return np.stack([v[:n] for v in views], axis=1)  # (N, 3, 90, 189)


def abs_std(x: np.ndarray) -> np.ndarray:
    """Match the paper input path: |x| then per-sample, per-view z-score."""
    x = np.abs(x.astype(np.float64))
    mu = x.mean(axis=(-2, -1), keepdims=True)
    sd = x.std(axis=(-2, -1), keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def structure_scores(z: np.ndarray) -> np.ndarray:
    """Rank repetitions by coherent high-response structure in all three views."""
    positive = np.maximum(z, 0.0)
    ridge = np.percentile(positive, 99.5, axis=(-2, -1)).mean(axis=1)
    time_profile = positive.max(axis=-1)
    temporal_contrast = time_profile.std(axis=-1).mean(axis=1)
    range_profile = positive.max(axis=-2)
    range_contrast = range_profile.std(axis=-1).mean(axis=1)
    weakest_view = np.percentile(positive, 99.0, axis=(-2, -1)).min(axis=1)
    return ridge + 0.45 * temporal_contrast + 0.25 * range_contrast + 0.35 * weakest_view


def common_limits(z: np.ndarray) -> tuple[float, float]:
    # Keep the negative standardized background visible without letting isolated
    # extrema dominate the colour scale.
    return float(np.percentile(z, 2.0)), float(np.percentile(z, 99.75))


def draw_heatmap(ax, arr: np.ndarray, vmin: float, vmax: float) -> None:
    ax.imshow(
        arr,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_candidates(z: np.ndarray, scores: np.ndarray, output: Path, count: int = 6) -> list[int]:
    ranked = np.argsort(scores)[::-1]
    # Avoid showing near-identical adjacent repeats when possible.
    chosen: list[int] = []
    for idx in ranked:
        if all(abs(int(idx) - old) >= 3 for old in chosen):
            chosen.append(int(idx))
        if len(chosen) == count:
            break
    subset = z[chosen]
    vmin, vmax = common_limits(subset)
    fig, axes = plt.subplots(count, 3, figsize=(10.6, 1.45 * count), dpi=180)
    for row, idx in enumerate(chosen):
        for col, lab in enumerate(VIEW_LABELS):
            ax = axes[row, col]
            draw_heatmap(ax, z[idx, col], vmin, vmax)
            if row == 0:
                ax.set_title(f"{lab} view", fontsize=10, pad=5, color="#263442", weight="bold")
        axes[row, 0].text(
            -0.025,
            0.5,
            f"rep {idx:02d}\nscore {scores[idx]:.2f}",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8,
            color="#4b5563",
        )
    fig.suptitle(
        "Real synchronized UWB-Gestures CIR candidates (ClutterRemoved + abs_std)",
        fontsize=12,
        weight="bold",
        color="#1f2937",
        y=0.995,
    )
    fig.subplots_adjust(left=0.115, right=0.995, top=0.955, bottom=0.015, wspace=0.045, hspace=0.16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return chosen


def render_selected(
    raw: np.ndarray,
    z: np.ndarray,
    repetition: int,
    source_paths: list[Path],
    asset_dir: Path,
    preview_path: Path,
    metadata_path: Path,
    subject: int,
    gesture: int,
) -> None:
    triplet = z[repetition]
    vmin, vmax = common_limits(triplet)
    asset_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    for view_idx, lab in enumerate(VIEW_LABELS):
        fig, ax = plt.subplots(figsize=(3.0, 1.65), dpi=320)
        draw_heatmap(ax, triplet[view_idx], vmin, vmax)
        fig.subplots_adjust(0, 0, 1, 1)
        out = asset_dir / f"fig2_real_cir_{lab}.png"
        fig.savefig(out, facecolor="white", bbox_inches=None, pad_inches=0)
        plt.close(fig)
        exported.append(str(out))

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.45), dpi=200)
    for idx, lab in enumerate(VIEW_LABELS):
        draw_heatmap(axes[idx], triplet[idx], vmin, vmax)
        axes[idx].set_title(f"{lab} view", fontsize=10, pad=5, color="#263442", weight="bold")
    fig.suptitle(
        f"HV{subject}, G{gesture}, repetition {repetition}: synchronized real CIR",
        fontsize=12,
        weight="bold",
        color="#1f2937",
        y=0.98,
    )
    fig.subplots_adjust(left=0.015, right=0.995, top=0.83, bottom=0.04, wspace=0.045)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_path, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    metadata = {
        "dataset": "UWB-Gestures (Figshare 12652592)",
        "subject": subject,
        "gesture": gesture,
        "repetition_zero_based": repetition,
        "view_order": list(VIEW_NAMES),
        "shape_per_view": [T, M],
        "data_type": "ClutterRemoved",
        "source_csv": [str(p) for p in source_paths],
        "preprocessing": "absolute value, then per-sample per-view z-score (abs_std)",
        "display": {
            "colormap": "magma",
            "interpolation": "nearest",
            "common_vmin_percentile": 2.0,
            "common_vmax_percentile": 99.75,
            "vmin": vmin,
            "vmax": vmax,
        },
        "raw_stats": [
            {
                "view": VIEW_NAMES[i],
                "min": float(raw[repetition, i].min()),
                "max": float(raw[repetition, i].max()),
                "mean": float(raw[repetition, i].mean()),
                "std": float(raw[repetition, i].std()),
            }
            for i in range(3)
        ],
        "exported_assets": exported,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\uwb_gestures\12652592"))
    parser.add_argument("--subject", type=int, default=5)
    parser.add_argument("--gesture", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=None)
    parser.add_argument("--candidate-count", type=int, default=6)
    parser.add_argument("--asset-dir", type=Path, default=Path(r"D:\uwb_gestures\paper1\figs\assets"))
    parser.add_argument("--work-dir", type=Path, default=Path(r"D:\uwb_gestures\paper1\tmp\pptx_fig2_cn"))
    args = parser.parse_args()

    paths = find_source_files(args.dataset_root, args.subject, args.gesture)
    raw = load_triplets(paths)
    z = abs_std(raw)
    scores = structure_scores(z)
    candidates_path = args.work_dir / "fig2_real_cir_candidates.png"
    chosen = render_candidates(z, scores, candidates_path, args.candidate_count)
    repetition = chosen[0] if args.repetition is None else args.repetition
    render_selected(
        raw,
        z,
        repetition,
        paths,
        args.asset_dir,
        args.work_dir / "fig2_real_cir_triplet_preview.png",
        args.work_dir / "fig2_real_cir_metadata.json",
        args.subject,
        args.gesture,
    )
    print(json.dumps({
        "candidates": chosen,
        "selected": repetition,
        "candidate_preview": str(candidates_path),
        "triplet_preview": str(args.work_dir / "fig2_real_cir_triplet_preview.png"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
