#!/usr/bin/env python3
"""Audit whether known exact triplicate duplicates cross participant IDs.

This reads only the participant vector and the duplicate index produced by the
label-free paired-time screen. It does not load gesture labels or radar arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="Restored cloud-data root containing cache/.")
    ap.add_argument("--diagnostic-summary", type=Path, required=True,
                    help="SUMMARY.json from the frozen paired-time diagnostic.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    subject_path = args.data / "cache" / "subj_clutterremoved.npy"
    subjects = np.load(subject_path, mmap_mode="r", allow_pickle=False)
    if subjects.ndim != 1 or subjects.dtype.kind not in "iu":
        raise ValueError(f"expected a 1-D integer subject vector, got {subjects.shape}/{subjects.dtype}")

    summary = json.loads(args.diagnostic_summary.read_text(encoding="utf-8"))
    rows = summary["duplicate_checks"]
    pairs: list[dict[str, int | bool]] = []
    relation_counts: Counter[tuple[int, int]] = Counter()
    for row in rows:
        a = int(row["original_id"])
        b = int(row["duplicate_id"])
        if not (0 <= a < len(subjects) and 0 <= b < len(subjects)):
            raise ValueError(f"duplicate index out of bounds: {a}, {b}")
        sa, sb = int(subjects[a]), int(subjects[b])
        relation_counts[(sa, sb)] += 1
        pairs.append({"original_id": a, "duplicate_id": b,
                      "original_subject": sa, "duplicate_subject": sb,
                      "cross_subject": sa != sb})

    cross = [p for p in pairs if p["cross_subject"]]
    output = {
        "analysis": "exact-duplicate participant-overlap audit",
        "scope": "participant IDs and previously verified exact triplet duplicate indices only",
        "gesture_labels_accessed": False,
        "radar_arrays_loaded": False,
        "model_fitting": False,
        "subject_vector": {
            "path": str(subject_path),
            "shape": list(subjects.shape),
            "dtype": str(subjects.dtype),
            "sha256": sha256(subject_path),
            "participant_counts": {
                str(int(k)): int(v)
                for k, v in zip(*np.unique(subjects, return_counts=True))
            },
        },
        "duplicate_pair_count": len(pairs),
        "duplicate_subject_relations": [
            {"original_subject": a, "duplicate_subject": b, "pairs": n}
            for (a, b), n in sorted(relation_counts.items())
        ],
        "cross_subject_duplicate_pair_count": len(cross),
        "cross_subject_examples": cross[:10],
        "interpretation": (
            "These verified exact duplicate triplets cannot cross a participant-disjoint split. "
            "They can still contaminate a random sample-level split if duplicate members are split apart."
            if not cross else
            "At least one exact duplicate triplet crosses participant IDs; any participant-independent "
            "evaluation must report and resolve that overlap before scoring."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({k: output[k] for k in (
        "duplicate_pair_count", "duplicate_subject_relations",
        "cross_subject_duplicate_pair_count", "gesture_labels_accessed",
        "radar_arrays_loaded", "model_fitting")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
