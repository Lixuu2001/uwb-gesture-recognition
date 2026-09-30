"""Resume the formal first-part queue with independent fold workers.

Each worker owns one LOSO fold, so output files never overlap.  The stage
boundaries remain serial: teachers are launched only after all baselines are
complete, and KD only after both teachers have written all-logit archives.
The training arguments are kept identical to suite.py (including seed and
microbatch) and this script only changes scheduling, not the protocol.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results_clean"
PY = sys.executable


def done(tag: str, seed: int, fold: int, teacher: bool = False) -> bool:
    j = RESULTS / f"{tag}_s{seed}_h{fold}.json"
    p = RESULTS / f"{tag}_s{seed}_h{fold}_predictions.npz"
    if not (j.exists() and p.exists()):
        return False
    if teacher and not (RESULTS / f"{tag}_s{seed}_h{fold}_all_logits.npz").exists():
        return False
    try:
        obj = json.loads(j.read_text(encoding="utf-8"))
        return obj.get("held_subject") == fold and obj.get("seed") == seed
    except Exception:
        return False


def run_stage(tag: str, mode: str, views: list[int], seed: int = 2026,
              workers: int = 2, readout: str | None = None,
              teacher: str | None = None, save_teacher: bool = False,
              aux: str | None = None) -> None:
    pending = []
    for fold in range(1, 9):
        if not done(tag, seed, fold, teacher=save_teacher):
            pending.append(fold)
    if not pending:
        print("STAGE_ALREADY_COMPLETE", tag, seed, flush=True)
        return
    print("STAGE_START", tag, "folds", pending, "workers", workers, flush=True)
    active: dict[int, tuple[subprocess.Popen, object]] = {}
    while pending or active:
        while pending and len(active) < workers:
            fold = pending.pop(0)
            cmd = [PY, "-u", str(BASE / "train.py"), "--tag", tag,
                   "--mode", mode, "--views", *map(str, views),
                   "--seed", str(seed), "--folds", str(fold),
                   "--microbatch", "32"]
            if readout:
                cmd += ["--readout", readout]
            if teacher:
                cmd += ["--teacher", teacher]
            if save_teacher:
                cmd += ["--save-teacher"]
            if aux:
                cmd += ["--aux", aux]
            lp = RESULTS / f"{tag}_seed{seed}_h{fold}.parallel.log"
            fh = lp.open("a", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=BASE, stdout=fh, stderr=subprocess.STDOUT)
            active[fold] = (proc, fh)
            print("DISPATCH_FOLD", tag, fold, seed, flush=True)
        finished = []
        for fold, (proc, fh) in list(active.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            if rc != 0:
                raise RuntimeError(f"{tag} fold {fold} failed rc={rc}; see parallel log")
            if not done(tag, seed, fold, teacher=save_teacher):
                raise RuntimeError(f"{tag} fold {fold} exited without complete artifacts")
            print("FINISHED_FOLD", tag, fold, seed, flush=True)
            finished.append(fold)
        for fold in finished:
            active.pop(fold, None)
        if active and not finished:
            import time
            time.sleep(1.0)
    print("STAGE_COMPLETE", tag, seed, flush=True)


def main() -> None:
    # Finish the view-specific controls that were in flight when the serial
    # queue was stopped, then run the teacher/KD dependency chain.
    for v, name in [(2, "R")]:
        run_stage(f"sv_{name}_only", "only", [v])
        run_stage(f"sv_{name}_concat", "concat", [v])
    run_stage("sv_L_signed", "signed", [0])

    for readout in ("mean", "concat"):
        run_stage(f"teacher_{readout}", "off", [0, 1, 2],
                  readout=readout, save_teacher=True)

    for v, name in enumerate(("L", "T", "R")):
        run_stage(f"sv_{name}_kd", "off", [v],
                  teacher="teacher_mean,teacher_concat")
    print("CORE_QUEUE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
