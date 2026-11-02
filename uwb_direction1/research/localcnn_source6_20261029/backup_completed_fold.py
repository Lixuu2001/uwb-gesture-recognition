"""Snapshot only an immutable completed fold; never decode prediction values.

The live training run is read-only. This is not a whole-job resume snapshot.
"""
import argparse
import hashlib
import json
import shutil
import tarfile
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def build(held, output):
    assert held in range(2,7)
    run = BASE/'run01'
    state = json.loads((run/'STATE.json').read_text())
    assert held in state['completed_folds']
    fold = run/f'held{held}'
    done = json.loads((fold/'FOLD_COMPLETE.json').read_text())
    assert done['held'] == held and done['epochs'] == 24
    assert sha(fold/'best.pt') == done['checkpoint_sha256']
    assert sha(fold/'predictions_unscored.npz') == done['prediction_sha256']
    frozen = json.loads((BASE/'FROZEN_HASHES.json').read_text())
    assert all(sha(BASE/p) == h for p,h in frozen.items())
    output.mkdir(exist_ok=False)
    staging = output/'payload'; staging.mkdir()
    members = []
    def copy(source, relative):
        target = staging/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha(source)
        shutil.copyfile(source, target)
        assert sha(source) == sha(target) == before, 'source changed during copy'
        members.append(dict(path=str(relative), bytes=target.stat().st_size, sha256=before))
    for name in ('train.py','launch.py','check_roles.py','PROTOCOL.json','PREFLIGHT.json',
                 'PREREGISTRATION.md','FROZEN_HASHES.json','FIRST_UPDATE_VERIFIED.json','LAUNCH.json',
                 'audit_complete.py','audit_launch.py','test_audit.py','AUDIT_HASHES.json',
                 'AUDIT_PREPARATION.json','AUDIT_TESTS.log','AUDIT_READINESS_CHECK.log',
                 'AUDIT_PREPARATION.md','backup_completed_fold.py'):
        copy(BASE/name, Path('study')/name)
    for name in ('IDENTITY.json','REUSED_HELD1.json'):
        copy(run/name, Path('run_metadata')/name)
    for path in sorted(fold.iterdir()):
        assert path.is_file() and path.suffix != '.tmp'
        copy(path, Path(f'held{held}')/path.name)
    copy(Path('/workspace/uwb-mvglf-20261029/model.py'), Path('dependencies/model.py'))
    copy(Path('/workspace/uwb-accuracy-loop-20261029/run01/source_ids.npz'), Path('dependencies/source_ids.npz'))
    copy(Path('/workspace/uwb-final-queue-backup-20261029/DRIVE_BACKUP_RECEIPT.json'), Path('dependencies/prior_final_queue_drive_receipt.json'))
    snapshot = dict(state='immutable_completed_fold_snapshot', held=held, epochs=24,
                    observed_unix=time.time(), live_state_at_start=state,
                    diagnostic_values_decoded=False, full_job_resume_snapshot=False,
                    current_in_progress_fold_included=False, members=members,
                    limitations='Only completed fold and prerequisites; raw data/source_maps use existing backups or deterministic regeneration. Other source folds and live optimizer state are not included. Never restart or duplicate the live worker from this bundle.')
    (staging/'MEMBERS.json').write_text(json.dumps(snapshot, indent=2))
    archive = output/f'localcnn_source6_held{held}.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(staging.rglob('*')):
            if path.is_file(): tar.add(path, arcname=str(path.relative_to(staging)), recursive=False)
    chunks = []
    with archive.open('rb') as handle:
        for i in range(1000):
            block = handle.read(7*1024*1024)
            if not block: break
            target = output/f'localcnn_source6_held{held}.part{i:03d}'
            target.write_bytes(block)
            chunks.append(dict(name=target.name, bytes=len(block), sha256=sha(target)))
    manifest = dict(archive_name=archive.name, archive_bytes=archive.stat().st_size,
                    archive_sha256=sha(archive), tracked_payload_files=len(members),
                    members_manifest_sha256=sha(staging/'MEMBERS.json'), chunks=chunks,
                    held=held, full_job_resume_snapshot=False)
    (output/'MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--held', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    build(args.held, args.out)
