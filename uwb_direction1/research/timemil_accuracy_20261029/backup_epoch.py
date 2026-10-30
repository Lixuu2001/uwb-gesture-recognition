"""Preserve a consistent completed-epoch snapshot without touching the worker.

No input arrays, labels, predictions, model forward passes or training updates
are evaluated. This is a recovery-material snapshot, not an automatic resume
entrypoint or a completion/accuracy audit.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import tarfile
import time
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import torch

BASE = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def finite(value):
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True

def build(output):
    torch.set_num_threads(1)
    cfg = json.loads((BASE / 'PROTOCOL.json').read_text())
    run = BASE / cfg['run_directory']
    frozen = json.loads((BASE / 'FROZEN_HASHES.json').read_text())
    audit = json.loads((BASE / 'AUDIT_HASHES.json').read_text())
    for name, expected in {**frozen, **audit}.items():
        assert sha(BASE / name) == expected, name
    live = json.loads((run / 'STATE.json').read_text())
    assert live['arm'] == 'timemil' and not live['completed_arms']
    assert not live['diagnostic_labels_read'] and not live['users7_8_labels_used']
    assert live['state'] in ('training', 'epoch_complete')
    output.mkdir(exist_ok=False)
    payload = output / 'payload'
    payload.mkdir()
    members = []

    def copy(source, name, stable=True):
        dest = payload / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        before = sha(source) if stable else None
        # Atomic checkpoint replacement means the opened file is a full saved
        # version; last/best epoch and log consistency are checked below.
        shutil.copyfile(source, dest)
        if stable:
            assert sha(source) == sha(dest) == before, str(source)
        return dest

    attempts = []
    for attempt in range(1, 4):
        last_path = copy(run / 'timemil/last.pt', 'run02/timemil/last.pt', False)
        best_path = copy(run / 'timemil/best.pt', 'run02/timemil/best.pt', False)
        last = torch.load(last_path, map_location='cpu', weights_only=False)
        best = torch.load(best_path, map_location='cpu', weights_only=False)
        rows = [json.loads(s) for s in (run / 'timemil/epochs.jsonl').read_text().splitlines() if s.strip()]
        epoch = last['epoch']
        rows = [r for r in rows if r['epoch'] <= epoch]
        aligned = (len(rows) == epoch and [r['epoch'] for r in rows] == list(range(1, epoch + 1))
                   and best['epoch'] <= epoch and best['epoch'] == -int(last['best'][2]))
        attempts.append(dict(attempt=attempt, last_epoch=epoch, best_epoch=best['epoch'], aligned=aligned))
        (output / 'SNAPSHOT_ATTEMPTS.json').write_text(json.dumps(attempts, indent=2))
        if aligned:
            break
    assert aligned, 'Snapshot raced an epoch boundary; preserve attempts, do not alter worker'
    assert last['identity']['frozen_hashes'] == best['identity']['frozen_hashes'] == frozen
    assert finite(last['model']) and finite(last['optimizer']) and finite(best['model'])
    steps = epoch * math.ceil(cfg['counts']['train'] / cfg['batch_size'])
    assert last['optimizer']['count'] == steps
    assert set(int(v['step']) for v in last['optimizer']['fast']['state'].values()) == {steps}
    assert last['optimizer']['alpha'] == .5 and last['optimizer']['k'] == 6
    assert last['optimizer']['slow']
    assert {'torch_rng', 'numpy_rng', 'python_rng'} <= set(last)
    from adapter import make_model, StableLookahead
    model = make_model('timemil', cfg['seed'])
    optimizer = StableLookahead(model.parameters())
    model.load_state_dict(last['model'])
    optimizer.load_state_dict(last['optimizer'])
    assert optimizer.count == steps and finite(optimizer.state_dict())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in last['model'].items())
    assert all(torch.equal(v, optimizer.slow[k]) for k, v in last['optimizer']['slow'].items())
    assert all(getattr(model, k).data_ptr() == getattr(model, k + '_').data_ptr() for k in ('wave1', 'wave2', 'wave3'))
    (payload / 'run02/timemil/epochs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    for name in ('initial.pt', 'first_update.pt'):
        copy(run / 'timemil' / name, Path('run02/timemil') / name)
    for name in ('IDENTITY.json', 'split.json', 'SOURCE_CACHE.json'):
        copy(run / name, Path('run02') / name)
    (payload / 'run02/STATE_CONTEXT.json').write_text(json.dumps(live, indent=2))
    for path in sorted(BASE.iterdir()):
        if path.is_file() and path.suffix in ('.py', '.json', '.md', '.txt', '.log'):
            copy(path, Path('study') / path.name)
    for folder in ('attempt01_predata_failure', 'supervisor01'):
        for path in sorted((BASE / folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                copy(path, Path(folder) / path.relative_to(BASE / folder))
    for path in sorted((BASE / 'vendor').rglob('*')):
        if path.is_file() and '.git' not in path.parts and '__pycache__' not in path.parts:
            copy(path, Path('vendor') / path.relative_to(BASE / 'vendor'))
    copy(Path(cfg['source_manifest']), Path('dependencies/source_ids.npz'))
    assert sha(payload / 'dependencies/source_ids.npz') == cfg['source_manifest_sha256']
    for key in ('previous_controller_state', 'previous_multirocket_audit', 'previous_cnn_summary'):
        copy(Path(cfg[key]), Path('dependencies') / (key + '.json'))
    copy(BASE / '/workspace/uwb-multirocket-20261029/launch.py', Path('dependencies/independent_supervisor.py'))
    copy(Path(cfg['data']) / 'RESTORE_VERIFIED.json', Path('dependencies/RESTORE_VERIFIED.json'))
    copy(Path('/workspace/uwb-final-queue-backup-20261029/DRIVE_BACKUP_RECEIPT.json'), Path('dependencies/prior_cnn_minir_multir_backup.json'))
    copy(output / 'SNAPSHOT_ATTEMPTS.json', Path('snapshot/SNAPSHOT_ATTEMPTS.json'))
    for path in sorted(payload.rglob('*')):
        if path.is_file():
            members.append(dict(path=str(path.relative_to(payload)), bytes=path.stat().st_size, sha256=sha(path)))
    receipt = dict(state='completed_epoch_snapshot_local_verified', observed_unix=time.time(),
                   run_directory=cfg['run_directory'], arm='timemil', completed_epoch=epoch,
                   selected_epoch_at_snapshot=best['epoch'], optimizer_steps=steps,
                   members=members, current_frozen_hashes_verified=len(frozen), audit_hashes_verified=len(audit),
                   finite_model_and_optimizer=True, model_and_optimizer_load_verified=True,
                   wavelet_storage_aliases_preserved=True, rng_states_preserved=True,
                   diagnostic_labels_read=False, users7_8_labels_used=False,
                   real_training_updates_performed_by_backup=0, real_model_forward_passes_performed_by_backup=0,
                   raw_radar_arrays_included=False, source_maps_included=False, gap_arm_included=False,
                   scientific_protocol_changed=False, original_failed_attempt_preserved=True,
                   recovery_limit='Recovery materials only. Current run_pilot.py intentionally refuses existing run directories. No production resume entrypoint is implemented/tested by this snapshot. If interrupted, inspect receipts and implement same-trajectory epoch-boundary recovery under remaining cumulative compute; do not restart/resample or select a new configuration.',
                   input_recovery_index='GitHub uwb_direction1/migration/20260930/CLOUD_DOWNLOAD_INDEX.json; source maps regenerate with original RMS/reshape and exact IDs.',
                   scope='Partial TimeMIL checkpoint preservation; neither full pair completion, diagnostic accuracy audit nor originality evidence.')
    (payload / 'MEMBERS.json').write_text(json.dumps(receipt, indent=2))
    archive = output / 'timemil_run02_partial.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(payload.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(payload)), recursive=False)
    chunks = []
    with archive.open('rb') as stream:
        for i in range(1000):
            data = stream.read(7 * 1024 * 1024)
            if not data:
                break
            dest = output / f'timemil_run02_partial.part{i:03d}'
            dest.write_bytes(data)
            chunks.append(dict(name=dest.name, bytes=len(data), sha256=sha(dest)))
    manifest = {k: v for k, v in receipt.items() if k != 'members'}
    manifest.update(archive_name=archive.name, archive_bytes=archive.stat().st_size,
                    archive_sha256=sha(archive), tracked_payload_files=len(members),
                    members_manifest_sha256=sha(payload / 'MEMBERS.json'), chunks=chunks)
    (output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))

def verify_readback(output):
    manifest = json.loads((output / 'MANIFEST.json').read_text())
    indices = json.loads((output / 'READBACK_INPUT.json').read_text())
    assert len(indices) == len(manifest['chunks'])
    archive = output / 'remote_reassembled.tar.gz'
    with archive.open('wb') as stream:
        for chunk, index in zip(manifest['chunks'], indices):
            source = Path(index['path'])
            assert index['name'] == chunk['name']
            assert source.stat().st_size == chunk['bytes'] and sha(source) == chunk['sha256']
            stream.write(source.read_bytes())
    assert archive.stat().st_size == manifest['archive_bytes'] and sha(archive) == manifest['archive_sha256']
    with tarfile.open(archive) as tar:
        members_data = tar.extractfile('MEMBERS.json').read()
        assert hashlib.sha256(members_data).hexdigest() == manifest['members_manifest_sha256']
        members = json.loads(members_data)['members']
        expected = {r['path'] for r in members} | {'MEMBERS.json'}
        assert set(tar.getnames()) == expected and len(tar.getmembers()) == len(expected)
        for row in members:
            data = tar.extractfile(row['path']).read()
            assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], row['path']
    result = dict(state='remote_archive_and_all_members_verified', archive_sha256=manifest['archive_sha256'],
                  chunks_verified=len(indices), tracked_payload_members_verified=len(members),
                  archive_members=len(expected), completed_epoch=manifest['completed_epoch'])
    (output / 'READBACK_VERIFIED.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'verify-readback'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    (build if args.action == 'build' else verify_readback)(args.out)
