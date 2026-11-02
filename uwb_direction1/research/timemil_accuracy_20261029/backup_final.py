"""Preserve the completed, independently audited TimeMIL/GAP pair privately."""
import argparse
import json
import shutil
import tarfile
import time
from pathlib import Path
from backup_epoch import sha
from verify_final_readback import verify_readback

BASE = Path(__file__).resolve().parent

def build(output):
    cfg = json.loads((BASE/'PROTOCOL.json').read_text())
    run = BASE/cfg['run_directory']
    assert json.loads((run/'STATE.json').read_text())['state'] == 'complete'
    audit = json.loads((BASE/'COMPLETE_RESULT_AUDIT.json').read_text())
    assert audit['state'] == 'pass'
    for folder in (cfg['supervisor_directory'], 'audit_supervisor01'):
        receipt = json.loads((BASE/folder/'TERMINATION.json').read_text())
        assert receipt['exit_code'] == 0 and not receipt['wall_timeout']
    cumulative = json.loads((BASE/'CUMULATIVE_TERMINATION.json').read_text())
    assert cumulative['cumulative_wall_seconds'] <= cfg['wall_budget_seconds']
    assert cumulative['cumulative_cpu_seconds'] <= cfg['cpu_budget_seconds']
    for index in ('FROZEN_HASHES.json', 'AUDIT_HASHES.json'):
        for name, expected in json.loads((BASE/index).read_text()).items():
            assert sha(BASE/name) == expected, name
    for name, expected in audit['artifact_hashes'].items():
        assert sha(Path(name)) == expected, name
    output.mkdir(exist_ok=False)
    payload = output/'payload'
    payload.mkdir()
    members = []
    def copy(source, name):
        dest = payload/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        before = sha(source)
        shutil.copyfile(source, dest)
        assert sha(source) == sha(dest) == before
        members.append(dict(path=str(name), bytes=dest.stat().st_size, sha256=before))
    for path in sorted(BASE.iterdir()):
        if path.is_file() and path.suffix in ('.py','.json','.md','.txt','.log'):
            copy(path, Path('study')/path.name)
    for folder in (cfg['run_directory'], cfg['supervisor_directory'], 'audit_supervisor01',
                   'attempt01_predata_failure', 'supervisor01', 'vendor'):
        for path in sorted((BASE/folder).rglob('*')):
            if (path.is_file() and path.name not in ('source_maps.npy','worker.lock')
                and path.suffix not in ('.tmp','.pyc') and '.git' not in path.parts
                and '__pycache__' not in path.parts):
                copy(path, Path(folder)/path.relative_to(BASE/folder))
    for key in ('source_manifest','previous_controller_state','previous_multirocket_audit','previous_cnn_summary'):
        source = Path(cfg[key])
        copy(source, Path('dependencies')/(key+source.suffix))
    copy(Path('/workspace/uwb-multirocket-20261029/launch.py'), Path('dependencies/independent_supervisor.py'))
    copy(Path(cfg['data'])/'RESTORE_VERIFIED.json', Path('dependencies/RESTORE_VERIFIED.json'))
    copy(Path('/workspace/uwb-final-queue-backup-20261029/DRIVE_BACKUP_RECEIPT.json'), Path('dependencies/prior_cnn_minir_multir_backup.json'))
    copy(BASE/'partial_backup_20261030/DRIVE_BACKUP_RECEIPT.json', Path('dependencies/prior_partial_epoch43_backup.json'))
    description = dict(state='complete_audited_pair', observed_unix=time.time(), members=members,
                       arms=['timemil','gap'], epochs_per_arm=64, all_model_optimizer_rng_states_included=True,
                       private_sample_ids_and_predictions_included=True, raw_radar_arrays_included=False,
                       source_maps_included=False, all_current_pair_models_in_this_archive=True,
                       input_recovery='Persistent GitHub migration/20260930/CLOUD_DOWNLOAD_INDEX.json; original exact IDs/RMS/reshape regenerate source maps.',
                       scope='Completed exploratory published-method transfer; no independent validation or originality proof.')
    (payload/'MEMBERS.json').write_text(json.dumps(description,indent=2))
    archive = output/'timemil_run02_final.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(payload.rglob('*')):
            if path.is_file():
                tar.add(path,arcname=str(path.relative_to(payload)),recursive=False)
    chunks = []
    with archive.open('rb') as stream:
        for i in range(1000):
            data = stream.read(7*1024*1024)
            if not data: break
            dest = output/f'timemil_run02_final.part{i:03d}'
            dest.write_bytes(data)
            chunks.append(dict(name=dest.name,bytes=len(data),sha256=sha(dest)))
    manifest = {k:v for k,v in description.items() if k!='members'}
    manifest.update(archive_name=archive.name,archive_bytes=archive.stat().st_size,
                    archive_sha256=sha(archive),tracked_payload_files=len(members),
                    members_manifest_sha256=sha(payload/'MEMBERS.json'),chunks=chunks)
    (output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['build','verify-readback'])
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    (build if args.action=='build' else verify_readback)(args.out)
