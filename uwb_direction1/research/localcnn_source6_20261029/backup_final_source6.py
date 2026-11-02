"""Recoverable completed sixfold set: final fold + final metadata + prior indices.

Raw inputs and regenerable source maps excluded. Previous models are referenced
through independently verified private backup receipts, not silently omitted.
"""
import argparse
import hashlib
import json
import shutil
import tarfile
import time
from pathlib import Path

BASE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def build(output):
    run=BASE/'run01'
    assert json.loads((run/'STATE.json').read_text())['state']=='complete'
    assert json.loads((BASE/'COMPLETE_AUDIT.json').read_text())['state']=='pass'
    for folder in ('supervisor01','audit_supervisor01'):
        receipt=json.loads((BASE/folder/'TERMINATION.json').read_text())
        assert receipt['exit_code']==0 and not receipt['wall_timeout']
    for name,expected in json.loads((BASE/'FROZEN_HASHES.json').read_text()).items():
        assert sha(BASE/name)==expected
    output.mkdir(exist_ok=False);payload=output/'payload';payload.mkdir()
    members=[]
    def copy(source,name):
        dest=payload/name;dest.parent.mkdir(parents=True,exist_ok=True)
        before=sha(source);shutil.copyfile(source,dest)
        assert sha(source)==sha(dest)==before
        members.append(dict(path=str(name),bytes=dest.stat().st_size,sha256=before))
    names=['train.py','launch.py','check_roles.py','PROTOCOL.json','PREFLIGHT.json',
           'PREREGISTRATION.md','FROZEN_HASHES.json','FIRST_UPDATE_VERIFIED.json','LAUNCH.json',
           'audit_complete.py','audit_launch.py','test_audit.py','AUDIT_HASHES.json',
           'AUDIT_PREPARATION.json','AUDIT_TESTS.log','AUDIT_READINESS_CHECK.log',
           'AUDIT_PREPARATION.md','backup_completed_fold.py','backup_final_source6.py',
           'COMPLETE_AUDIT.json','FINAL_REPORT.md']
    for name in names:copy(BASE/name,Path('study')/name)
    for name in ('STATE.json','SUMMARY.json','IDENTITY.json','REUSED_HELD1.json',
                 'ALL_NEW_PREDICTIONS_FROZEN.json','FILES.json'):
        copy(run/name,Path('run_metadata')/name)
    for held in range(2,7):
        for path in sorted((run/f'held{held}').iterdir()):
            assert path.is_file() and path.suffix!='.tmp'
            if held==6 or path.suffix!='.pt':copy(path,Path(f'held{held}')/path.name)
    for folder in ('supervisor01','audit_supervisor01'):
        for name in ('TERMINATION.json','process.log'):
            copy(BASE/folder/name,Path(folder)/name)
    copy(Path('/workspace/uwb-mvglf-20261029/model.py'),Path('dependencies/model.py'))
    copy(Path('/workspace/uwb-accuracy-loop-20261029/run01/source_ids.npz'),Path('dependencies/source_ids.npz'))
    copy(Path('/workspace/uwb-final-queue-backup-20261029/DRIVE_BACKUP_RECEIPT.json'),Path('prior_backups/held1_and_mini6_final_queue.json'))
    prior={}
    for held in range(2,6):
        folder=BASE/f"backup_held{held}_{'20261030' if held==5 else '20261029'}"
        receipt=json.loads((folder/'DRIVE_BACKUP_RECEIPT.json').read_text())
        assert receipt['state']=='private_readback_verified'
        assert receipt['manifest']['held']==held and receipt['all_tracked_payload_members_verified']==31
        prior[str(held)]=dict(receipt=f'prior_backups/held{held}.json',archive_sha256=receipt['manifest']['archive_sha256'])
        copy(folder/'DRIVE_BACKUP_RECEIPT.json',Path(f'prior_backups/held{held}.json'))
    description=dict(state='completed_source6_recoverable_composite',observed_unix=time.time(),members=members,
                     current_archive_models=[6],prior_model_archives=prior,
                     held1_and_mini6='prior_backups/held1_and_mini6_final_queue.json',
                     full_set_requires_prior_archives=True,raw_radar_arrays_included=False,
                     source_maps_included=False,source_maps_regeneration='Original signed-map bilinear64x128 then per-record/view RMS; exact original model.py, source IDs, code, cache/input hashes preserved.',
                     restore_note='Restore the original input backup and this archive; resolve private prior indices for held1 and held2-5/mini6 tensors. All trajectories are completed, no training restart required. Never claim this archive alone contains every tensor.')
    (payload/'MEMBERS.json').write_text(json.dumps(description,indent=2))
    archive=output/'source6_final.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(payload.rglob('*')):
            if path.is_file():tar.add(path,arcname=str(path.relative_to(payload)),recursive=False)
    chunks=[]
    with archive.open('rb') as stream:
        for i in range(1000):
            data=stream.read(7*1024*1024)
            if not data:break
            dest=output/f'source6_final.part{i:03d}';dest.write_bytes(data)
            chunks.append(dict(name=dest.name,bytes=len(data),sha256=sha(dest)))
    manifest=dict(archive_name=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),
                  tracked_payload_files=len(members),members_manifest_sha256=sha(payload/'MEMBERS.json'),
                  chunks=chunks,full_set_requires_prior_archives=True,models_in_this_archive=[6])
    (output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    build(parser.parse_args().out)
