"""Copy a coherent committed producer/replay state without touching live training.

Only immutable committed journals, an atomic checkpoint slot, bounded log cuts,
private IDs and lineage are copied. No radar values/labels are loaded or scored.
"""
from pathlib import Path
import json,hashlib,shutil,os,time,sys
ROOT=Path('/workspace/uwb-softvote-source6-real-20261103')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def copy(src,dest):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
def cut_log(src,dest,p):
    with Path(src).open('rb') as f:
        size=os.fstat(f.fileno()).st_size;raw=f.read(size)
    assert len(raw)==size and size>=p['prefix_bytes']
    prefix=raw[:p['prefix_bytes']]
    assert hashlib.sha256(prefix).hexdigest()==p['prefix_sha256'] and len(prefix.splitlines())==p['updates']
    end=raw.rfind(b'\n')+1;complete=raw[:end]
    rows=complete.splitlines();assert len(rows)>=p['updates']
    for i,line in enumerate(rows):assert read_line(line)['update']==i+1
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(complete)
    # The exact byte cut is retained even if its last append was incomplete.
    dest.with_name('steps_raw_cut.bin').write_bytes(raw)
    return dict(captured_bytes=size,complete_rows=len(rows),partial_tail_bytes=size-end,
        committed_updates=p['updates'],uncommitted_complete_rows=len(rows)-p['updates'],original_raw_cut_sha256=hashlib.sha256(raw).hexdigest())
def read_line(line):return json.loads(line)
def capture_journal(src,dest,producer=True):
    src=Path(src);dest=Path(dest);p=read(src/'PROGRESS.json');dest.mkdir(parents=True,exist_ok=False)
    copy(src/p['slot'],dest/p['slot']);assert sha(dest/p['slot'])==p['checkpoint_sha256'],'Atomic slot changed; reject this capture, never train again'
    previous=None
    for epoch in range(p['epoch']+1):
        for name in [f'epoch{epoch:03}.json',f'journal{epoch:03}.json']:copy(src/name,dest/name)
        j=read(dest/f'journal{epoch:03}.json');r=read(dest/f'epoch{epoch:03}.json')
        assert j['epoch']==r['epoch']==epoch and j['previous_journal_sha256']==previous
        assert j['record_sha256']==sha(dest/f'epoch{epoch:03}.json') and j['state_sha256']==r['state_sha256']
        previous=sha(dest/f'journal{epoch:03}.json')
        if producer and epoch:copy(src/f'validation_epoch{epoch:03}.npz',dest/f'validation_epoch{epoch:03}.npz')
    assert previous==p['journal_sha256'] and j['checkpoint_sha256']==p['checkpoint_sha256']
    log=cut_log(src/'steps.jsonl',dest/'steps.jsonl',p) if producer else None
    # Fail closed if a new commit was published while this cut was assembled.
    assert read(src/'PROGRESS.json')==p,'Progress advanced during cut; preserve failed attempt and capture a fresh one'
    write(dest/'PROGRESS.json',p)
    return dict(progress=p,step_log=log,checkpoint_sha256=sha(dest/p['slot']))
def capture_producer(src,dest):
    src=Path(src);dest=Path(dest)
    if (src/'ARM_COMPLETE.json').exists():
        complete=read(src/'ARM_COMPLETE.json');dest.mkdir(parents=True,exist_ok=False)
        for n,h in complete['files'].items():copy(src/n,dest/n);assert sha(dest/n)==h,n
        copy(src/'ARM_COMPLETE.json',dest/'ARM_COMPLETE.json');return dict(state='complete_immutable_member',progress=read(dest/'PROGRESS.json'))
    report=capture_journal(src,dest)
    for n in ['initial.pt','first_update.pt','split.npz','FIRST_READY.json','FIRST_RELEASE.json','first_audit/CERTIFICATE.json','first_audit/TERMINATION.json']:
        copy(src/n,dest/n)
    cert=read(dest/'first_audit/CERTIFICATE.json');term=read(dest/'first_audit/TERMINATION.json')
    assert cert['state']=='pass' and term['exit_code']==0 and term['resource_stop_reason'] is None and term['cgroup_OOM_kill_delta']==0
    assert cert['first_checkpoint_sha256']==sha(dest/'first_update.pt')
    report['state']='coherent_running_member_checkpoint_and_exact_log_cut';return report
def main():
    assert len(sys.argv)==3;held=int(sys.argv[1]);assert held in [2,3,4,5,6];out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
    sys.path.insert(0,str(ROOT));from common import require_initial,alive
    require_initial();identity=read(ROOT/'CONTROLLER_IDENTITY.json');assert alive(identity),'Inspect stopped execution before capture/resume'
    report={}
    for arm in ['timemil','shuffle']:
        src=ROOT/'folds'/f'held{held}'/arm;dest=out/'folds'/f'held{held}'/arm;report[arm]=capture_producer(src,dest)
        if (src/'replay/PROGRESS.json').exists():report[arm]['own_replay']=capture_journal(src/'replay',dest/'replay',producer=False)
    for name in ['CONFIG.json','METHOD_LOCK.json','FROZEN_HASHES.json','REMOTE_REAL_PREREGISTRATION.json','INITIAL_PRIVATE_INSURANCE.json','CONTROLLER_IDENTITY.json','RESOURCE_LEDGER.json','STATE.json','LAUNCH.json']:
        copy(ROOT/name,out/name)
    for name in ['MANIFEST.json','CLOUD_DOWNLOAD_INDEX.json','DRIVE_BACKUP_RECEIPT.json','READBACK_VERIFIED.json']:copy(ROOT/'initial_backup'/name,out/'initial_lineage'/name)
    for path in (ROOT/'attempts').rglob('*'):
        if path.is_file() and (path.name in ['START.json','PROCESSES.json'] or path.name.endswith('_TERM.json')):copy(path,out/'attempts'/path.relative_to(ROOT/'attempts'))
    write(out/'SNAPSHOT.json',dict(state='coherent_committed_checkpoint_overlay_captured_without_stopping_live_training',source=str(ROOT),held=held,members=report,captured_unix=time.time(),initial_archive_required=True,private_actual_IDs_included=True,raw_radar_arrays_included=False,diagnostic_labels_read=False,users7_8_labels_used=False,new_model_fits_started=False,independent_full_training_replay_performed=False))
    write(out/'PAYLOAD_HASHES.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps(dict(state='snapshot_cut_complete_needs_native_state_verification_and_private_raw_readback',members={arm:report[arm]['progress']['epoch'] for arm in report})))
if __name__=='__main__':main()
