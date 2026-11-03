"""Private committed replay checkpoint only; never certifies full100 or scores."""
from pathlib import Path
import argparse,datetime,json,os,signal,sys,time

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    b=Path(parser.parse_args().root).resolve();sys.path.insert(0,str(b))
    from trial_common import ready,read,write,sha
    from stream_archive import build
    c=ready()
    from audit_journal import load
    from adam_engine import semantic
    pid=read(b/'audit_supervisor01/PID.json')['pid'];proc=Path(f'/proc/{pid}')
    stat=proc.joinpath('stat').read_text().split();ticks=stat[21]
    assert stat[3]==str(read(b/'STATE.json')['pid'])
    assert str(b/'audit_full.py').encode() in proc.joinpath('cmdline').read_bytes().split(b'\0')
    assert not (b/'audit_supervisor01/TERMINATION.json').exists()
    assert not (b/'COMPLETE_AUDIT.json').exists() and not (b/'run01/SUMMARY.json').exists()
    paused=False;target=None;started=time.monotonic()
    try:
        os.kill(pid,signal.SIGSTOP);paused=True
        for _ in range(100):
            if proc.joinpath('stat').read_text().split()[2]=='T':break
            time.sleep(.02)
        else:raise RuntimeError('Auditor failed to stop for consistent serialization')
        state=load(b/'audit01',b/'run01');assert state is not None and 1<=state['epoch']<100
        progress=read(b/'audit01/PROGRESS.json');epoch=progress['epoch']
        assert state['updates']==235*epoch
        target=b/f'audit_insurance_epoch{epoch:03d}';assert not target.exists()
        scope=f'incomplete_independent_Adam100_replay_checkpoint_epoch{epoch:03d}'
        info=dict(state='incomplete_replay_state_privately_preserved_not_full_certification',
            scope=scope,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            auditor_pid=pid,auditor_start_ticks=ticks,committed_epoch=epoch,
            verified_prefix_updates=235*epoch,checkpoint_state_sha256=semantic(state),
            checkpoint_slot=progress['slot'],checkpoint_sha256=progress['checkpoint_sha256'],
            last_journal_sha256=progress['journal_sha256'],frozen_sha256=sha(b/'FROZEN_HASHES.json'),
            full100_replay_complete=False,full_RMS_pixels_audit_complete=False,
            diagnostic_labels_scored=False,users7_8_labels_used=False,
            automatic_resume_authorized=False,producer_checkpoint_used_as_audit_state=False)
        note=b/f'AUDIT_INSURANCE_EPOCH{epoch:03d}.json';assert not note.exists();write(note,info)
        files={'study/'+n:b/n for n in read(b/'FROZEN_HASHES.json')}
        for n in ['FROZEN_HASHES.json','REMOTE_REAL_PREREGISTRATION.json']:
            files['study/'+n]=b/n
        files['audit01/PROGRESS.json']=b/'audit01/PROGRESS.json'
        files['audit01/'+progress['slot']]=b/'audit01'/progress['slot']
        for i in range(1,epoch+1):
            files[f'audit01/epoch{i:03d}.json']=b/'audit01'/f'epoch{i:03d}.json'
            files[f'producer_epoch_records/epoch{i:03d}.json']=b/'run01'/f'epoch{i:03d}.json'
        files['producer/FILES.json']=b/'run01/FILES.json'
        files['PRESERVATION_STATE.json']=note
        files['operations/preserve_active_audit.py']=Path(__file__).resolve()
        m=build(target,f'adam100_independent_replay_epoch{epoch:03d}_20261103',files,scope,
                chunk_size=7340032,limit=67108864)
        m.update(state='partial_committed_audit_checkpoint_insurance_streamed',
            complete_modelset=False,full100_replay_complete=False,numeric_audit_certified=False)
        write(target/'MANIFEST.json',m)
        print(json.dumps(dict(directory=str(target),epoch=epoch,verified_updates=235*epoch,
            parts=len(m['chunks']),archive_bytes=m['archive_bytes'],payload_members=m['tracked_payload_files'],
            full_numeric_audit_claimed=False)))
    finally:
        if paused:
            assert proc.joinpath('stat').read_text().split()[21]==ticks
            os.kill(pid,signal.SIGCONT)
            result=dict(auditor_pid=pid,auditor_start_ticks=ticks,resumed=True,
                        pause_seconds=time.monotonic()-started)
            write(b/'LAST_AUDIT_PRESERVATION_PAUSE.json',result)
            if target is not None and target.exists():write(target/'PRESERVATION_PAUSE.json',result)

if __name__=='__main__':main()
