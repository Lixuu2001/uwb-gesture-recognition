"""Serial full8, cumulative wall/CPU caps, private interfold handoff; no retries."""
from pathlib import Path
import os,sys,time,signal,subprocess,fcntl,resource,shutil,math
from core import ROOT,read,write,sha
from supervisor import proc
from admission import require_production
def external_charges():
    path=ROOT/'EXTERNAL_PHASE_CHARGES.json';rows=read(path) if path.exists() else []
    if len({x['id'] for x in rows})!=len(rows):raise ValueError('Duplicate external charge')
    for r in rows:
        if not math.isfinite(r['CPU_seconds']) or r['CPU_seconds']<0:raise ValueError('Invalid charge')
        if sha(r['receipt'])!=r['receipt_sha256']:raise ValueError('External cost receipt changed')
    return rows
def verify_previous_private(held):
    charges={r['id'] for r in external_charges()}
    for h in range(1,held):
        fold=ROOT/'folds'/f'held{h}';b=fold/'private_backup';release=read(fold/'PRIVATE_FOLD_RELEASE.json')
        if release['source_frozen_sha256']!=sha(ROOT/'FROZEN_HASHES.json'):raise PermissionError('Private fold belongs to another source version')
        frozen=read(fold/'PREDICTIONS_FROZEN.json');m=read(b/'MANIFEST.json');proof=read(b/'READBACK_VERIFIED.json');rows=read(b/'READBACK_INPUT.json')
        if not(frozen['all3_arms_prod64_fresh64_audited_unscored'] and release['held']==h and release['frozen_outputs_sha256']==sha(fold/'PREDICTIONS_FROZEN.json')):raise PermissionError('Previous fold output freeze changed')
        if release['archive_manifest_sha256']!=sha(b/'MANIFEST.json') or release['raw_proof_sha256']!=sha(b/'READBACK_VERIFIED.json') or release['readback_input_sha256']!=sha(b/'READBACK_INPUT.json'):raise PermissionError('Private proof binding changed')
        if proof['archive_sha256']!=m['archive_sha256'] or proof['tracked_payload_members_verified']!=m['tracked_payload_files'] or proof['chunks_verified']!=len(m['chunks']):raise PermissionError('Raw allmember proof incomplete')
        if len(rows)!=len(m['chunks']) or len({r['file_id'] for r in rows})!=len(rows):raise PermissionError('Private parts missing')
        for c,r in zip(m['chunks'],rows):
            if any(c[k]!=r[k] for k in ['name','bytes','sha256']):raise PermissionError('Private part binding mismatch')
            md=r['metadata']
            if md['shared'] is not False or len(md['permissions'])!=1 or md['permissions'][0]['role']!='owner' or not r['raw_fetch'] or not r['download_verified'] or r['include_base64'] is not False:raise PermissionError('Private raw/owner evidence missing')
        members={x['path']:x for x in read(b/'MEMBERS.json')['members']}
        for path,hsh in frozen['files'].items():
            if sha(path)!=hsh:raise PermissionError('Frozen scientific output changed')
            key='fold/'+str(Path(path).relative_to(fold))
            if key not in members or members[key]['sha256']!=hsh:raise PermissionError('Frozen payload absent from backup')
        if not set(release['external_charge_ids'])<=charges:raise PermissionError('Private backup CPU uncharged')
    # Local transport cache can be evicted after raw verification; permanent IDs/proofs persist.
def resource_reason(cost,c,rss,free):
    if cost['wall']>=c['phase_wall_cap_seconds']:return 'fixed_phase_wall_cap'
    if cost['CPU']+cost['external_CPU']>=c['phase_CPU_cap_seconds']:return 'fixed_phase_CPU_cap'
    if rss>c['phase_RSS_cap_KiB']:return 'fixed_phase_RSS_cap'
    if free<c['disk_reserve_bytes']:return 'fixed_disk_reserve'
    return None
def stop(child):
    if child is None:return
    if child.poll() is None:
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
def main():
    held=int(sys.argv[1]);require_production(held)
    lock=(ROOT/'phase.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=read(ROOT/'PROTOCOL.json');phase=ROOT/'PHASE_LEDGER.json';boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    origin=read(ROOT/'PRELAUNCH_ORIGIN.json') if (ROOT/'PRELAUNCH_ORIGIN.json').exists() else dict(boot_id=boot,start_monotonic=time.monotonic())
    if origin['boot_id']!=boot:raise PermissionError('Prelaunch origin boot mismatch')
    ledger=read(phase) if phase.exists() else dict(boot_id=boot,start_monotonic=origin['start_monotonic'],CPU=0.,next_held=1,state='ready_next_fold',completed=[],stages=[])
    if ledger['boot_id']!=boot:raise PermissionError('Boot changed; reconcile costs and processes before recovery')
    if ledger['next_held']!=held or ledger['state'] not in ['ready_next_fold','waiting_private_backup']:raise PermissionError('Duplicate fold or automatic restart refused')
    out=ROOT/'folds'/f'held{held}';base=ledger['CPU'];self_start=0.;before=resource.getrusage(resource.RUSAGE_CHILDREN);child=None;started=False;old_handler=signal.getsignal(signal.SIGTERM)
    def interrupted(signum,frame):raise KeyboardInterrupt('Controller termination')
    signal.signal(signal.SIGTERM,interrupted)
    def charge(live=None):
        r=resource.getrusage(resource.RUSAGE_CHILDREN);v=base+r.ru_utime+r.ru_stime-before.ru_utime-before.ru_stime+time.process_time()-self_start
        if live is not None:
            s=proc(live.pid)
            if s and s['state']!='Z':v+=s['CPU']
        return dict(CPU=v,external_CPU=sum(x['CPU_seconds'] for x in external_charges()),wall=time.monotonic()-ledger['start_monotonic'])
    try:
        verify_previous_private(held)
        reason=resource_reason(charge(),c,0,shutil.disk_usage(ROOT).free)
        if reason:raise RuntimeError(reason+' before child startup')
        out.mkdir(parents=True,exist_ok=False);started=True
        write(out/'CONTROLLER_IDENTITY.json',proc(os.getpid()));ledger['state']='fold_running';ledger['active_held']=held;ledger.update(charge());write(phase,ledger)
        for stage in ['train','replay','output_audit']:
            reason=resource_reason(charge(),c,0,shutil.disk_usage(ROOT).free)
            if reason:raise RuntimeError(reason+' before stage startup')
            d=out/'execution'/stage;d.mkdir(parents=True,exist_ok=False)
            command=[sys.executable,str(ROOT/'fold_output_audit.py'),'--held',str(held)] if stage=='output_audit' else [sys.executable,str(ROOT/'worker.py'),stage,'--held',str(held)]
            with (d/'stdout.log').open('w') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(d/'IDENTITY.json',proc(child.pid));reason=None
                while child.poll() is None:
                    s=proc(child.pid);cost=charge(child);reason=resource_reason(cost,c,s['RSS_KiB'] if s else 0,shutil.disk_usage(ROOT).free)
                    write(out/'LIVE_RESOURCE.json',dict(stage=stage,**cost,child=s))
                    if reason:stop(child);break
                    time.sleep(.5)
                child.wait();term=dict(exit_code=child.returncode,resource_stop=reason,**charge());write(d/'TERMINATION.json',term);ledger['stages'].append(dict(held=held,stage=stage,**term));child=None
                ledger.update(charge());write(phase,ledger)
                if term['exit_code'] or reason:raise RuntimeError('Stage failed; evidence-preserving explicit recovery required: '+stage)
        frozen=read(out/'PREDICTIONS_FROZEN.json')
        if not frozen['all3_arms_prod64_fresh64_audited_unscored']:raise ValueError('Output audit incomplete')
        ledger['completed'].append(held);ledger['next_held']=held+1;ledger.pop('active_held',None);ledger['state']='waiting_private_backup' if held<8 else 'all8_audited_unscored_waiting_private_backup'
    except BaseException as exc:
        stop(child);child=None
        if started:ledger['state']='execution_incomplete_explicit_recovery_required';ledger['error']=repr(exc)
        else:ledger['last_preflight_rejection']=repr(exc)
        raise
    finally:
        stop(child);ledger.update(charge());write(phase,ledger);signal.signal(signal.SIGTERM,old_handler);lock.close()
if __name__=='__main__':main()
