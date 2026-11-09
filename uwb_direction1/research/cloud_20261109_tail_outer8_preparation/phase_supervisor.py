"""Serial full8 phase supervisor; explicit production release and private interfold handoff."""
from pathlib import Path
import os,sys,time,signal,subprocess,fcntl,resource,shutil
from core import ROOT,read,write,sha
from supervisor import proc
from admission import require_production
def verify_previous_private(held):
    if held==1:return
    from stream_archive import verify
    for h in range(1,held):
        b=ROOT/'folds'/f'held{h}'/'private_backup';m=read(b/'MANIFEST.json');rows=read(b/'READBACK_INPUT.json')
        if verify(m,rows)!=read(b/'READBACK_VERIFIED.json'):raise PermissionError('Previous fold private raw proof incomplete')
        frozen=read(ROOT/'folds'/f'held{h}'/'PREDICTIONS_FROZEN.json')
        if not frozen['all3_arms_prod64_fresh64_audited_unscored']:raise PermissionError('Previous fold audits incomplete')
        for p,hsh in frozen['files'].items():
            if sha(p)!=hsh:raise PermissionError('Prior frozen fold changed')
def main():
    held=int(sys.argv[1]);require_production(held);release=read(ROOT/'PRODUCTION_RELEASE.json')
    for n in ['phase_supervisor.py','fold_output_audit.py']:
        if release.get('additional_sources',{}).get(n)!=sha(ROOT/n):raise PermissionError('New phase source not explicitly released')
    lock=(ROOT/'phase.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_previous_private(held);c=read(ROOT/'PROTOCOL.json');phase=ROOT/'PHASE_LEDGER.json';boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if phase.exists():
        ledger=read(phase)
        if ledger['boot_id']!=boot:raise PermissionError('Different boot: reconcile complete wall/CPU evidence before restart')
        if ledger['next_held']!=held or ledger['state'] not in ['ready_next_fold','waiting_private_backup']:raise PermissionError('No automatic restart or duplicate fold')
    else:
        if held!=1:raise PermissionError('Start with fold1')
        ledger=dict(boot_id=boot,start_monotonic=time.monotonic(),CPU=0.,next_held=1,state='ready_next_fold',completed=[],stages=[])
    out=ROOT/'folds'/f'held{held}';out.mkdir(parents=True,exist_ok=False)
    write(out/'CONTROLLER_IDENTITY.json',proc(os.getpid()));base_cpu=ledger['CPU'];self_cpu=time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    def charge(child=None):
        r=resource.getrusage(resource.RUSAGE_CHILDREN);v=base_cpu+(r.ru_utime+r.ru_stime-before.ru_utime-before.ru_stime)+(time.process_time()-self_cpu)
        if child is not None:
            s=proc(child.pid)
            if s and s['state']!='Z':v+=s['CPU']
        # External archive/verification CPU is explicit measured receipts, never assumed zero.
        external=read(ROOT/'EXTERNAL_PHASE_CHARGES.json') if (ROOT/'EXTERNAL_PHASE_CHARGES.json').exists() else []
        if len({x['id'] for x in external})!=len(external):raise ValueError('Duplicate external CPU charge')
        return dict(CPU=v,external_CPU=sum(x['CPU_seconds'] for x in external),wall=time.monotonic()-ledger['start_monotonic'])
    try:
        for stage in ['train','replay','output_audit']:
            d=out/'execution'/stage;d.mkdir(parents=True,exist_ok=False)
            command=[sys.executable,str(ROOT/'fold_output_audit.py'),'--held',str(held)] if stage=='output_audit' else [sys.executable,str(ROOT/'worker.py'),stage,'--held',str(held)]
            with (d/'stdout.log').open('w') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(d/'IDENTITY.json',proc(child.pid));reason=None
                while child.poll() is None:
                    s=proc(child.pid);cost=charge(child)
                    if cost['wall']>c['phase_wall_cap_seconds'] or cost['CPU']+cost['external_CPU']>c['phase_CPU_cap_seconds'] or (s and s['RSS_KiB']>c['phase_RSS_cap_KiB']):reason='fixed_phase_resource_cap'
                    if shutil.disk_usage(ROOT).free<c['disk_reserve_bytes']:reason='fixed_disk_reserve'
                    write(out/'LIVE_RESOURCE.json',dict(stage=stage,**cost,child=s))
                    if reason:os.killpg(child.pid,signal.SIGTERM);break
                    time.sleep(1)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            term=dict(exit_code=child.returncode,resource_stop=reason,**charge());write(d/'TERMINATION.json',term);ledger['stages'].append(dict(held=held,stage=stage,**term))
            if child.returncode or reason:raise RuntimeError('Failed stage; no automatic retry: '+stage)
        ledger['completed'].append(held);ledger['next_held']=held+1;ledger['state']='waiting_private_backup' if held<8 else 'all8_audited_unscored_waiting_private_backup'
    except BaseException as exc:
        ledger['state']='execution_incomplete_explicit_evidence_preserving_recovery_required';ledger['error']=repr(exc);raise
    finally:
        ledger.update(charge());write(phase,ledger);lock.close()
if __name__=='__main__':main()
