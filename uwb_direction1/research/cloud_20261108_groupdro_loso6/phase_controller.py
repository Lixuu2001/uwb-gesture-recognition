"""One bounded full-sixfold controller. Never score any held user's labels."""
from pathlib import Path
import os,sys,json,time,subprocess,fcntl,resource,signal
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'shared'))
from common import read,write,sha,ready
from controller import identity
def stop_owned(fold,child):
    # Every actual worker/auditor has its own session; verify boot/startticks before signalling.
    paths=list((fold/'run/execution').rglob('IDENTITY.json'))+list((fold/'run/execution').rglob('FIRST_AUDIT_IDENTITY.json')) if (fold/'run/execution').exists() else []
    for path in paths:
        old=read(path);now=identity(old['pid'])
        if now and now['state']!='Z' and all(now[k]==old[k] for k in ['pid','startticks','boot_id']):
            try:
                assert os.getpgid(now['pid'])==now['pid'];os.killpg(now['pid'],signal.SIGTERM)
            except ProcessLookupError:pass
    if child and child.poll() is None:
        try:
            assert os.getpgid(child.pid)==child.pid;os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    if child:child.wait()
def main():
    start=time.monotonic();ready(True)
    for root in ['/workspace/uwb-mmd-loso6-20261106','/workspace/uwb-mmd-gap64-20261106','/workspace/uwb-vrex-loso6-20261106','/workspace/uwb-fishr-native-loso6-20261107','/workspace/uwb-irm-loso6-20261107']:
        for name in ['PHASE_IDENTITY.json','CONTROLLER_IDENTITY.json']:
            q=Path(root)/name
            if q.exists():
                old=read(q);now=identity(old['pid'])
                assert now is None or now['state']=='Z' or any(now[k]!=old[k] for k in ['startticks','boot_id']),'Active prior model phase; no duplicate startup'
    lock=(P/'phase_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not(P/'PHASE_IDENTITY.json').exists() and not(P/'STATE.json').exists(),'Existing attempt requires preserved recovery, never relaunch'
    write(P/'PHASE_IDENTITY.json',identity(os.getpid()));prior=resource.getrusage(resource.RUSAGE_CHILDREN);completed=[]
    def cost(live=None):
        now=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=now.ru_utime+now.ru_stime-prior.ru_utime-prior.ru_stime+time.process_time()
        if live and live.poll() is None:
            j=identity(live.pid)
            if j and j['state']!='Z':cpu+=j['cpu']
        return dict(wall=time.monotonic()-start,cpu=cpu)
    child=None;current=None;last_cost=dict(wall=0.,cpu=0.)
    def interrupted(sig,frame):raise InterruptedError('phase interrupted by signal '+str(sig))
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,interrupted)
    try:
        for h in range(1,7):
            current=h;fold=P/'folds'/f'held{h}';assert not(fold/'run').exists()
            env=dict(os.environ,UWB_GROUPDRO_LOSO_FOLD_ROOT=str(fold))
            child=subprocess.Popen([sys.executable,str(P/'shared/controller.py'),'--run'],env=env,stdin=subprocess.DEVNULL,stdout=(fold/'controller.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True)
            write(fold/'LAUNCH_IDENTITY.json',identity(child.pid))
            while child.poll() is None:
                q=cost(child);ledger=read(fold/'RESOURCE_LEDGER.json') if (fold/'RESOURCE_LEDGER.json').exists() else dict(cpu=0,RSS_KiB=0,stage='starting')
                # Controller CPU excludes its not-yet-waited live descendants; fold ledger includes them.
                q['cpu']+=ledger['cpu'];rss=ledger.get('RSS_KiB',0)
                last_cost=dict(q)
                write(P/'RESOURCE_LEDGER.json',dict(**q,active_held=h,fold_stage=ledger['stage'],RSS_KiB=rss,completed_folds=list(completed),prior_MMD6_wall=30865.434969089023,prior_MMD6_cpu=30891.040083422995,prior_VREx6_wall=31157.163366941997,prior_VREx6_cpu=31189.589142517,prior_Fishr6_wall=29559.94331930898,prior_Fishr6_cpu=29599.386025808,prior_IRM6_wall=29509.953316993022,prior_IRM6_cpu=29548.175221702))
                if q['wall']>43200 or q['cpu']>43200 or rss>12582912:
                    stop_owned(fold,child);raise TimeoutError('fixed full-sixfold cumulative resource cap')
                time.sleep(1)
            child.wait();write(fold/'PHASE_TERMINATION.json',dict(exit_code=child.returncode,cumulative_remaining_phase=cost(),new_diagnostic_metrics_computed=False))
            assert child.returncode==0
            assert read(fold/'STATE.json')['state']=='pilot_audited_frozen_unscored_requires_complete_private_backup_before_once_gate'
            c=read(fold/'PROTOCOL.json')
            for side in ['train','replay']:
                result=read(fold/f'run/{side}/COMPLETE.json');assert result['epochs']==64 and result['updates']==c['expected_updates'] and not result['diagnostic_labels_read']
            completed.append(h);child=None
            write(P/'STATE.json',dict(state='full_sixfold_in_progress_unscored',completed_folds=list(completed),**cost(),diagnostic_labels_read=False))
        files={}
        for h in range(1,7):
            for n in ['selected.pt','final_full.pt','diagnostic_unscored.npz']:
                f=P/f'folds/held{h}/run/train/{n}';files[f'held{h}/'+n]=dict(path=str(f),sha256=sha(f))
        write(P/'ALL_SIX_PREDICTIONS_FROZEN.json',dict(files=files,all_held_users=[1,2,3,4,5,6],held1_reused=False,diagnostic_labels_read=False,users7_8_labels_used=False))
        write(P/'STATE.json',dict(state='all_six_method_folds_frozen_unscored_requires_complete_private_backup_before_cohort_once',completed_folds=completed,**cost(),diagnostic_labels_read=False))
    except BaseException as ex:
        if current is not None:stop_owned(P/'folds'/f'held{current}',child)
        failure_cost=cost();failure_cost['cpu']=max(failure_cost['cpu'],last_cost['cpu'])+5.
        write(P/'STATE.json',dict(state='execution_incomplete_preserve_evidence_no_scientific_negative',error=repr(ex),active_held=current,completed_folds=completed,**failure_cost,conservative_unknown_CPU_charge_seconds=5.,diagnostic_labels_read=False));raise
    finally:lock.close()
if __name__=='__main__':assert sys.argv[1:]==['--run'];main()
