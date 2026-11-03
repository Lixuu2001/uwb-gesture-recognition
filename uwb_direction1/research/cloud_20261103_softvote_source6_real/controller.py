"""Sequential five folds, parallel native members, sealed held outputs.

Resumption requires every recorded process to be inactive. Unknown elapsed CPU
is conservatively charged at five CPU-seconds per wall-second, not reset.
"""
from pathlib import Path
import sys,os,time,json,subprocess,resource,fcntl,math,shutil
from common import B,read,write,sha,require_initial,process_identity,alive,config
from checks import member,verify_member,verify_audits
def usage():
    a=resource.getrusage(resource.RUSAGE_CHILDREN);s=resource.getrusage(resource.RUSAGE_SELF)
    return a.ru_utime+a.ru_stime+s.ru_utime+s.ru_stime
def events():return {x.split()[0]:int(x.split()[1]) for x in Path('/sys/fs/cgroup/memory.events').read_text().splitlines()}
def budget_reason(folds,held,p,rss):
    row=folds[str(held)]
    if row['wall']>p['per_fold_wall_seconds']:return 'fold_wall_budget'
    if row['cpu']>p['per_fold_CPU_seconds']:return 'fold_CPU_budget'
    if sum(x['wall'] for x in folds.values())>p['total5fold_wall_upper_seconds']:return 'total_wall_budget'
    if sum(x['cpu'] for x in folds.values())>p['total5fold_CPU_upper_seconds']:return 'total_CPU_budget'
    if rss>p['combined_RSS_stop_KiB']:return 'combined_RSS_budget'
    return None
def ensure_inactive():
    for path in (B/'attempts').glob('*/held*/PROCESSES.json'):
        for identity in read(path)['identities']:
            assert not alive(identity),'Recorded process still active; no duplicate or unproved restart'
    if (B/'CONTROLLER_IDENTITY.json').exists():assert not alive(read(B/'CONTROLLER_IDENTITY.json'))
def release_first(out):
    out=Path(out);a=read(out/'first_audit/CERTIFICATE.json');t=read(out/'first_audit/TERMINATION.json')
    assert a['state']=='pass' and t['exit_code']==0 and t['resource_stop_reason'] is None and t['cgroup_OOM_kill_delta']==0
    assert a['first_checkpoint_sha256']==sha(out/'first_update.pt')
    write(out/'FIRST_RELEASE.json',dict(first_sha256=sha(out/'first_update.pt'),certificate_sha256=sha(out/'first_audit/CERTIFICATE.json')))
def valid_release(out):
    try:
        r=read(out/'FIRST_RELEASE.json');a=read(out/'first_audit/CERTIFICATE.json');t=read(out/'first_audit/TERMINATION.json')
        return a['state']=='pass' and t['exit_code']==0 and t['resource_stop_reason'] is None and t['cgroup_OOM_kill_delta']==0 and r['first_sha256']==sha(out/'first_update.pt') and r['certificate_sha256']==sha(out/'first_audit/CERTIFICATE.json')
    except (FileNotFoundError,KeyError):return False
def preserve(path):
    path=Path(path)
    if path.exists():
        d=path.parent/'orphans';d.mkdir(exist_ok=True);path.rename(d/(path.name+'.'+str(time.time_ns())))
def valid_fold(held):
    root=B/'folds'/f'held{held}'
    try:
        a=read(root/'fusion/FOLD_FROZEN.json');t=read(root/'FUSION_TERMINATION.json')
        assert a['state']=='three_arms_frozen_before_any_new_held_truth' and not a['diagnostic_labels_read']
        for n,h in a['files'].items():assert sha(root/'fusion'/n)==h
        return t['exit_code']==0 and t['resource_stop_reason'] is None and not t['wall_timeout'] and t['cgroup_OOM_kill_delta']==0
    except FileNotFoundError:return False
class Session:
    def __init__(self,held,attempt,p,ledger):
        self.held=held;self.attempt=attempt;self.p=p;self.ledger=ledger;self.jobs=[];self.identities=[]
        self.start=time.monotonic();self.u0=usage();self.base=dict(ledger['folds'][str(held)]);self.ev0=events();self.last_save=0;self.reason=None
    def tick(self,force=False):
        rss=0;active_cpu=0
        for j in self.jobs:
            if j['done']:continue
            ident=process_identity(j['p'].pid)
            if ident is not None and ident['state']!='Z':active_cpu+=ident['CPU_seconds'];j['observed_cpu']=ident['CPU_seconds']
            try:
                value=sum(int(x.split()[1]) for x in Path(f'/proc/{j["p"].pid}/status').read_text().splitlines() if x.startswith('VmRSS:'))
                rss+=value;j['peak_rss']=max(j['peak_rss'],value)
            except FileNotFoundError:pass
        elapsed=time.monotonic()-self.start
        self.ledger['folds'][str(self.held)]=dict(wall=self.base['wall']+elapsed,cpu=self.base['cpu']+usage()-self.u0+active_cpu)
        self.reason=budget_reason(self.ledger['folds'],self.held,self.p,rss)
        if events()['oom_kill']!=self.ev0['oom_kill']:self.reason='cgroup_OOM_kill'
        if force or elapsed-self.last_save>=5:
            self.ledger.update(active_held=self.held,last_observed_unix=time.time(),last_observed_combined_RSS_KiB=rss)
            write(B/'RESOURCE_LEDGER.json',self.ledger);self.last_save=elapsed
        if self.reason:raise RuntimeError('Resource stop: '+self.reason)
    def spawn(self,name,script,args,receipt):
        self.tick();remaining=int(self.p['per_fold_CPU_seconds']-self.ledger['folds'][str(self.held)]['cpu']);assert remaining>20
        logpath=self.attempt/(name+'.log');log=logpath.open('x')
        def limits():resource.setrlimit(resource.RLIMIT_CPU,(remaining-10,remaining))
        child=subprocess.Popen([sys.executable,'-u',str(B/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits,cwd=B)
        identity=process_identity(child.pid);assert identity is not None
        self.identities.append(identity);write(self.attempt/'PROCESSES.json',dict(identities=self.identities))
        job=dict(name=name,p=child,log=log,receipt=Path(receipt),start=time.monotonic(),peak_rss=0,observed_cpu=0,done=False)
        self.jobs.append(job);return job
    def finish(self,j,reason=None):
        if j['done']:return
        code=j['p'].poll()
        if code is None:return
        j['log'].close();j['done']=True
        write(j['receipt'],dict(exit_code=code,resource_stop_reason=reason,wall_timeout=reason in ['fold_wall_budget','total_wall_budget'],pid=j['p'].pid,wall_seconds=time.monotonic()-j['start'],peak_observed_RSS_KiB=j['peak_rss'],observed_child_CPU_seconds_lower_bound=j['observed_cpu'],individual_CPU_exact=False,CPU_exact_scope='fold/global RESOURCE_LEDGER uses summed RUSAGE plus live-process CPU; per-child last observation can be below final CPU',cgroup_OOM_kill_delta=events()['oom_kill']-self.ev0['oom_kill'],diagnostic_accuracy_measured=False))
    def wait(self,jobs):
        while any(not j['done'] for j in jobs):
            for j in jobs:
                self.finish(j)
                if j['done'] and j['p'].returncode!=0:raise RuntimeError('Child failed: '+j['name'])
            self.tick();time.sleep(.2)
    def stop(self):
        for j in self.jobs:
            if j['p'].poll() is None:
                j['p'].terminate()
                try:j['p'].wait(timeout=3)
                except subprocess.TimeoutExpired:j['p'].kill();j['p'].wait()
            self.finish(j,self.reason or 'controller_stopped_after_peer_failure')
        try:self.tick(force=True)
        except RuntimeError:pass
    def producers(self,resume):
        trainers={};auditors={}
        for arm in ['timemil','shuffle']:
            out=member(self.held,arm)
            if (out/'ARM_COMPLETE.json').exists():verify_member(self.held,arm);continue
            exists=out.exists();assert not exists or resume
            if exists and not valid_release(out):
                preserve(out/'FIRST_RELEASE.json');preserve(out/'first_audit')
            args=[self.held,arm]+(['--resume'] if exists else [])
            trainers[arm]=self.spawn(arm+'_train','produce.py',args,self.attempt/(arm+'_train_TERM.json'))
        while any(not j['done'] for j in trainers.values()) or any(not j['done'] for j in auditors.values()):
            for arm,j in trainers.items():
                out=member(self.held,arm)
                if (out/'FIRST_READY.json').exists() and not valid_release(out) and arm not in auditors:
                    auditors[arm]=self.spawn(arm+'_first','first_audit.py',[self.held,arm],out/'first_audit/TERMINATION.json')
                self.finish(j)
                if j['done'] and j['p'].returncode!=0:raise RuntimeError('Producer failed: '+arm)
            for arm,j in auditors.items():
                was=j['done'];self.finish(j)
                if j['done'] and j['p'].returncode!=0:raise RuntimeError('First update audit failed: '+arm)
                if j['done'] and not was:release_first(member(self.held,arm))
            self.tick();time.sleep(.2)
        for arm in ['timemil','shuffle']:verify_member(self.held,arm)
    def audit_job(self,arm,kind):
        out=member(self.held,arm)
        if kind=='replay':
            dest=out/'replay';done=dest/'COMPLETE_AUDIT.json'
            if done.exists():
                if (dest/'TERMINATION.json').exists():
                    t=read(dest/'TERMINATION.json')
                    if t['exit_code']==0 and t['resource_stop_reason'] is None and t['cgroup_OOM_kill_delta']==0:return None
                preserve(done);preserve(dest/'TERMINATION.json')
            return self.spawn('shuffle_replay','replay_shuffle.py',[self.held],dest/'TERMINATION.json')
        dest=out/'inference_audit'
        if (dest/'CERTIFICATE.json').exists() and (dest/'TERMINATION.json').exists():
            t=read(dest/'TERMINATION.json')
            if t['exit_code']==0 and t['resource_stop_reason'] is None:return None
        preserve(dest)
        return self.spawn(arm+'_inference','audit_inference.py',[self.held,arm],dest/'TERMINATION.json')
    def run(self,resume):
        try:
            self.producers(resume)
            write(B/'STATE.json',dict(state='auditing_members_unscored',held=self.held))
            self.wait([j for j in [self.audit_job('shuffle','replay'),self.audit_job('timemil','inference')] if j])
            j=self.audit_job('shuffle','inference')
            if j:self.wait([j])
            verify_audits(self.held)
            root=B/'folds'/f'held{self.held}'
            if not valid_fold(self.held):
                preserve(root/'fusion');preserve(root/'FUSION_TERMINATION.json')
                j=self.spawn('freeze_three_arms','fuse.py',[self.held],root/'FUSION_TERMINATION.json');self.wait([j])
            self.tick(force=True)
        finally:self.stop()
def main():
    resume=len(sys.argv)==2 and sys.argv[1]=='--resume'
    assert len(sys.argv)==1 or resume
    lock=(B/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ensure_inactive();start=time.monotonic();u0=usage();c,p=require_initial()
    if (B/'RESOURCE_LEDGER.json').exists():
        assert resume;ledger=read(B/'RESOURCE_LEDGER.json');held=ledger.get('active_held')
        if held is not None:
            unknown=max(0,time.time()-ledger['last_observed_unix']);ledger['folds'][str(held)]['wall']+=unknown;ledger['folds'][str(held)]['cpu']+=5*unknown
    else:
        assert not resume;ledger=dict(folds={str(h):dict(wall=0,cpu=0) for h in [2,3,4,5,6]},last_observed_unix=time.time(),active_held=None)
    pending=[h for h in [2,3,4,5,6] if not valid_fold(h)]
    if pending:
        row=ledger['folds'][str(pending[0])];row['wall']+=time.monotonic()-start;row['cpu']+=usage()-u0
    write(B/'CONTROLLER_IDENTITY.json',process_identity(os.getpid()));write(B/'RESOURCE_LEDGER.json',ledger)
    attempts=B/'attempts';attempts.mkdir(exist_ok=True);attempt=attempts/f'attempt{len(list(attempts.iterdir())):03}';attempt.mkdir(exist_ok=False)
    write(attempt/'START.json',dict(resume=resume,identity=process_identity(os.getpid()),registration=read(B/'REMOTE_REAL_PREREGISTRATION.json')['commit'],UTC_unix=time.time(),recovery_unknown_CPU_policy='5CPU per unknown elapsed wall second',scientific_configuration_unchanged=True))
    try:
        for held in [2,3,4,5,6]:
            frozen=B/'folds'/f'held{held}'/'fusion/FOLD_FROZEN.json'
            if valid_fold(held):verify_audits(held);continue
            write(B/'STATE.json',dict(state='training_fixed_members_unscored',held=held,users7_8_labels_used=False))
            sub=attempt/f'held{held}';sub.mkdir();session=Session(held,sub,p,ledger);session.run(resume)
            ledger.update(active_held=None,last_observed_unix=time.time());write(B/'RESOURCE_LEDGER.json',ledger)
        for held in [2,3,4,5,6]:verify_audits(held)
        write(B/'ALL_PREDICTIONS_FROZEN.json',dict(state='all5newfolds_3arms_frozen_unscored',folds={str(h):sha(B/'folds'/f'held{h}'/'fusion/FOLD_FROZEN.json') for h in [2,3,4,5,6]},method_lock_sha256=sha(B/'METHOD_LOCK.json'),diagnostic_labels_read=False,users7_8_labels_used=False))
        write(B/'STATE.json',dict(state='all5_audited_unscored_requires_private_backup_then_once_aggregate',held=None,diagnostic_labels_read=False))
    except BaseException as exc:
        import traceback
        write(B/'STATE.json',dict(state='execution_incomplete_no_accuracy_claim',error=repr(exc),traceback=traceback.format_exc(),diagnostic_labels_read=False,scientific_negative=False));raise
if __name__=='__main__':main()
