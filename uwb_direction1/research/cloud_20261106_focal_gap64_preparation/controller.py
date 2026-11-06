"""Bounded actual process supervision; no scoring, no automatic new configuration."""
from common import B,ready,read,write,sha,protocol
from pathlib import Path
import os,sys,subprocess,time,resource,signal,fcntl,json
def identity(pid):
    try:
        s=Path(f'/proc/{pid}/stat').read_text();q=s[s.rfind(')')+2:].split()
        return dict(pid=pid,startticks=int(q[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state=q[0],cpu=(int(q[11])+int(q[12]))/os.sysconf('SC_CLK_TCK'))
    except FileNotFoundError:return None
def rss_KiB(pid):
    try:
        raw=Path(f'/proc/{pid}/status').read_text()
        return next((int(line.split()[1]) for line in raw.splitlines() if line.startswith('VmRSS:')),0)
    except FileNotFoundError:return 0
def run_synthetic(mode):
    c=ready();assert mode in ['synthetic','profile'];out=B/(mode+'_execution');out.mkdir(exist_ok=False)
    p=subprocess.Popen([sys.executable,str(B/('capacity_profile.py' if mode=='profile' else 'synthetic.py'))],stdout=(out/'stdout.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
    start=time.monotonic();reason=None
    while p.poll() is None:
        q=identity(p.pid);rss=rss_KiB(p.pid) if q and q['state']!='Z' else 0
        if time.monotonic()-start>c['profile_wall_cap'] or (q and q['cpu']>c['profile_CPU_cap']) or rss>c['profile_RSS_KiB']:
            reason='fixed_synthetic_resource_cap';os.killpg(p.pid,signal.SIGTERM);break
        time.sleep(.5)
    p.wait();v=dict(exit_code=p.returncode,resource_stop_reason=reason,wall=time.monotonic()-start,new_UWB_fit_started=False);write(out/'TERMINATION.json',v)
    assert p.returncode==0 and reason is None,v
def main():
    c=ready(True);root=B/'run';root.mkdir(exist_ok=False)
    lock=(B/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # No duplicate controller/model startup. This version refuses restart; recovery must first preserve and account the interrupted state.
    assert not (B/'CONTROLLER_IDENTITY.json').exists()
    write(B/'CONTROLLER_IDENTITY.json',identity(os.getpid()));start=time.monotonic();base=resource.getrusage(resource.RUSAGE_CHILDREN);stages=[]
    def charge(live):
        r=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=r.ru_utime+r.ru_stime-base.ru_utime-base.ru_stime
        for p in live:
            q=identity(p.pid)
            if q and p.poll() is None:cpu+=q['cpu']
        return dict(wall=time.monotonic()-start,cpu=cpu)
    def execute(stage):
        d=root/'execution'/stage;d.mkdir(parents=True);log=(d/'stdout.log').open('w')
        p=subprocess.Popen([sys.executable,str(B/'worker.py'),stage],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(d/'IDENTITY.json',identity(p.pid));first=None;first_done=False;reason=None
        while p.poll() is None:
            live=[p]+([first] if first else []);cost=charge(live);rss=0
            for child in live:
                q=identity(child.pid)
                if q and q['state']!='Z':
                    rss+=rss_KiB(child.pid)
            write(B/'RESOURCE_LEDGER.json',dict(**cost,stage=stage,RSS_KiB=rss))
            if cost['wall']>c['real_wall_cap'] or cost['cpu']>c['real_CPU_cap'] or rss>c['real_RSS_KiB']:reason='fixed_pilot_resource_cap'
            if stage=='train' and (root/'train/FIRST_READY.json').exists() and first is None:
                first=subprocess.Popen([sys.executable,str(B/'worker.py'),'first'],stdout=(d/'first_audit.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True);write(d/'FIRST_AUDIT_IDENTITY.json',identity(first.pid))
            if first and first.poll() is not None and not first_done:
                first.wait();first_done=True;write(d/'FIRST_AUDIT_TERMINATION.json',dict(exit_code=first.returncode,native_first_release_exists=(root/'train/FIRST_RELEASE.json').exists()))
                if first.returncode!=0:reason='first_update_audit_failed'
            if reason:
                for child in live:
                    if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
                break
            time.sleep(1)
        p.wait()
        if first:first.wait()
        log.close();cost=charge([]);term=dict(exit_code=p.returncode,resource_stop_reason=reason,cumulative=cost,diagnostic_labels_read=False,users7_8_labels_used=False)
        write(d/'TERMINATION.json',term);stages.append(stage)
        assert p.returncode==0 and reason is None,term
    try:
        for stage in ['train','replay','output']:execute(stage)
        write(B/'STATE.json',dict(state='pilot_audited_frozen_unscored_requires_complete_private_backup_before_once_gate',completed=stages,diagnostic_labels_read=False,users7_8_labels_used=False,**charge([])))
    except BaseException as exc:
        write(B/'STATE.json',dict(state='execution_incomplete_requires_evidence_preserving_recovery',error=repr(exc),completed=stages,scientific_negative=False,diagnostic_labels_read=False,**charge([])));raise
    finally:lock.close()
if __name__=='__main__':
    if sys.argv[1] in ['synthetic','profile']:run_synthetic(sys.argv[1])
    else:assert sys.argv[1]=='--run';main()
