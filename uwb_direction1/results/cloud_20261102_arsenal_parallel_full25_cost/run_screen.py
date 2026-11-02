"""Bounded4CPU/12GiB screening; retain direct RSS and cgroup OOM evidence."""
from pathlib import Path
import sys,time,json,subprocess,resource
from guard import B,ready
def events():
    p=Path('/sys/fs/cgroup/memory.events')
    return {s.split()[0]:int(s.split()[1]) for s in p.read_text().splitlines()} if p.exists() else {}
def main():
    policy=ready();assert not (B/'profile01').exists() and not (B/'supervisor01').exists();out=B/'supervisor01';out.mkdir()
    before=resource.getrusage(resource.RUSAGE_CHILDREN);ev_before=events();start=time.monotonic();peak=0;reason=None
    def limits():resource.setrlimit(resource.RLIMIT_CPU,(14380,14400))
    with (out/'process.log').open('w') as log:
        child=subprocess.Popen([sys.executable,'-u',str(B/'profile_cost.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
        (out/'PID.json').write_text(json.dumps(dict(pid=child.pid,argv=[sys.executable,'-u',str(B/'profile_cost.py')])))
        while child.poll() is None:
            try:
                for line in Path(f'/proc/{child.pid}/status').read_text().splitlines():
                    if line.startswith('VmRSS:'):peak=max(peak,int(line.split()[1]))
            except FileNotFoundError:pass
            if peak>12582912:reason='RSS_resource_limit'
            elif time.monotonic()-start>3600:reason='wall_resource_limit'
            if reason:
                child.terminate()
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:child.kill()
                break
            time.sleep(.2)
        code=child.wait()
    ev_after=events();oom_delta=ev_after.get('oom_kill',0)-ev_before.get('oom_kill',0)
    if reason is None and code==-9 and oom_delta>0:reason='cgroup_OOM_kill_recorded'
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt=dict(pid=child.pid,exit_code=code,wall_timeout=reason=='wall_resource_limit',resource_stop_reason=reason,peak_observed_RSS_KiB=peak,RSS_stop_KiB=12582912,cgroup_events_before=ev_before,cgroup_events_after=ev_after,cgroup_oom_kill_delta=oom_delta,supervisor_wall_seconds=time.monotonic()-start,child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,cpu_soft_seconds=14380,cpu_hard_seconds=14400,scientific_negative=False,accuracy_measured=False,real_UWB_values_labels_fit=False)
    (out/'TERMINATION.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(0 if code==0 else 1)
if __name__=='__main__':main()
