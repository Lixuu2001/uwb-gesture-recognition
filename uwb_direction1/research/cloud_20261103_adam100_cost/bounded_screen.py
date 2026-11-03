"""UNEXECUTED synthetic-only supervisor; requires a separate promoted study."""
from pathlib import Path
import sys,time,json,subprocess,resource
from cost_guard import B,ready
def events():
    p=Path('/sys/fs/cgroup/memory.events')
    return {x.split()[0]:int(x.split()[1]) for x in p.read_text().splitlines()} if p.exists() else {}
def main():
    p=ready()
    assert not (B/'profile01').exists()
    out=B/'supervisor01';out.mkdir(exist_ok=False)
    before=resource.getrusage(resource.RUSAGE_CHILDREN);e0=events()
    start=time.monotonic();peak=0;reason=None
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU,(
            p['profile_CPU_soft_seconds'],p['profile_CPU_hard_seconds']))
    with (out/'process.log').open('x') as log:
        try:
            child=subprocess.Popen([sys.executable,'-u',str(B/'profile_cost.py')],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
        except (OSError,subprocess.SubprocessError) as e:
            failure=dict(pid=None,exit_code=None,resource_stop_reason='profile_launch_failed',
                wall_timeout=False,error_type=type(e).__name__,
                supervisor_wall_seconds=time.monotonic()-start,
                new_UWB_values_or_labels_fit=False,accuracy_measured=False,
                scientific_negative=False)
            (out/'TERMINATION.json').write_text(json.dumps(failure,indent=2)+'\n')
            raise
        (out/'PID.json').write_text(json.dumps(dict(pid=child.pid,
            script=str(B/'profile_cost.py')))+'\n')
        while child.poll() is None:
            try:
                lines=Path('/proc/'+str(child.pid)+'/status').read_text().splitlines()
                values=[int(x.split()[1]) for x in lines if x.startswith('VmRSS:')]
                if values:peak=max(peak,values[0])
                elif child.poll() is None:reason='RSS_observation_unavailable'
            except FileNotFoundError:pass
            except OSError:reason='RSS_observation_unavailable'
            if peak>p['RSS_stop_KiB']:reason='RSS_resource_limit'
            if time.monotonic()-start>p['profile_wall_seconds']:reason='wall_resource_limit'
            if reason:
                child.terminate()
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:child.kill()
                break
            time.sleep(.2)
        code=child.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN);e1=events()
    # A successful short-lived process may have escaped a periodic RSS sample.
    peak=max(peak,int(after.ru_maxrss))
    oom=e1.get('oom_kill',0)-e0.get('oom_kill',0)
    if peak>p['RSS_stop_KiB']:reason='RSS_resource_limit'
    if oom:reason='cgroup_OOM_kill_recorded'
    r=dict(pid=child.pid,exit_code=code,resource_stop_reason=reason,
        wall_timeout=reason=='wall_resource_limit',peak_observed_RSS_KiB=peak,
        supervisor_wall_seconds=time.monotonic()-start,
        child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        cgroup_events_before=e0,cgroup_events_after=e1,cgroup_oom_kill_delta=oom,
        cgroup_events_available='oom_kill' in e0 and 'oom_kill' in e1,
        new_UWB_values_or_labels_fit=False,accuracy_measured=False,scientific_negative=False)
    (out/'TERMINATION.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r))
    raise SystemExit(0 if code==0 and reason is None and oom==0 else 1)
if __name__=='__main__':main()
