"""Prospective bounded real-trial child supervisor; no activation by itself."""
from pathlib import Path
import sys,time,json,subprocess,resource
def events():
    p=Path('/sys/fs/cgroup/memory.events')
    return {x.split()[0]:int(x.split()[1]) for x in p.read_text().splitlines()} if p.exists() else {}
def bounded_run(argv,out,wall,soft_cpu,hard_cpu,rss_stop):
    p=dict(profile_CPU_soft_seconds=soft_cpu,profile_CPU_hard_seconds=hard_cpu,profile_wall_seconds=wall,RSS_stop_KiB=rss_stop)
    out=Path(out);out.mkdir(exist_ok=False)
    before=resource.getrusage(resource.RUSAGE_CHILDREN);e0=events()
    start=time.monotonic();peak=0;reason=None
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU,(
            p['profile_CPU_soft_seconds'],p['profile_CPU_hard_seconds']))
    with (out/'process.log').open('x') as log:
        try:
            child=subprocess.Popen(argv,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
        except (OSError,subprocess.SubprocessError) as e:
            failure=dict(pid=None,exit_code=None,resource_stop_reason='child_launch_failed',
                wall_timeout=False,error_type=type(e).__name__,
                supervisor_wall_seconds=time.monotonic()-start,
                accuracy_measured=False,
                scientific_negative=False)
            (out/'TERMINATION.json').write_text(json.dumps(failure,indent=2)+'\n')
            raise
        (out/'PID.json').write_text(json.dumps(dict(pid=child.pid,
            argv=argv))+'\n')
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
        accuracy_measured=False,scientific_negative=False)
    (out/'TERMINATION.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r))
    return r
