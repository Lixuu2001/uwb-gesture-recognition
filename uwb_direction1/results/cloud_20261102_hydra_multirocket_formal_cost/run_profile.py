"""Independent bounded supervisor with explicit full-pipeline RSS monitoring."""
from pathlib import Path
import os,sys,time,json,signal,resource,subprocess
from guard import B,ready
def main():
    p=ready();assert not (B/'profile01').exists() and not (B/'supervisor01').exists()
    out=B/'supervisor01';out.mkdir();before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic()
    def limits():resource.setrlimit(resource.RLIMIT_CPU,(1180,1200))
    peak=0;reason=None
    with (out/'process.log').open('w') as log:
        child=subprocess.Popen([sys.executable,'-u',str(B/'profile_cost.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
        (out/'PID.json').write_text(json.dumps(dict(pid=child.pid,argv=[sys.executable,'-u',str(B/'profile_cost.py')])))
        while child.poll() is None:
            try:
                status=Path(f'/proc/{child.pid}/status').read_text()
                for line in status.splitlines():
                    if line.startswith('VmRSS:'):peak=max(peak,int(line.split()[1]))
            except FileNotFoundError:pass
            if peak>p['profile_RSS_stop_KiB']:reason='RSS_resource_limit'
            elif time.monotonic()-start>1200:reason='wall_resource_limit'
            if reason:
                child.terminate()
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:child.kill()
                break
            time.sleep(.2)
        code=child.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt=dict(pid=child.pid,exit_code=code,wall_timeout=reason=='wall_resource_limit',resource_stop_reason=reason,
        peak_observed_RSS_KiB=peak,RSS_stop_KiB=p['profile_RSS_stop_KiB'],supervisor_wall_seconds=time.monotonic()-start,
        child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,cpu_soft_seconds=1180,cpu_hard_seconds=1200,
        scientific_negative=False,accuracy_measured=False)
    (out/'TERMINATION.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(0 if code==0 else 1)
if __name__=='__main__':main()
