"""Retain termination receipt on wall, CPU or observed RSS resource failure."""
from pathlib import Path
import os,sys,resource,subprocess,time,json
def bounded_run(argv,out,wall,soft_cpu,hard_cpu):
    out=Path(out);out.mkdir(exist_ok=False);before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic()
    def limits():resource.setrlimit(resource.RLIMIT_CPU,(soft_cpu,hard_cpu))
    peak=0;reason=None
    with (out/'process.log').open('w') as log:
        child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limits,start_new_session=True)
        (out/'PID.json').write_text(json.dumps(dict(pid=child.pid,argv=argv)))
        while child.poll() is None:
            try:
                for line in Path(f'/proc/{child.pid}/status').read_text().splitlines():
                    if line.startswith('VmRSS:'):peak=max(peak,int(line.split()[1]))
            except FileNotFoundError:pass
            if peak>14680064:reason='RSS_resource_limit'
            elif time.monotonic()-start>wall:reason='wall_resource_limit'
            if reason:
                child.terminate()
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:child.kill()
                break
            time.sleep(.2)
        code=child.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    result=dict(pid=child.pid,exit_code=code,wall_timeout=reason=='wall_resource_limit',resource_stop_reason=reason,
       peak_observed_RSS_KiB=peak,RSS_stop_KiB=14680064,supervisor_wall_seconds=time.monotonic()-start,
       child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
       cpu_soft_seconds=soft_cpu,cpu_hard_seconds=hard_cpu,scientific_negative=False)
    (out/'TERMINATION.json').write_text(json.dumps(result,indent=2)+'\n');return result
