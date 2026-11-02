"""Independent supervisor; preserves an incomplete receipt even after hard kill."""
import json,os,resource,subprocess,sys,time
from pathlib import Path
B=Path(__file__).resolve().parent
def bounded_run(argv,out,wall,soft_cpu,hard_cpu):
    out=Path(out);out.mkdir(exist_ok=False)
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    started=time.monotonic()
    def limits():resource.setrlimit(resource.RLIMIT_CPU,(soft_cpu,hard_cpu))
    with (out/'process.log').open('w') as log:
        child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,
            preexec_fn=limits,start_new_session=True)
        forced=False
        try:code=child.wait(timeout=wall)
        except subprocess.TimeoutExpired:
            forced=True;child.terminate()
            try:code=child.wait(timeout=3)
            except subprocess.TimeoutExpired:child.kill();code=child.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt=dict(pid=child.pid,exit_code=code,wall_timeout=forced,
        supervisor_wall_seconds=time.monotonic()-started,
        child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        cpu_soft_seconds=soft_cpu,cpu_hard_seconds=hard_cpu,
        scientific_negative=False)
    (out/'TERMINATION.json').write_text(json.dumps(receipt,indent=2))
    return receipt
def main():
    assert sys.argv[1:]==['--run']
    assert not (B/'run01').exists(),'inspect existing run, never duplicate'
    result=bounded_run([sys.executable,'-u',str(B/'run_pilot.py'),'--run'],
        B/'supervisor01',1790,1780,1790)
    state=B/'run01/STATE.json'
    if state.exists():
        previous=json.loads(state.read_text())
        if result['exit_code']!=0 and previous.get('state')!='complete':
            previous.update(state='incomplete',pid=None,supervisor=result,scientific_negative=False)
            tmp=state.with_suffix('.supervisor.tmp');tmp.write_text(json.dumps(previous,indent=2));tmp.replace(state)
    print(json.dumps(result))
    raise SystemExit(0 if result['exit_code']==0 else 1)
if __name__=='__main__':main()
