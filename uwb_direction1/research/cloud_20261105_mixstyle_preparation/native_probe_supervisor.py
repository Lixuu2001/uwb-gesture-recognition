"""Bounded separate synthetic subprocesses, actual exit/cost evidence, no real runner."""
from pathlib import Path
import json,os,signal,subprocess,sys,time
B=Path(__file__).resolve().parent
def run(stage):
    d=B/('native_'+stage+'_execution');d.mkdir(exist_ok=False);started=time.monotonic();reason=None
    with (d/'stdout.log').open('w') as f:
        p=subprocess.Popen([sys.executable,str(B/('native_checks.py' if stage=='checks' else 'capacity_profile.py'))],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        while p.poll() is None:
            try:
                raw=Path(f'/proc/{p.pid}/stat').read_text();v=raw[raw.rfind(')')+2:].split();cpu=(int(v[11])+int(v[12]))/os.sysconf('SC_CLK_TCK');status=Path(f'/proc/{p.pid}/status').read_text();rss=int(status.split('VmRSS:')[1].split()[0]) if 'VmRSS:' in status else 0
            except FileNotFoundError:continue
            if time.monotonic()-started>600 or cpu>600 or rss>12582912:reason='fixed_synthetic_resource_cap';os.killpg(p.pid,signal.SIGTERM);break
            time.sleep(.5)
        p.wait()
    q=dict(exit_code=p.returncode,resource_stop_reason=reason,wall_seconds=time.monotonic()-started,new_UWB_fit_started=False)
    (d/'TERMINATION.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(dict(stage=stage,**q)),flush=True)
    assert p.returncode==0 and reason is None,q
if __name__=='__main__':
    run('checks');run('capacity')
