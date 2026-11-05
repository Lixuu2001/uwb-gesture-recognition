"""One intermediate capture after same original producer completes, without fitting."""
from pathlib import Path
import sys,time,json,subprocess,os
D=Path(__file__).resolve().parent;B=Path('/workspace/uwb-mixstyle-gap64-20261105');M=Path('/workspace/uwb-mixstyle-gap-meta-20261105')
def read(p):return json.loads(p.read_text())
def main():
 start=time.monotonic();q=read(M/'RECOVERY_CONTROLLER_IDENTITY.json')
 while True:
  assert time.monotonic()-start<1200,'Original job left running, bounded insurance wait ends'
  s=Path(f"/proc/{q['pid']}/stat").read_text();a=s[s.rfind(')')+2:].split();assert int(a[19])==q['startticks'] and a[0]!='Z'
  assert not (M/'RECOVERY_CONTROLLER_FAILURE.json').exists()
  p=B/'run/replay/PROGRESS.json'
  if (B/'run/execution/train/TERMINATION.json').exists() and p.exists() and read(p)['epoch']>=1:break
  time.sleep(2)
 for stage in ['capture','native','archive']:
  r=subprocess.run([sys.executable,str(D/'snapshot.py'),stage],stdout=(D/(stage+'.log')).open('x'),stderr=subprocess.STDOUT)
  (D/(stage+'_TERMINATION.json')).write_text(json.dumps(dict(exit_code=r.returncode,new_model_fit_started=False,diagnostic_labels_read=False),indent=2)+'\n');assert r.returncode==0,stage
if __name__=='__main__':
 try:main();q=dict(exit_code=0,new_model_fit_started=False,diagnostic_labels_read=False)
 except BaseException as e:
  (D/'SUPERVISOR_TERMINATION.json').write_text(json.dumps(dict(exit_code=1,error=repr(e),new_model_fit_started=False),indent=2)+'\n');raise
 else:(D/'SUPERVISOR_TERMINATION.json').write_text(json.dumps(q,indent=2)+'\n')
