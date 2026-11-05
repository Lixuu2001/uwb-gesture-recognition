"""Bounded one-shot synthetic checks/profile; records actual exit, never retries."""
from pathlib import Path
import subprocess,sys,time,os,json,signal
B=Path(__file__).resolve().parent
def main():
 mode=sys.argv[1];assert mode in ['native','profile'];out=B/(mode+'_execution');out.mkdir(exist_ok=False)
 p=subprocess.Popen([sys.executable,str(B/('native_checks.py' if mode=='native' else 'capacity_profile.py'))],stdin=subprocess.DEVNULL,stdout=(out/'stdout.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic();reason=None
 while p.poll() is None:
  try:
   raw=Path(f'/proc/{p.pid}/stat').read_text();a=raw[raw.rfind(')')+2:].split();cpu=(int(a[11])+int(a[12]))/os.sysconf('SC_CLK_TCK');status=Path(f'/proc/{p.pid}/status').read_text();rss=next((int(x.split()[1]) for x in status.splitlines() if x.startswith('VmRSS:')),0)
  except FileNotFoundError:continue
  if time.monotonic()-start>600 or cpu>600 or rss>12582912:reason='fixed_synthetic_resource_cap';os.killpg(p.pid,signal.SIGTERM);break
  time.sleep(.2)
 p.wait();q=dict(exit_code=p.returncode,resource_stop_reason=reason,wall=time.monotonic()-start,new_UWB_fit_started=False,diagnostic_labels_read=False);(out/'TERMINATION.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q));assert p.returncode==0 and reason is None
if __name__=='__main__':main()
