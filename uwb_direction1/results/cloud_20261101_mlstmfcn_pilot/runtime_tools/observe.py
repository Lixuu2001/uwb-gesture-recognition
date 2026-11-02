"""Read-only OS/state observation; outside frozen study, no labels/model load."""
from pathlib import Path
import datetime,json,subprocess
B=Path('/workspace/uwb-mlstmfcn-pilot-20261101');HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def proc(pid):
 cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0');status={}
 for line in Path(f'/proc/{pid}/status').read_text().splitlines():
  if ':' in line:
   k,v=line.split(':',1)
   if k in ['State','PPid','VmRSS','Threads']:status[k]=v.strip()
 if status['State'].startswith('Z'):raise RuntimeError('Zombie process')
 return {'pid':pid,'command':[v.decode() for v in cmd if v],'status':status}
def main():
 launch=read(B/'LAUNCH_RECEIPT.json');state=read(B/'STATE.json') if (B/'STATE.json').exists() else {'state':'controller_preflight'}
 controller=proc(launch['controller_pid']);assert str(B/'controller.py') in controller['command']
 p=subprocess.run(['ps','--ppid',str(launch['controller_pid']),'-o','pid='],capture_output=True,text=True)
 children=[proc(int(v)) for v in p.stdout.split()]
 runstate=read(B/'run01/STATE.json') if (B/'run01/STATE.json').exists() else None
 for child in children:assert str(B/'run_pilot.py') in child['command'] or str(B/'audit_complete.py') in child['command']
 r={'state':'existing_MLSTM_pilot_OS_and_metadata_verified','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller':controller,'children':children,'controller_state':state,'training_state':runstate,'initial_checkpoint_exists':(B/'run01/initial.pt').exists(),'first_checkpoint_exists':(B/'run01/first_update.pt').exists(),'training_termination_exists':(B/'supervisor01/TERMINATION.json').exists(),'audit_termination_exists':(B/'audit_supervisor01/TERMINATION.json').exists(),'diagnostic_score_exists':(B/'run01/SUMMARY.json').exists(),'configuration_changed':False,'labels_loaded_by_observer':False}
 (HERE/'LIVE_OBSERVATION.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
