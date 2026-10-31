"""Read-only process/state observation, no UWB arrays, model loads or scores."""
from pathlib import Path
import datetime,json,os,sys,time
B=Path('/workspace/uwb-hinception-sixfold-20261031')
def read(p):return json.loads(p.read_bytes())
def proc(pid):
 p=Path('/proc')/str(pid);status={}
 for line in (p/'status').read_text().splitlines():
  if ':' in line:
   k,v=line.split(':',1)
   if k in ['Name','State','PPid','VmRSS','Threads']:status[k]=v.strip()
 args=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
 assert status['State'].split()[0] not in ['Z','X'];return dict(pid=pid,command=args,status=status)
def main():
 sys.path.insert(0,str(B));from readiness import ready
 ready(read(B/'PROTOCOL.json'));state=read(B/'STATE.json');assert state['state']=='running_unscored_source_confirmation' and not state['failed'];processes=[proc(state['pid'])];folds={}
 for held,pid in state['active'].items():
  job=read(B/'jobs'/('held'+held)/'STATE.json');worker=read(B/'run01'/('held'+held)/'STATE.json');assert job['state']=='training' and worker['state']=='training' and worker['total_steps']>=1 and worker['diagnostic_labels_read'] is False
  processes.extend([proc(pid),proc(worker['pid'])]);assert processes[-1]['command'][-2:]==['--held',held];folds[held]=dict(job=job,worker=worker)
 out=dict(state='verified_actual_source_confirmation_training_processes',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue=state,folds=folds,processes=processes,all1465_frozen_dependencies_reverified=True,no_new_diagnostic_labels_scored=True,users7_8_labels_used=False,complete_folds=0,source_sixfold_result_available=False,complete_private_modelset_verified=False)
 (B/'LIVE_OBSERVATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(state=out['state'],active=list(folds),epoch_steps={h:[r['worker']['epoch'],r['worker']['total_steps']] for h,r in folds.items()})))
if __name__=='__main__':main()
