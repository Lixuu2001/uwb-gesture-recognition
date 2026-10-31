"""Read-only observation of producer-complete folds2/3 and real audit processes."""
from pathlib import Path
import datetime,json,os,sys
B=Path('/workspace/uwb-hinception-sixfold-20261031')
def read(p):return json.loads(Path(p).read_bytes())
def process(pid):
 p=Path('/proc')/str(pid);status={}
 for line in (p/'status').read_text().splitlines():
  if ':' in line:
   k,v=line.split(':',1)
   if k in ['Name','State','PPid','VmRSS','Threads']:status[k]=v.strip()
 assert status['State'].split()[0] not in ['Z','X'];args=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x];fields=(p/'stat').read_text().rsplit(')',1)[1].split();cpu=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK');return dict(pid=pid,command=args,status=status,observed_process_cpu_seconds=cpu)
def main():
 sys.path.insert(0,str(B));from readiness import ready
 ready(read(B/'PROTOCOL.json'));queue=read(B/'STATE.json');assert queue['state']=='running_unscored_source_confirmation' and not queue['failed'] and set(queue['active'])=={'2','3'};procs=[process(queue['pid'])];folds={}
 for held in ['2','3']:
  jobpid=queue['active'][held];job=read(B/'jobs'/('held'+held)/'STATE.json');producer=read(B/'run01'/('held'+held)/'STATE.json');term=read(B/'jobs'/('held'+held)/'train/TERMINATION.json');assert job['state']=='numeric_audit' and producer['state']=='complete' and producer['pid'] is None and producer['all_predictions_frozen'] and producer['diagnostic_labels_read'] is False and term['exit_code']==0 and not term['wall_timeout'];procs.append(process(jobpid));children=[int(p) for p in (Path('/proc')/str(jobpid)/'task'/str(jobpid)/'children').read_text().split()];auditors=[]
  for child in children:
   proc=process(child)
   if str(B/'audit_fold.py') in proc['command']:
    assert proc['command'][-2:]==['--held',held] and int(proc['status']['PPid'])==jobpid;auditors.append(proc);procs.append(proc)
  assert len(auditors)==1 and not (B/'jobs'/('held'+held)/'audit/TERMINATION.json').exists() and not (B/'run01'/('held'+held)/'COMPLETE_AUDIT.json').exists();folds[held]=dict(job=job,producer=producer,training_termination=term,audit_worker_pid=auditors[0]['pid'],audit_running=True,audit_progress_step_known=False)
 out=dict(state='verified_actual_producer_complete_numeric_audits_running',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue=queue,folds=folds,processes=procs,all1465_frozen_dependencies_reverified=True,completed_producer_folds=[2,3],fully_audited_new_folds=[],no_new_diagnostic_labels_scored=True,users7_8_labels_used=False,source_sixfold_result_available=False,complete_private_modelset_verified=False)
 (B/'LIVE_OBSERVATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(state=out['state'],auditors={h:r['audit_worker_pid'] for h,r in folds.items()},completed_producers=out['completed_producer_folds'],full_audits_complete=False)))
if __name__=='__main__':main()
