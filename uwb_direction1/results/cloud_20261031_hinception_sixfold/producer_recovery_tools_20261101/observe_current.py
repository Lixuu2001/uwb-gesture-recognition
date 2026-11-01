"""Read-only source-queue observation across mixed training/audit phases."""
from pathlib import Path
import datetime,json,os,subprocess,sys
B=Path('/workspace/uwb-hinception-sixfold-20261031')
def read(p):return json.loads(Path(p).read_bytes())
def state(p):
 value=read(p);keys=['state','held','pid','epoch','batch','total_steps','all_predictions_frozen','diagnostic_labels_read','users7_8_labels_used','wall_seconds','cpu_seconds','observed_unix','best_epoch','best_training_loss','automatic_retry'];return {k:value[k] for k in keys if k in value}
def process(pid,script):
 try:
  p=Path('/proc')/str(pid);args=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x];status={}
  for line in (p/'status').read_text().splitlines():
   if ':' in line:
    k,v=line.split(':',1)
    if k in ['Name','State','PPid','VmRSS','Threads']:status[k]=v.strip()
  if script not in args or status['State'].split()[0] in ['Z','X']:return None
  fields=(p/'stat').read_text().rsplit(')',1)[1].split();cpu=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK');return dict(pid=pid,command=args,status=status,process_cpu_seconds=cpu)
 except (FileNotFoundError,ProcessLookupError):return None
def observe():
 sys.path.insert(0,str(B));from readiness import ready
 ready(read(B/'PROTOCOL.json'));q=read(B/'STATE.json');procs=[];folds={};transitions=[];train=False;audit=False
 if q.get('pid'):
  p=process(q['pid'],str(B/'controller.py'))
  if p:procs.append(p)
  else:transitions.append('controller_process_not_verified')
 for held,pid in q['active'].items():
  job=state(B/'jobs'/('held'+held)/'STATE.json');fold=B/'run01'/('held'+held);worker=state(fold/'STATE.json') if (fold/'STATE.json').exists() else dict(state='not_yet_materialized');phase=job['state'];p=process(pid,str(B/'job.py'));current=None
  if p:procs.append(p)
  else:transitions.append('job'+held+'_process_not_verified')
  kind='run_fold.py' if phase=='training' else 'audit_fold.py' if phase=='numeric_audit' else None
  if kind:
   result=subprocess.run(['ps','--ppid',str(pid),'-o','pid='],capture_output=True,text=True)
   for child in result.stdout.split():
    candidate=process(int(child),str(B/kind))
    if candidate:
     assert int(candidate['status']['PPid'])==pid and candidate['command'][-2:]==['--held',held];assert current is None;current=candidate;procs.append(candidate)
   if current:
    train|=phase=='training';audit|=phase=='numeric_audit'
   else:transitions.append('fold'+held+'_child_transition_or_unverified')
  else:transitions.append('fold'+held+'_phase_'+phase)
  folds[held]=dict(phase=phase,job=job,worker=worker,active_process=current,audit_progress_step_known=False if phase=='numeric_audit' else None)
 completed_producers=[];audited=[]
 for held in range(2,7):
  p=B/'run01'/('held'+str(held))/'STATE.json'
  if p.exists() and read(p)['state']=='complete':completed_producers.append(held)
 for held in q['completed']:
  a=read(B/'run01'/('held'+str(held))/'COMPLETE_AUDIT.json');t=read(B/'jobs'/('held'+str(held))/'audit/TERMINATION.json');assert a['state']=='pass' and a['diagnostic_labels_read'] is False and t['exit_code']==0 and t['wall_timeout'] is False;audited.append(held)
 assert all(r['worker'].get('diagnostic_labels_read',False) is False for r in folds.values())
 out=dict(state='verified_source_queue_mixed_phase_observation' if not transitions else 'source_queue_phase_transition_requires_recheck',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue=q,folds=folds,processes=procs,completed_producer_folds=completed_producers,fully_audited_new_folds=sorted(audited),completed_audit_metadata_checked=True,new_completed_audit_artifact_hashes_reverified_by_this_observer=False,all1465_frozen_prerequisites_reverified=True,background_original_training_running=train,background_numeric_audit_running=audit,phase_transitions_or_unverified=transitions,diagnostic_scoring_performed=False,users7_8_labels_used=False,source6_summary_file_exists=(B/'run01/SUMMARY.json').exists(),complete_private_modelset_verified=False)
 (B/'LIVE_OBSERVATION.json').write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':
 o=observe();print(json.dumps(dict(state=o['state'],completed_producers=o['completed_producer_folds'],fully_audited=o['fully_audited_new_folds'],active_phases={h:r['phase'] for h,r in o['folds'].items()},active_children={h:r['active_process']['pid'] if r['active_process'] else None for h,r in o['folds'].items()},transitions=o['phase_transitions_or_unverified'],diagnostic_scoring=False)))
