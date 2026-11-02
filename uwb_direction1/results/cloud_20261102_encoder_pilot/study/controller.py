"""One bounded training child then full audit child, no retry, no scoring."""
from pathlib import Path
import fcntl,importlib.util,os,sys,time,traceback
from guards import ready
from artifacts import write
B=Path(__file__).resolve().parent
def main():
 lock=(B/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if any((B/n).exists() for n in ['run01','supervisor01','audit_supervisor01','STATE.json']):raise RuntimeError('Inspect existing run; no duplicate launch')
 c=ready();s={'state':'starting','pid':os.getpid(),'automatic_retry':False,'Drive_or_Git_capability':False,'users7_8_labels_used':False}
 def record(**kw):s.update(kw);s['observed_unix']=time.time();write(B/'STATE.json',s)
 spec=importlib.util.spec_from_file_location('Encoder_independent_supervisor',B/'independent_supervisor.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 try:
  record(state='training');t=m.bounded_run([sys.executable,'-u',str(B/'run_pilot.py')],B/'supervisor01',c['train_wall_seconds'],c['train_CPU_soft'],c['train_CPU_hard'])
  if t['exit_code']!=0 or t['wall_timeout']:raise RuntimeError('Producer failed/incomplete; preserve and repair, no retry')
  record(state='full_numeric_audit');t=m.bounded_run([sys.executable,'-u',str(B/'audit_complete.py')],B/'audit_supervisor01',c['audit_wall_seconds'],c['audit_CPU_soft'],c['audit_CPU_hard'])
  if t['exit_code']!=0 or t['wall_timeout']:raise RuntimeError('Full audit failed/incomplete; preserve and repair, no retry')
  record(state='complete_audited_unscored_requires_private_backup_and_finalization',pid=None,complete_private_backup_done=False)
 except BaseException as e:record(state='incomplete_requires_inspection',pid=None,error=repr(e),traceback=traceback.format_exc(),scientific_negative=False);raise
 finally:lock.close()
if __name__=='__main__':main()
