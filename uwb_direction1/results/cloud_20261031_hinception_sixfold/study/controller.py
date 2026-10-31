"""Two independently supervised fold jobs; no duplicate/retry/Drive/Git."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from readiness import read,ready
B=Path(__file__).resolve().parent
def main():
 lock=open(B/'queue.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg=read(B/'PROTOCOL.json');ready(cfg);assert not (B/'jobs').exists() and not (B/'run01').exists();(B/'jobs').mkdir();(B/'run01').mkdir();pending=list(cfg['new_held_users']);active={};logs={};completed=[];failed=[];state=dict(state='running_unscored_source_confirmation',pid=os.getpid(),automatic_retry=False,Drive_or_Git_capability=False,users7_8_labels_used=False)
 def record(**kw):state.update(kw);state.update(pending=pending,active={str(h):p.pid for h,p in active.items()},completed=completed,failed=failed,observed_unix=time.time());(B/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
 try:
  while pending or active:
   while pending and len(active)<cfg['queue_slots'] and not failed:
    held=pending.pop(0);logs[held]=(B/'jobs'/('held'+str(held)+'.log')).open('xb');active[held]=subprocess.Popen([sys.executable,'-u',str(B/'job.py'),'--held',str(held)],stdout=logs[held],stderr=subprocess.STDOUT,start_new_session=True);record()
   for held,p in list(active.items()):
    code=p.poll()
    if code is not None:
     logs[held].close();del active[held]
     if code==0:completed.append(held)
     else:failed.append(dict(held=held,exit_code=code))
     record()
   if failed and not active:raise RuntimeError('A fold failed; remaining queue paused without retry')
   if pending or active:time.sleep(5)
  record(state='all5_new_folds_audited_before_aggregate_scoring')
  with (B/'AGGREGATE.log').open('xb') as log:r=subprocess.run([sys.executable,'-u',str(B/'aggregate.py'),'--run'],stdout=log,stderr=subprocess.STDOUT,timeout=600)
  if r.returncode!=0:raise RuntimeError('Aggregate failed; preserve, no retry')
  record(state='source6_audited_complete_requires_full_private_modelset_backup_and_publication',pid=None,complete_private_backup_done=False)
 except BaseException as e:record(state='source_confirmation_incomplete',pid=None,error=repr(e),traceback=traceback.format_exc(),scientific_negative=False);raise
 finally:lock.close()
if __name__=='__main__':assert sys.argv[1:]==['--run'];main()
