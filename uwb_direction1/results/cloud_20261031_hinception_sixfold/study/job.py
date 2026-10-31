"""One fold supervisor: one training, one complete audit; no scoring/retry."""
import importlib.util,json,sys,time,traceback,os
from pathlib import Path
from readiness import read,sha,ready
B=Path(__file__).resolve().parent
def main(held):
 cfg=read(B/'PROTOCOL.json');ready(cfg);out=B/'jobs'/('held'+str(held));out.mkdir(exist_ok=False);state=dict(state='training',held=held,pid=os.getpid(),diagnostic_labels_read=False,automatic_retry=False)
 def record(**kw):state.update(kw);state['observed_unix']=time.time();(out/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
 try:
  helper=Path('/workspace/uwb-multirocket-20261029/launch.py');assert sha(helper)=='b8f8038cc8242d20158b1e4870283fc61f884a039931fa08cf9940ea6b3d2eab';spec=importlib.util.spec_from_file_location('source_confirmation_supervisor',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);record()
  t=m.bounded_run([sys.executable,'-u',str(B/'run_fold.py'),'--held',str(held)],out/'train',43100,43000,43100)
  if t['exit_code']!=0 or t['wall_timeout']:raise RuntimeError('Fold training failed; no retry')
  record(state='numeric_audit')
  t=m.bounded_run([sys.executable,'-u',str(B/'audit_fold.py'),'--held',str(held)],out/'audit',14400,14390,14400)
  if t['exit_code']!=0 or t['wall_timeout']:raise RuntimeError('Fold numeric audit failed; no retry')
  a=read(B/'run01'/('held'+str(held))/'COMPLETE_AUDIT.json');assert a['state']=='pass' and a['held']==held and a['diagnostic_labels_read'] is False
  record(state='complete_audited_unscored_fold',pid=None)
 except BaseException as e:record(state='incomplete',pid=None,error=repr(e),traceback=traceback.format_exc(),scientific_negative=False);raise
if __name__=='__main__':assert len(sys.argv)==3 and sys.argv[1]=='--held' and int(sys.argv[2]) in [2,3,4,5,6];main(int(sys.argv[2]))
