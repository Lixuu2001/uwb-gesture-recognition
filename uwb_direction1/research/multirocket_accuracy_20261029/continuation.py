"""Single-use, locked continuation: finish CNN -> audit -> fixed pilot -> audit.

No new experimental configuration is selected here. Fail closed; no auto retries.
"""
import fcntl,hashlib,json,os,sys,time,traceback
from pathlib import Path
from launch import bounded_run
B=Path(__file__).resolve().parent
C=Path('/workspace/uwb-mvglf-20261029')
O=B/'continuation01'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def decision(state,alive,receipts):
 if alive:return 'wait'
 if state=='complete' and receipts:return 'ready'
 return 'blocked'
def alive(pid,script):
 try:return str(script).encode() in Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')
 except FileNotFoundError:return False
def verify_frozen():
 for p,h in json.loads((B/'FROZEN_HASHES.json').read_text()).items():assert sha(B/p)==h
 for p,h in json.loads((B/'CONTINUATION_HASHES.json').read_text()).items():assert sha(B/p)==h
def main():
 assert sys.argv[1:]==['--run']
 O.mkdir(exist_ok=True)
 lock=open(O/'controller.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (O/'STATE.json').exists(),'existing continuation needs diagnosis, no automatic restart'
 started=time.monotonic();state=dict(pid=os.getpid(),state='checking',analysis_started=False,
  controller_sha256=sha(Path(__file__)),cnn_audit_started=False,multirocket_started=False,
  multirocket_audit_started=False,scientific_negative=False)
 def record(**kw):
  state.update(kw);state.update(observed_unix=time.time(),elapsed_wall_seconds=time.monotonic()-started)
  p=O/'STATE.tmp';p.write_text(json.dumps(state,indent=2));p.replace(O/'STATE.json')
 def checked_phase(name,argv,wall,soft,hard):
  record(state='running_'+name,phase_argv=argv)
  receipt=bounded_run(argv,O/name,wall,soft,hard)
  record(last_phase=name,last_phase_receipt=receipt)
  assert receipt['exit_code']==0 and not receipt['wall_timeout'],name+' incomplete; inspect exact receipt, do not retry automatically'
 try:
  verify_frozen();record(state='waiting_for_original_cnn')
  while True:
   assert time.monotonic()-started<14*3600,'controller wait ceiling reached'
   previous=json.loads((C/'run01/STATE.json').read_text())
   lives=alive(42028,C/'train.py')
   receipts=all((C/'run01'/p).exists() for p in ['FILES.json','SUMMARY.json','full/ARM_COMPLETE.json','local/ARM_COMPLETE.json'])
   choice=decision(previous['state'],lives,receipts)
   record(state='waiting_for_original_cnn',cnn_state_observed=previous['state'],
    cnn_arm_observed=previous.get('arm'),cnn_epoch_observed=previous.get('epoch'),
    original_worker_alive=lives)
   if choice=='ready':break
   assert choice!='blocked','upstream worker absent/incomplete: diagnose instead of restarting'
   time.sleep(20)
  verify_frozen()
  ca=C/'COMPLETE_AUDIT.json'
  # Reuse only a complete, identity-bound audit; never overwrite or re-run it.
  if ca.exists():
   prior=json.loads(ca.read_text())
   assert prior['state']=='pass' and prior['scope']=='complete' and prior['auditor_sha256']==sha(C/'audit.py')
   for path,digest in json.loads((C/'run01/FILES.json').read_text()).items():assert sha(C/'run01'/path)==digest
   record(cnn_complete_audit_reused=True)
  else:
   record(cnn_audit_started=True)
   checked_phase('cnn_audit01',[sys.executable,str(C/'audit.py'),'--out',str(ca)],895,870,890)
  audit=json.loads(ca.read_text());assert audit['state']=='pass' and audit['scope']=='complete'
  assert audit['identity']==json.loads((C/'run01/IDENTITY.json').read_text())
  summary=json.loads((C/'run01/SUMMARY.json').read_text());assert audit['scores']==summary['scores']
  record(cnn_audit_passed=True,cnn_scores=audit['scores'],cnn_gate_passed=summary['full_gate_passed'])
  verify_frozen();assert not (B/'run01').exists() and not (B/'supervisor01').exists()
  record(multirocket_started=True)
  checked_phase('multirocket01',[sys.executable,str(B/'launch.py'),'--run'],1800,1780,1790)
  assert json.loads((B/'run01/STATE.json').read_text())['state']=='complete'
  verify_frozen();record(multirocket_audit_started=True)
  output=B/'COMPLETE_RESULT_AUDIT.json';assert not output.exists()
  checked_phase('multirocket_audit01',[sys.executable,str(B/'audit_result.py'),'--out',str(output)],895,870,890)
  audit=json.loads(output.read_text());assert audit['state']=='pass'
  record(state='complete_registered_queue',pid=None,multirocket_scores=audit['scores'],
   multirocket_gate_passed=audit['gate_passed'],new_scientific_experiments_started=1,
   independent_validation=False,novel_contribution_validated=False,
   next_action='preserve private artifacts, review frozen gates; any sixfold expansion requires a separate protocol')
 except BaseException as e:
  record(state='blocked_or_incomplete',pid=None,error=repr(e),traceback=traceback.format_exc(),scientific_negative=False)
  raise
if __name__=='__main__':main()
