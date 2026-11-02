"""Metadata-only observer handles successful controller exit or zombie.
No model/data/gesture-label imports. Scientific study stays frozen.
"""
from pathlib import Path
import datetime,json,subprocess
B=Path('/workspace/uwb-encoder-pilot-20261102');HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def proc(pid):
 root=Path(f'/proc/{pid}')
 try:
  cmd=root.joinpath('cmdline').read_bytes().split(b'\0');status={}
  for line in root.joinpath('status').read_text().splitlines():
   if ':' in line:
    k,v=line.split(':',1)
    if k in ['State','PPid','VmRSS','Threads']:status[k]=v.strip()
 except FileNotFoundError:return dict(pid=pid,present=False,active=False,command=[],status={})
 return dict(pid=pid,present=True,active=not status['State'].startswith('Z'),command=[x.decode() for x in cmd if x],status=status)
def completed(c,state,train,train_term,audit_term,cert,active_children):
 if state.get('state')!='complete_audited_unscored_requires_private_backup_and_finalization':return False
 if state.get('pid') is not None or active_children:raise RuntimeError('Completion conflicts with active work')
 for t in [train_term,audit_term]:
  if not t or t.get('exit_code')!=0 or t.get('wall_timeout') is not False:raise RuntimeError('Missing successful termination')
 if not train or train.get('state')!='complete' or train.get('pid') is not None or not train.get('all_predictions_frozen'):raise RuntimeError('Producer incomplete')
 if not cert or cert.get('state')!='pass' or cert.get('updates_replayed')!=c['max_training_updates'] or cert.get('epochs_replayed')!=c['epochs'] or cert.get('diagnostic_labels_read') is not False:raise RuntimeError('Missing complete fixed audit evidence')
 return True
def main():
 launch=read(HERE/'LAUNCH_RECEIPT.json');c=read(B/'PROTOCOL.json');state=read(B/'STATE.json') if (B/'STATE.json').exists() else {'state':'controller_preflight'}
 controller=proc(launch['pid']);children=[]
 if controller['active']:
  if str(B/'controller.py') not in controller['command']:raise RuntimeError('Controller PID reused; inspect actual process')
  p=subprocess.run(['ps','--ppid',str(launch['pid']),'-o','pid='],capture_output=True,text=True,check=True)
  children=[proc(int(x)) for x in p.stdout.split()]
 active=[x for x in children if x['active']]
 for child in active:
  if not any(str(B/n) in child['command'] for n in ['run_pilot.py','audit_complete.py']):raise RuntimeError('Unexpected active child')
 def optional(n):p=B/n;return read(p) if p.exists() else None
 train=optional('run01/STATE.json');tt=optional('supervisor01/TERMINATION.json');at=optional('audit_supervisor01/TERMINATION.json');cert=optional('COMPLETE_AUDIT.json')
 done=completed(c,state,train,tt,at,cert,active)
 if not done and not controller['active']:raise RuntimeError('Controller inactive without complete audit evidence')
 r=dict(state='completed_Encoder_audit_metadata_verified' if done else 'existing_Encoder_pilot_OS_and_metadata_verified',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller=controller,children=children,controller_state=state,training_state=train,training_termination=tt,audit_termination=at,full_audit_metadata_verified=done,diagnostic_score_exists=(B/'run01/SUMMARY.json').exists(),configuration_changed=False,labels_loaded_by_observer=False)
 (HERE/'LIVE_OBSERVATION.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
