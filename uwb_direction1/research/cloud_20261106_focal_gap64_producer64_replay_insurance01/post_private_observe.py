"""Observe original completed producer and SAME live replay; no forward, fit or score."""
from pathlib import Path
import sys,json,hashlib,io,time,copy
S=Path('/workspace/uwb-focal-gap64-20261106');D=Path(__file__).resolve().parent
sys.path.insert(0,str(S))
from common import read,write,sha,ready
from controller import identity
def main():
 ready(True);old=read(S/'run/execution/replay/IDENTITY.json');ctrl=read(S/'CONTROLLER_IDENTITY.json')
 for q in [old,ctrl]:
  now=identity(q['pid']);assert now and now['state']!='Z' and all(now[k]==q[k] for k in ['pid','startticks','boot_id'])
 assert not(S/'FINALIZE_ONCE.lock').exists() and not(S/'SUMMARY.json').exists()
 for _ in range(4):
  p=read(S/'run/replay/PROGRESS.json');raw=(S/'run/replay'/p['slot']).read_bytes()
  if hashlib.sha256(raw).hexdigest()==p['checkpoint_sha256']:break
 else:raise AssertionError('No coherent current native replay state; retain original process')
 import engine as e
 state=e.torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);rec=read(S/f'run/replay/epoch{p["epoch"]:03}.json');assert e.semantic(state)==rec['state_sha256'] and state['updates']==p['updates']
 assert rec==read(S/f'run/train/epoch{p["epoch"]:03}.json')
 m,o,g=e.make();e.restore(state,m,o,g);assert e.semantic(e.capture(m,o,g,state['epoch'],state['updates'],state['best_key'],state['best_model'],state['best_epoch']))==e.semantic(state)
 prefix=(S/'run/replay/steps.jsonl').read_bytes()[:p['prefix_bytes']];assert len(prefix.splitlines())==p['updates'] and hashlib.sha256(prefix).hexdigest()==p['prefix_sha256']
 assert prefix==(S/'run/train/steps.jsonl').read_bytes()[:p['prefix_bytes']]
 proof=read(D/'private_backup/READBACK_VERIFIED.json');assert proof['tracked_payload_members_verified']==read(D/'private_backup/MANIFEST.json')['tracked_payload_files'] and not proof['complete_modelset_verified']
 ledger=read(S/'RESOURCE_LEDGER.json');assert ledger['stage']=='replay' and ledger['wall']<43200 and ledger['cpu']<43200 and ledger['RSS_KiB']<12582912
 q=dict(state='original_producer64_complete_SAME_OWN_fresh_replay_active_native_exact_private_partial_insurance_verified_unscored',observed_unix=time.time(),controller=identity(ctrl['pid']),replay_worker=identity(old['pid']),replay_progress=p,replay_native_optimizer_allRNG_restore_recapture_exact=True,replay_committed_epoch_and_steps_equal_producer=True,producer_complete=read(S/'run/train/COMPLETE.json'),producer_actual_exit=read(S/'run/execution/train/TERMINATION.json'),resource_ledger=ledger,private_insurance_verified_payloads=proof['tracked_payload_members_verified'],complete_modelset_verified=False,scientific25_unchanged=True,new_model_started_or_prior_restart=False,budget_reset=False,diagnostic_labels_read=False,users7_8_labels_used=False)
 write(D/'POST_PRIVATE_OBSERVATION.json',q);print(json.dumps(q))
if __name__=='__main__':main()
