"""Byte and closed-predecessor guard before candidate import or synthetic timing."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def ready(marker=True):
 if marker:
  r=read(B/'REMOTE_PREREGISTRATION.json');assert r['state']=='remote_cost_preregistration_exact_readback_verified' and r['cost_hashes_sha256']==sha(B/'COST_HASHES.json')
 for n,d in read(B/'COST_HASHES.json').items():assert sha(B/n)==d,n
 for p,d in read(B/'DEPENDENCIES.json')['files'].items():assert sha(p)==d,p
 f=Path('/workspace/uwb-fcn-formal-cost-20261101');v=read(f/'VERIFIED_COST_RESULT.json');t=read(f/'supervisor01/TERMINATION.json')
 assert v['profile_completed_successfully'] and not v['resource_gate_passed'] and not v['accuracy_measured'] and not v['real_UWB_values_labels_fit'] and v['no_smaller_epoch_batch_fallback']
 assert (read(f/'POLICY.json')['epochs'],read(f/'POLICY.json')['batch_size'])==(2000,16) and t['exit_code']==0 and not t['wall_timeout']
 assert v['result_sha256']==sha(f/'profile01/COST_CHECK.json') and v['termination_sha256']==sha(f/'supervisor01/TERMINATION.json')
 c=Path('/workspace/uwb-mlstmfcn-pilot-20261101');s=read(c/'run01/SUMMARY.json');r=read(c/'complete_backup/DRIVE_BACKUP_RECEIPT.json')
 assert not s['predeclared_gate_passed'] and s['complete_private_modelset_verified'] and s['route']=='close_exact_full250_B128_seed_configuration_no_rescue' and r['complete_modelset_verified']
 source=Path('/workspace/uwb-timecnn-source-probe-20261101')
 for p,d in read(source/'PREPARATION_HASHES.json')['files'].items():assert sha(source/p)==d,p
 runtime=read('/workspace/uwb-litemv-source-probe-20261031/dependencies/DEPENDENCIES.json');root=Path('/workspace/uwb-resnet-keras-screen-20261030/deps')
 for p,d in runtime['files'].items():assert sha(root/p)==d,p
 return read(B/'POLICY.json')
