"""No candidate import or timing until exact preregistration and predecessor closure."""
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
  r=read(B/'REMOTE_PREREGISTRATION.json')
  assert r['state']=='remote_cost_preregistration_exact_readback_verified' and r['cost_hashes_sha256']==sha(B/'COST_HASHES.json')
 for n,d in read(B/'COST_HASHES.json').items():assert sha(B/n)==d,n
 deps=read(B/'DEPENDENCIES.json')
 for p,d in deps['files'].items():assert sha(p)==d,p
 c=Path('/workspace/uwb-mlstmfcn-pilot-20261101');s=read(c/'run01/SUMMARY.json');a=read(c/'COMPLETE_AUDIT.json');v=read(c/'complete_backup/READBACK_VERIFIED.json');r=read(c/'complete_backup/DRIVE_BACKUP_RECEIPT.json')
 assert not s['predeclared_gate_passed'] and s['route']=='close_exact_full250_B128_seed_configuration_no_rescue' and s['complete_private_modelset_verified'] and not s['users7_8_labels_used']
 assert a['state']=='pass' and a['updates_replayed']==9250 and a['epochs_replayed']==250 and not a['diagnostic_labels_read']
 assert read(c/'audit_supervisor01/TERMINATION.json')['exit_code']==0 and v['complete_modelset_verified'] and r['complete_modelset_verified'] and v['archive_sha256']==r['archive_sha256']
 source=Path('/workspace/uwb-fcn-source-probe-20261101')
 for p,d in read(source/'PREPARATION_HASHES.json')['files'].items():assert sha(source/p)==d,p
 runtime=read('/workspace/uwb-litemv-source-probe-20261031/dependencies/DEPENDENCIES.json')
 root=Path('/workspace/uwb-resnet-keras-screen-20261030/deps')
 for p,d in runtime['files'].items():assert sha(root/p)==d,p
 return read(B/'POLICY.json')
