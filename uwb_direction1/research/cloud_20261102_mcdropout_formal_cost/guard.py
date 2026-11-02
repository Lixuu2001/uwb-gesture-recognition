from pathlib import Path
import json,hashlib
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
 for n,d in read(B/'DEPENDENCIES.json')['files'].items():assert sha(n)==d,n
 d=Path('/workspace/uwb-dlinear-pilot-20261102');s=read(d/'run01/SUMMARY.json');a=read(d/'COMPLETE_AUDIT.json');t=read(d/'audit_supervisor01/TERMINATION.json');r=read(d/'complete_backup/DRIVE_BACKUP_RECEIPT.json')
 assert not s['predeclared_gate_passed'] and s['route']=='close_exact_DLinear10_B32_patience3_seed_configuration_no_rescue' and s['complete_private_modelset_verified'] and a['state']=='pass' and t['exit_code']==0 and not t['wall_timeout'] and r['complete_modelset_verified']
 return read(B/'POLICY.json')
