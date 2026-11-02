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
 for n,d in read(B/'DEPENDENCIES.json')['files'].items():assert sha(n)==d,n
 e=Path('/workspace/uwb-encoder-pilot-20261102');s=read(e/'run01/SUMMARY.json');r=read(e/'complete_backup/DRIVE_BACKUP_RECEIPT.json');a=read(e/'COMPLETE_AUDIT.json');t=read(e/'audit_supervisor01/TERMINATION.json')
 assert not s['predeclared_gate_passed'] and s['route']=='close_exact_full100_B12_seed_configuration_no_rescue' and s['complete_private_modelset_verified']
 assert r['scope']=='complete_audited_full100_modelset' and r['complete_modelset_verified'] and a['state']=='pass' and t['exit_code']==0 and not t['wall_timeout']
 return read(B/'POLICY.json')
