"""Verify source/runtime bytes and closed predecessor before candidate import."""
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
 c=Path('/workspace/uwb-timecnn-pilot-20261101');s=read(c/'run01/SUMMARY.json');a=read(c/'COMPLETE_AUDIT.json');t=read(c/'audit_supervisor01/TERMINATION.json');r=read(c/'complete_backup/DRIVE_BACKUP_RECEIPT.json');v=read(c/'complete_backup/READBACK_VERIFIED.json')
 assert not s['predeclared_gate_passed'] and s['route']=='close_exact_full2000_B16_seed_configuration_no_rescue' and s['complete_private_modelset_verified']
 assert a['state']=='pass' and a['updates_replayed']==588000 and t['exit_code']==0 and not t['wall_timeout']
 assert r['scope']==v['scope']=='complete_audited_full2000_modelset' and r['complete_modelset_verified'] and v['complete_modelset_verified'] and r['archive_sha256']==v['archive_sha256']
 return read(B/'POLICY.json')
