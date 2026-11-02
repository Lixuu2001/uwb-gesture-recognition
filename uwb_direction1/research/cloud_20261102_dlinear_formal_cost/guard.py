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
 t=Path('/workspace/uwb-tlenet-formal-cost-20261102');r=read(t/'VERIFIED_COST_RESULT.json');s=read(t/'supervisor01/TERMINATION.json')
 assert r['profile_completed_successfully'] and not r['resource_gate_passed'] and not r['accuracy_measured'] and not r['real_UWB_values_labels_fit'] and s['exit_code']==0 and not s['wall_timeout']
 return read(B/'POLICY.json')
