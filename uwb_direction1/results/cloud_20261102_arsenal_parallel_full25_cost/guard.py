from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4194304),b''):h.update(block)
    return h.hexdigest()
def ready():
    r=json.loads((B/'REMOTE_PREREGISTRATION.json').read_bytes())
    assert r['state']=='remote_cost_preregistration_exact_readback_verified' and r['branch_head_verified'] and r['cost_hashes_sha256']==sha(B/'COST_HASHES.json')
    for path,digest in json.loads((B/'COST_HASHES.json').read_bytes()).items():assert sha(B/path)==digest,path
    for path,digest in json.loads((B/'DEPENDENCIES.json').read_bytes())['files'].items():assert sha(path)==digest,path
    previous=json.loads(Path('/workspace/uwb-arsenal-full25-cost-20261102/VERIFIED_EXECUTION_CLOSURE_20261102.json').read_bytes())
    assert previous['no_old_profile_retry'] and not previous['actual_UWB_values_labels_fit']
    return json.loads((B/'POLICY.json').read_bytes())
