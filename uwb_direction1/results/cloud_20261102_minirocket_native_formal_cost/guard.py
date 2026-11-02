from pathlib import Path
import json,hashlib
B=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
    return h.hexdigest()
def ready(marker=True):
    if marker:
        r=json.loads((B/'REMOTE_PREREGISTRATION.json').read_text())
        assert r['state']=='remote_cost_preregistration_exact_readback_verified'
        assert r['cost_hashes_sha256']==sha(B/'COST_HASHES.json')
    for p,d in json.loads((B/'COST_HASHES.json').read_text()).items():assert sha(B/p)==d,p
    for p,d in json.loads((B/'DEPENDENCIES.json').read_text())['files'].items():assert sha(p)==d,p
    m=Path('/workspace/uwb-rocket-pilot-v2-20261102/run01/SUMMARY.json')
    s=json.loads(m.read_text());assert not s['predeclared_gate_passed'] and s['complete_private_modelset_verified']
    return json.loads((B/'POLICY.json').read_text())
