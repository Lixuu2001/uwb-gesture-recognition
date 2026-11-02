from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parent
def ready():
    manifest=json.loads((B/'COST_HASHES.json').read_bytes())
    for row in manifest['files']:
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    for row in json.loads((B/'DEPENDENCIES.json').read_bytes())['files']:
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    p=json.loads((B/'REMOTE_PREREGISTRATION.json').read_bytes())
    assert p['state']=='exact_remote_readback_verified_before_synthetic_timing' and p['branch_head_verified']
    assert p['cost_hashes_sha256']==hashlib.sha256((B/'COST_HASHES.json').read_bytes()).hexdigest()
    return json.loads((B/'POLICY.json').read_bytes())
