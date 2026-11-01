"""Run once only after exact remote preregistration; independent bounded child."""
from pathlib import Path
import hashlib,importlib.util,json,sys
B=Path(__file__).resolve().parent
def main():
 marker=json.loads((B/'REMOTE_PREREGISTRATION.json').read_text());hashes=json.loads((B/'COST_HASHES.json').read_text())
 assert marker['state']=='remote_cost_preregistration_exact_readback_verified' and marker['cost_hashes_sha256']==hashlib.sha256((B/'COST_HASHES.json').read_bytes()).hexdigest()
 for name,digest in hashes.items():assert hashlib.sha256((B/name).read_bytes()).hexdigest()==digest
 assert not (B/'profile01').exists() and not (B/'supervisor01').exists()
 spec=importlib.util.spec_from_file_location('disjoint_bounded_cost',B/'independent_supervisor.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 result=m.bounded_run([sys.executable,'-u',str(B/'profile_cost.py'),'--policy',str(B/'POLICY.json'),'--out',str(B/'profile01'),'--prior','/workspace/uwb-hinception-sixfold-20261031'],B/'supervisor01',620,590,600)
 print(json.dumps(result));raise SystemExit(0 if result['exit_code']==0 else 1)
if __name__=='__main__':main()
