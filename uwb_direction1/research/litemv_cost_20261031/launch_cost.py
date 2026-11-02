"""One independently bounded synthetic profile after verified prior closure."""
import sys,json,importlib.util,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if __name__=='__main__':
 assert sys.argv[1:]==['--run'];sys.path.insert(0,'/workspace/uwb-litemv-engine-20261031');from readiness_guards import prior_closed
 policy=json.loads((B/'POLICY.json').read_bytes());assert prior_closed(policy['prior_workspace'])
 for n,h in json.loads((B/'COST_HASHES.json').read_bytes()).items():assert sha(B/n)==h,n
 assert not (B/'profile01').exists() and not (B/'supervisor01').exists()
 helper=Path('/workspace/uwb-multirocket-20261029/launch.py');assert sha(helper)=='b8f8038cc8242d20158b1e4870283fc61f884a039931fa08cf9940ea6b3d2eab';spec=importlib.util.spec_from_file_location('litemv_cost_supervisor',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 result=m.bounded_run([sys.executable,'-u','/workspace/uwb-litemv-cost-tools-20261031/profile_cost.py','--policy',str(B/'POLICY.json'),'--out',str(B/'profile01'),'--prior',policy['prior_workspace']],B/'supervisor01',620,590,600);print(json.dumps(result));raise SystemExit(0 if result['exit_code']==0 else 1)
