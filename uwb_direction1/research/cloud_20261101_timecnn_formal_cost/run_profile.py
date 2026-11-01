"""Single bounded synthetic profile only after exact remote readback."""
from pathlib import Path
import importlib.util,json,sys
from guard import read,sha
B=Path(__file__).resolve().parent
def main():
 m=read(B/'REMOTE_PREREGISTRATION.json');assert m['state']=='remote_cost_preregistration_exact_readback_verified' and m['cost_hashes_sha256']==sha(B/'COST_HASHES.json')
 for n,d in read(B/'COST_HASHES.json').items():assert sha(B/n)==d,n
 assert not (B/'profile01').exists() and not (B/'supervisor01').exists()
 spec=importlib.util.spec_from_file_location('TimeCNN_cost_supervisor',B/'independent_supervisor.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
 r=s.bounded_run([sys.executable,'-u',str(B/'profile_cost.py')],B/'supervisor01',620,590,600);print(json.dumps(r));raise SystemExit(0 if r['exit_code']==0 else 1)
if __name__=='__main__':main()
