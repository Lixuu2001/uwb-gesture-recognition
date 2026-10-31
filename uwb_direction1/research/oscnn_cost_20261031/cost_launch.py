"""One-shot independently bounded cost process, gated by completed ResNet close."""
import sys,json,importlib.util,hashlib
from pathlib import Path
sys.path.insert(0,'/workspace/uwb-oscnn-engine-20261031')
from readiness_guards import prior_closed
B=Path(__file__).resolve().parent
if __name__=='__main__':
 assert sys.argv[1:]==['--run'] and not (B/'cost_supervisor01').exists() and not (B/'COST_CHECK.json').exists();prior_closed('/workspace/uwb-resnet-keras-pilot-20261031')
 for n,h in json.loads((B/'COST_HASHES.json').read_text()).items():assert hashlib.sha256((B/n).read_bytes()).hexdigest()==h,n
 p=Path('/workspace/uwb-multirocket-20261029/launch.py');assert hashlib.sha256(p.read_bytes()).hexdigest()=='b8f8038cc8242d20158b1e4870283fc61f884a039931fa08cf9940ea6b3d2eab';spec=importlib.util.spec_from_file_location('OSCNNcostsupervisor',p);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);r=s.bounded_run([sys.executable,'-u',str(B/'cost_check.py')],B/'cost_supervisor01',600,590,600);print(json.dumps(r));raise SystemExit(0 if r['exit_code']==0 else 1)
