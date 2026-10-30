"""No automatic real-model launch or retry; bounded synthetic timer only."""
import importlib.util,json,sys,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent
if __name__=='__main__':
 assert sys.argv[1:]==['--run'] and not (BASE/'COST_CHECK.json').exists()
 for n,h in json.loads((BASE/'COST_HASHES.json').read_text()).items(): assert hashlib.sha256((BASE/n).read_bytes()).hexdigest()==h,n
 helper=Path('/workspace/uwb-multirocket-20261029/launch.py');assert hashlib.sha256(helper.read_bytes()).hexdigest()=='b8f8038cc8242d20158b1e4870283fc61f884a039931fa08cf9940ea6b3d2eab'
 spec=importlib.util.spec_from_file_location('convtran_cost_supervisor',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 result=module.bounded_run([sys.executable,'-u',str(BASE/'cost_check.py')],BASE/'cost_supervisor01',420,410,420)
 print(json.dumps(result));raise SystemExit(0 if result['exit_code']==0 else 1)
