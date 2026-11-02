"""Explicit one-shot independently bounded synthetic CPU cost screen."""
import sys,json,hashlib,importlib.util
from pathlib import Path
B=Path(__file__).resolve().parent
if __name__=='__main__':
 assert sys.argv[1:]==['--run'] and not (B/'cost_supervisor01').exists() and not (B/'COST_CHECK.json').exists()
 for n,h in json.loads((B/'COST_HASHES.json').read_text()).items():assert hashlib.sha256((B/n).read_bytes()).hexdigest()==h,n
 p=Path('/workspace/uwb-multirocket-20261029/launch.py');assert hashlib.sha256(p.read_bytes()).hexdigest()=='b8f8038cc8242d20158b1e4870283fc61f884a039931fa08cf9940ea6b3d2eab'
 s=importlib.util.spec_from_file_location('cost_supervisor',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);r=m.bounded_run([sys.executable,'-u',str(B/'cost_check.py')],B/'cost_supervisor01',600,590,600);print(json.dumps(r));raise SystemExit(0 if r['exit_code']==0 else 1)
