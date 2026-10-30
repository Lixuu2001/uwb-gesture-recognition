"""Explicit one-shot real pilot with independent hard limits; no auto retry."""
import importlib.util,json,sys
from pathlib import Path
from run_pilot import ready,sha
B=Path(__file__).resolve().parent
if __name__=='__main__':
 assert sys.argv[1:]==['--run'];cfg=json.loads((B/'PROTOCOL.json').read_text());ready(cfg);assert cfg['cost_gate_passed'] and not (B/'run01').exists()
 for n,h in json.loads((B/'FROZEN_HASHES.json').read_text()).items():assert sha(B/n)==h,n
 helper=Path('/workspace/uwb-multirocket-20261029/launch.py');assert sha(helper)=='b8f8038cc8242d20158b1e4870283fc61f884a039931fa08cf9940ea6b3d2eab'
 spec=importlib.util.spec_from_file_location('ts2vec_pilot_supervisor',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 result=m.bounded_run([sys.executable,'-u',str(B/'run_pilot.py'),'--run'],B/'supervisor01',43100,43000,43100)
 print(json.dumps(result));raise SystemExit(0 if result['exit_code']==0 else 1)
