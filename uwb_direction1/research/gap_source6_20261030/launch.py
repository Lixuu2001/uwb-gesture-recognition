"""Explicit one-shot launch; independent single-thread budget supervisor."""
import importlib.util,json,sys
from pathlib import Path
from train import sha
B=Path(__file__).resolve().parent
if __name__=='__main__':
 assert sys.argv[1:]==['--run'] and not (B/'run01').exists()
 for n,h in json.loads((B/'FROZEN_HASHES.json').read_text()).items(): assert sha(B/n)==h,n
 p=Path('/workspace/uwb-multirocket-20261029/launch.py');assert sha(p)=='b8f8038cc8242d20158b1e4870283fc61f884a039931fa08cf9940ea6b3d2eab'
 spec=importlib.util.spec_from_file_location('gap_source6_supervisor',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 receipt=module.bounded_run([sys.executable,'-u',str(B/'train.py'),'--run'],B/'supervisor01',43100,43000,43100)
 print(json.dumps(receipt));raise SystemExit(0 if receipt['exit_code']==0 else 1)
