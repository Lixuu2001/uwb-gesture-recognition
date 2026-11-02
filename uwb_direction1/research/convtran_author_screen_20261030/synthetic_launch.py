"""Explicit one-shot synthetic check under an independent supervisor."""
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
if __name__=='__main__':
 assert sys.argv[1:]==['--run'];assert not (BASE/'SYNTHETIC_CHECK.json').exists()
 helper=Path('/workspace/uwb-multirocket-20261029/launch.py')
 import hashlib
 assert hashlib.sha256(helper.read_bytes()).hexdigest()=='b8f8038cc8242d20158b1e4870283fc61f884a039931fa08cf9940ea6b3d2eab'
 spec=importlib.util.spec_from_file_location('convtran_synthetic_supervisor',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 result=module.bounded_run([sys.executable,'-u',str(BASE/'synthetic_check.py')],BASE/'synthetic_supervisor01',300,290,300)
 print(json.dumps(result));raise SystemExit(0 if result['exit_code']==0 else 1)
