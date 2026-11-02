"""Explicit bounded validation-only audit; never launches/retries training."""
import importlib.util,json,sys
from pathlib import Path
from audit_pretrained_phase import require_ready,digest
BASE=Path(__file__).resolve().parent
if __name__=='__main__':
 assert sys.argv[1:]==['--run'];require_ready(BASE);output=BASE/'PRETRAINED_PHASE_AUDIT.json';assert not output.exists()
 for index in ('AUDIT_HASHES.json','PRETRAINED_PHASE_AUDIT_HASHES.json'):
  for n,h in json.loads((BASE/index).read_text()).items():assert digest(BASE/n)==h,n
 helper=Path('/workspace/uwb-multirocket-20261029/launch.py');assert digest(helper)=='b8f8038cc8242d20158b1e4870283fc61f884a039931fa08cf9940ea6b3d2eab'
 spec=importlib.util.spec_from_file_location('pretrained_phase_audit_supervisor',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 result=module.bounded_run([sys.executable,'-u',str(BASE/'audit_pretrained_phase.py'),'--out',str(output)],BASE/'pretrained_audit_supervisor01',300,290,300)
 print(json.dumps(result));raise SystemExit(0 if result['exit_code']==0 else 1)
