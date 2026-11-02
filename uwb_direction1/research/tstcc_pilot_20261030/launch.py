"""Explicit one-shot independently bounded training or completion audit."""
import importlib.util,json,sys
from pathlib import Path
from run_pilot import ready,read,sha
from audit_complete import require_complete
B=Path(__file__).resolve().parent
if __name__=='__main__':
 assert sys.argv[1:] in (['--run'],['--audit']);audit=sys.argv[1]=='--audit';cfg=read(B/'PROTOCOL.json')
 if audit:require_complete(B);assert not (B/'COMPLETE_AUDIT.json').exists() and not (B/'audit_supervisor01').exists()
 else:ready(cfg);assert cfg['cost_gate_passed'] and not (B/'run01').exists() and not (B/'supervisor01').exists()
 for n,h in read(B/'FROZEN_HASHES.json').items():assert sha(B/n)==h,n
 helper=Path('/workspace/uwb-multirocket-20261029/launch.py');assert sha(helper)=='b8f8038cc8242d20158b1e4870283fc61f884a039931fa08cf9940ea6b3d2eab'
 spec=importlib.util.spec_from_file_location('tstcc_independent_supervisor',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 result=m.bounded_run([sys.executable,'-u',str(B/('audit_complete.py' if audit else 'run_pilot.py')),'--run'],B/('audit_supervisor01' if audit else 'supervisor01'),7200 if audit else 43100,7190 if audit else 43000,7200 if audit else 43100)
 if audit and result['exit_code']!=0:(B/'AUDIT_INCOMPLETE.json').write_text(json.dumps(dict(state='audit_incomplete',scientific_negative=False,receipt=result),indent=2))
 print(json.dumps(result));raise SystemExit(0 if result['exit_code']==0 else 1)
