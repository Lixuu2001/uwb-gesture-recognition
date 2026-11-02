"""Bounded completed-run training-reconstruction/SVM audit; no auto launch."""
import importlib.util,json,sys
from pathlib import Path
from audit_complete import require_complete,sha
B=Path(__file__).resolve().parent
if __name__=='__main__':
 assert sys.argv[1:]==['--run'];require_complete(B);output=B/'COMPLETE_AUDIT.json';assert not output.exists()
 for n,h in json.loads((B/'FROZEN_HASHES.json').read_text()).items():assert sha(B/n)==h,n
 helper=Path('/workspace/uwb-multirocket-20261029/launch.py');assert sha(helper)=='b8f8038cc8242d20158b1e4870283fc61f884a039931fa08cf9940ea6b3d2eab'
 spec=importlib.util.spec_from_file_location('ts2vec_complete_audit_supervisor',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 result=m.bounded_run([sys.executable,'-u',str(B/'audit_complete.py'),'--out',str(output)],B/'audit_supervisor01',3600,3590,3600)
 if result['exit_code']!=0:(B/'AUDIT_INCOMPLETE.json').write_text(json.dumps(dict(state='audit_incomplete',scientific_negative=False,receipt=result),indent=2))
 print(json.dumps(result));raise SystemExit(0 if result['exit_code']==0 else 1)
