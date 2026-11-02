"""Explicit bounded audit-only engineering repair; producer never restarted."""
import importlib.util,json,sys
from pathlib import Path
from audit_complete import require_complete,sha
B=Path(__file__).resolve().parent
if __name__=='__main__':
 assert sys.argv[1:]==['--run'];require_complete(B)
 assert not (B/'COMPLETE_AUDIT.json').exists() and not (B/'audit_supervisor02').exists()
 repair=json.loads((B/'AUDIT_ENGINEERING_REPAIR01.json').read_text())
 assert sha(B/'audit_complete.py')==repair['original_auditor_sha256'] and sha(B/'audit_complete_repair01.py')==repair['repaired_auditor_sha256']
 for n,h in json.loads((B/'FROZEN_HASHES.json').read_text()).items():assert sha(B/n)==h,n
 helper=Path('/workspace/uwb-multirocket-20261029/launch.py');assert sha(helper)=='b8f8038cc8242d20158b1e4870283fc61f884a039931fa08cf9940ea6b3d2eab'
 spec=importlib.util.spec_from_file_location('repair_audit_supervisor',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 result=m.bounded_run([sys.executable,'-u',str(B/'audit_complete_repair01.py'),'--out',str(B/'COMPLETE_AUDIT.json')],B/'audit_supervisor02',3600,3590,3600)
 receipt=dict(state='pass' if result['exit_code']==0 and not result['wall_timeout'] else 'incomplete',termination=result,supervisor_folder='audit_supervisor02',repair_record_sha256=sha(B/'AUDIT_ENGINEERING_REPAIR01.json'),repaired_auditor_sha256=sha(B/'audit_complete_repair01.py'))
 (B/'AUDIT_COMPLETION_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));raise SystemExit(0 if receipt['state']=='pass' else 1)
