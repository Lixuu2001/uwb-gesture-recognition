"""One-shot bounded audits, never a watcher, retry or training launcher."""
import importlib.util,json,sys
from pathlib import Path
from audit_complete import require_complete,require_phase,digest,PHASES
BASE=Path(__file__).resolve().parent
def main():
 assert sys.argv[1:] in (['--run'],['--ssl-phase-only'])
 phase=sys.argv[1:] == ['--ssl-phase-only']
 if phase: require_phase(BASE,*PHASES[0])
 else: require_complete(BASE)
 output=BASE/('SSL_PHASE_AUDIT.json' if phase else 'COMPLETE_AUDIT.json');assert not output.exists()
 for n,h in json.loads((BASE/'AUDIT_HASHES.json').read_text()).items(): assert digest(BASE/n)==h,n
 helper=Path('/workspace/uwb-multirocket-20261029/launch.py');assert digest(helper)=='b8f8038cc8242d20158b1e4870283fc61f884a039931fa08cf9940ea6b3d2eab'
 spec=importlib.util.spec_from_file_location('tslanet_audit_supervisor',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 argv=[sys.executable,'-u',str(BASE/'audit_complete.py'),'--out',str(output)]+(['--ssl-phase-only'] if phase else [])
 result=module.bounded_run(argv,BASE/('ssl_audit_supervisor01' if phase else 'audit_supervisor01'),180 if phase else 890,170 if phase else 880,180 if phase else 890)
 if result['exit_code']!=0: (BASE/('SSL_AUDIT_INCOMPLETE.json' if phase else 'AUDIT_INCOMPLETE.json')).write_text(json.dumps(dict(state='audit_incomplete',scientific_negative=False,receipt=result),indent=2))
 print(json.dumps(result));raise SystemExit(0 if result['exit_code']==0 else 1)
if __name__=='__main__':main()
