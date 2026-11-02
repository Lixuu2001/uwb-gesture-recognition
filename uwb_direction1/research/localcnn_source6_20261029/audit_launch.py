"""One-shot read-only audit supervisor; independent wall/CPU limits and receipt."""
import importlib.util
import json
import sys
from pathlib import Path
from audit_complete import require_complete, digest

BASE = Path(__file__).resolve().parent

def main():
    assert sys.argv[1:] == ['--run']
    # No audit output directory and no arrays until the training supervisor exits.
    require_complete(BASE)
    output = BASE / 'COMPLETE_AUDIT.json'
    assert not output.exists()
    hashes = json.loads((BASE / 'AUDIT_HASHES.json').read_text())
    for path, expected in hashes.items():
        assert digest(BASE / path) == expected, path
    helper = Path('/workspace/uwb-multirocket-20261029/launch.py')
    assert digest(helper) == 'b8f8038cc8242d20158b1e4870283fc61f884a039931fa08cf9940ea6b3d2eab'
    spec = importlib.util.spec_from_file_location('audit_supervisor_existing', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.bounded_run([sys.executable, '-u', str(BASE / 'audit_complete.py'), '--out', str(output)],
                                BASE / 'audit_supervisor01', 1790, 1780, 1790)
    if result['exit_code'] != 0:
        (BASE / 'AUDIT_INCOMPLETE.json').write_text(json.dumps(dict(
            state='audit_incomplete', scientific_negative=False, receipt=result), indent=2))
    print(json.dumps(result))
    raise SystemExit(0 if result['exit_code'] == 0 else 1)

if __name__ == '__main__':
    main()
