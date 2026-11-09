"""Fail-closed preregistration, private recovery and resource admission guards."""
from core import ROOT,read,sha,verify_sources
def sources():
    for n,h in read(ROOT/'FROZEN_HASHES.json').items():
        if sha(ROOT/n)!=h:raise PermissionError('Frozen source changed: '+n)
    verify_sources()
def require_probe():
    sources();r=read(ROOT/'PREREGISTRATION_RELEASE.json')
    if not(r['branch_head_verified'] and r['force'] is False and r['frozen_sha256']==sha(ROOT/'FROZEN_HASHES.json')):raise PermissionError('No exact source preregistration')
    from stream_archive import verify
    b=ROOT/'initial_backup';v=verify(read(b/'MANIFEST.json'),read(b/'READBACK_INPUT.json'))
    if v!=read(b/'READBACK_VERIFIED.json') or v['archive_sha256']!=r['initial_archive_sha256']:raise PermissionError('No full private initial recovery')
    generated=read(ROOT/'GENERATED_V2_CHECK.json')
    if not generated['passed']:raise PermissionError('Generated safeguards failed')
    for n,h in generated['sources'].items():
        if sha(ROOT/n)!=h:raise PermissionError('Generated evidence source changed')
def require_production(held):
    require_probe()
    if held not in range(1,9):raise PermissionError('Held user invalid')
    r=read(ROOT/'PRODUCTION_RELEASE.json')
    if r.get('production_admitted') is not True or r.get('protocol_sha256')!=sha(ROOT/'PROTOCOL.json'):raise PermissionError('Production not admitted')
    if not read(ROOT/'RESOURCE_GATE.json')['passed']:raise PermissionError('Cost/storage admission missing')
    if not read(ROOT/'NATIVE_OUTPUT_AND_OPTIMIZER_AUDIT.json')['passed']:raise PermissionError('Native audit admission incomplete')
