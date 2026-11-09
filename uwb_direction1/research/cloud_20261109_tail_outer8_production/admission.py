"""Versioned production admission; old failed gate and successful probe stay immutable."""
import shutil
from pathlib import Path
from core import ROOT,read,sha,verify_sources
PREP=Path('/workspace/uwb-tail-outer8-20261109')
def sources():
    for n,h in read(ROOT/'FROZEN_HASHES.json').items():
        if sha(ROOT/n)!=h:raise PermissionError('Frozen production source changed: '+n)
    verify_sources()
def require_probe():raise PermissionError('Successful preparation probe is closed; never repeat it')
def require_production(held):
    if type(held) is not int or held not in range(1,9):raise PermissionError('Held user invalid')
    sources();r=read(ROOT/'PRODUCTION_RELEASE.json');p=read(ROOT/'REMOTE_PREREGISTRATION.json')
    if not(r.get('production_admitted') is True and r['protocol_sha256']==sha(ROOT/'PROTOCOL.json') and r['frozen_sha256']==sha(ROOT/'FROZEN_HASHES.json')):raise PermissionError('No exact production release')
    if not(p['branch_head_verified'] and p['force'] is False and p['frozen_sha256']==r['frozen_sha256'] and p['commit']==r['source_commit']):raise PermissionError('No exact remote publication')
    from stream_archive import verify
    b=ROOT/'initial_backup';v=verify(read(b/'MANIFEST.json'),read(b/'READBACK_INPUT.json'))
    if v!=read(b/'READBACK_VERIFIED.json') or v['archive_sha256']!=r['initial_archive_sha256']:raise PermissionError('Private initial source/model recovery missing')
    if sha(ROOT/'initial.pt')!=r['initial_model_sha256']:raise PermissionError('Private initial checkpoint changed')
    gate=read(PREP/'RESOURCE_GATE.json');storage=read(PREP/'RESOURCE_STORAGE_RECHECK.json');binding=read(ROOT/'RESOURCE_ADMISSION_V2.json')
    if not gate['resource_pass'] or gate['disk_pass'] is not False:raise PermissionError('Original resource attempt changed')
    if not(storage['resource_and_storage_now_passed'] and storage['original_failed_gate_preserved']):raise PermissionError('Storage repair proof incomplete')
    if binding['original_resource_gate_sha256']!=sha(PREP/'RESOURCE_GATE.json') or binding['storage_recheck_sha256']!=sha(PREP/'RESOURCE_STORAGE_RECHECK.json'):raise PermissionError('Resource upgrade binding changed')
    if sha(ROOT/'PROTOCOL.json')!=sha(PREP/'PROTOCOL.json'):raise PermissionError('Scientific protocol changed after probe')
    for n in ['core.py','role_data.py','lazy_outer_data.py','lazy_bn.py','journal.py']:
        if sha(ROOT/n)!=sha(PREP/n):raise PermissionError('Arithmetic/input implementation changed after measurement')
    if not read(PREP/'NATIVE_OUTPUT_AND_OPTIMIZER_AUDIT.json')['passed']:raise PermissionError('Native reference audit missing')
    for name in ['GENERATED_NATIVE64_CHECK.json','GENERATED_PHASE_CHECK.json']:
        t=read(ROOT/name)
        if not t['passed']:raise PermissionError('Generated integration check failed')
        for n,h in t['sources'].items():
            if sha(ROOT/n)!=h:raise PermissionError('Tested production source changed: '+n)
    if shutil.disk_usage(ROOT).free<read(ROOT/'PROTOCOL.json')['disk_reserve_bytes']:raise PermissionError('Disk reserve breached')
