"""Archive a finished audited fold; register raw recovery/cost before next fold."""
from pathlib import Path
import sys,json
from core import ROOT,read,write,sha
from stream_archive import build,verify
def archive(h):
    fold=ROOT/'folds'/f'held{h}';f=read(fold/'PREDICTIONS_FROZEN.json')
    if not f['all3_arms_prod64_fresh64_audited_unscored']:raise ValueError('Fold not frozen')
    for p,d in f['files'].items():
        if sha(p)!=d:raise ValueError('Frozen file changed')
    files={}
    for p in fold.rglob('*'):
        if p.is_file() and 'private_backup' not in p.relative_to(fold).parts:
            if p.suffix=='.tmp':raise ValueError('Interrupted write needs recovery')
            files['fold/'+str(p.relative_to(fold))]=p
    m=build(fold/'private_backup',f'tail_outer8_held{h}_complete_20261109',files,'complete frozen native/tail/nativeBN production64/fresh64/fullstates/logs/IDs references for held'+str(h),limit=192*1024**2)
    rows=[dict(**c,path=str(fold/'private_backup'/c['name'])) for c in m['chunks']];write(fold/'private_backup/LOCAL_VERIFIED.json',verify(m,rows));print(json.dumps(dict(held=h,parts=len(m['chunks']),bytes=m['archive_bytes'],payload=m['tracked_payload_files'])))
def verify_raw(h):
    fold=ROOT/'folds'/f'held{h}';b=fold/'private_backup';m=read(b/'MANIFEST.json');rows=[read(b/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
    for r in rows:
        assert r['metadata']['shared'] is False and len(r['metadata']['permissions'])==1 and r['metadata']['permissions'][0]['role']=='owner' and r['raw_fetch'] and r['download_verified'] and not r['include_base64']
    write(b/'READBACK_INPUT.json',rows);write(b/'READBACK_VERIFIED.json',verify(m,rows))
    print('Raw allmember fold verification complete')
def finalize(h):
    fold=ROOT/'folds'/f'held{h}';b=fold/'private_backup';m=read(b/'MANIFEST.json')
    proof=read(b/'READBACK_VERIFIED.json')
    if proof['archive_sha256']!=m['archive_sha256']:raise PermissionError('Raw proof missing or changed')
    # The operator must create measured/upper-bound CPU charges before release.
    costs=read(fold/'BACKUP_CHARGE_IDS.json');known={x['id'] for x in read(ROOT/'EXTERNAL_PHASE_CHARGES.json')}
    if not costs or not set(costs)<=known:raise PermissionError('Backup CPU ledger incomplete')
    write(fold/'PRIVATE_FOLD_RELEASE.json',dict(held=h,frozen_outputs_sha256=sha(fold/'PREDICTIONS_FROZEN.json'),archive_manifest_sha256=sha(b/'MANIFEST.json'),raw_proof_sha256=sha(b/'READBACK_VERIFIED.json'),readback_input_sha256=sha(b/'READBACK_INPUT.json'),external_charge_ids=costs,source_frozen_sha256=sha(ROOT/'FROZEN_HASHES.json'),whole_job_resume_authorized=False,held_scores_computed=False))
    print('Private fold release complete; next held user may start without scoring')
if __name__=='__main__':
    if sys.argv[1]=='build':archive(int(sys.argv[2]))
    elif sys.argv[1]=='verify':verify_raw(int(sys.argv[2]))
    elif sys.argv[1]=='finalize':finalize(int(sys.argv[2]))
    else:raise ValueError('Unknown operation')
