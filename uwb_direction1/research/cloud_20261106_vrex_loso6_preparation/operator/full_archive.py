"""Archive all six fully audited, unscored VREx folds, including full replay models."""
from pathlib import Path
import sys,json
P=Path('/workspace/uwb-vrex-loso6-20261106');sys.path.insert(0,str(P/'shared'))
from common import ready,read,write,sha
from stream_archive import build,verify
from controller import identity
def create():
    c=ready(True);assert not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
    old=read(P/'PHASE_IDENTITY.json');now=identity(old['pid']);assert now is None or now['state']=='Z' or any(now[k]!=old[k] for k in ['startticks','boot_id'])
    assert read(P/'STATE.json')['state']=='all_six_method_folds_frozen_unscored_requires_complete_private_backup_before_cohort_once'
    frozen=read(P/'ALL_SIX_PREDICTIONS_FROZEN.json')
    for row in frozen['files'].values():assert sha(row['path'])==row['sha256']
    files={'phase/'+n:P/n for n in [*read(P/'FROZEN_HASHES.json'),'FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json','REAL_RELEASE.json','SYNTHETIC_SUMMARY.json','INITIAL_ASSET.json','STATE.json','RESOURCE_LEDGER.json','PHASE_IDENTITY.json','ALL_SIX_PREDICTIONS_FROZEN.json']}
    for h in range(1,7):
        fold=P/f'folds/held{h}'
        assert read(fold/'PHASE_TERMINATION.json')['exit_code']==0
        for stage in ['train','replay','output']:
            term=read(fold/f'run/execution/{stage}/TERMINATION.json');assert term['exit_code']==0 and term['resource_stop_reason'] is None
        for f in fold.rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and not f.name.endswith(('.tmp','.lock')):files['phase/'+str(f.relative_to(P))]=f
    for sub in ['initial_backup','operator']:
        for f in (P/sub).rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and not f.name.endswith(('.tmp','.lock')):files['phase/'+str(f.relative_to(P))]=f
    for n in read(P/'shared/DEPENDENCIES.json')['files']:files['native_source/'+str(Path(n).relative_to(c['native_source']))]=Path(n)
    files['private_IDs/source_ids.npz']=Path(c['ids'])
    layout=dict(archive_to_expected_path={k:str(v) for k,v in files.items()},raw_arrays_and_runtime_external=True,all_six_trained_method_models_included=True)
    write(P/'FULL_SOURCE_LAYOUT.json',layout);files['SOURCE_LAYOUT.json']=P/'FULL_SOURCE_LAYOUT.json'
    m=build(P/'full_backup','vrex_loso6_all6_full_20261106',files,'VREx_LOSO_all6_complete_models_audits_outputs_before_single_cohort_score',chunk_size=7340032,limit=1073741824)
    m.update(complete_modelset=True,modelset_scope='all six held1..6 method folds; baseline has its original separate recovery chain',complete_all_six=True);write(P/'full_backup/MANIFEST.json',m);return m
def check():
    d=P/'full_backup';m=read(d/'MANIFEST.json');rows=[read(d/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
    for r in rows:
        q=r['metadata'];assert not q['shared'] and q['permissions']==[dict(type='user',role='owner')] and q['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf']
        assert r['raw_fetch'] and r['download_verified'] and not r['include_base64']
    v=verify(m,rows);write(d/'READBACK_INPUT.json',rows);write(d/'READBACK_VERIFIED.json',v)
    receipt=dict(state='complete_all_six_VREx_raw_allmember_verified',archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],members_manifest_sha256=m['members_manifest_sha256'],scope=m['scope'],complete_all_six=True,chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows])
    write(d/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',raw_fetch_options=dict(download_raw_file=True,include_base64=False),max_concurrency=3,**receipt));return v
if __name__=='__main__':print(json.dumps(create() if sys.argv[1]=='build' else check()))
