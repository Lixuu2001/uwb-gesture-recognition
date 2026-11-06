"""Complete unscored pilot recovery; never read held truth or fit a model."""
from pathlib import Path
import sys,json
sys.path.insert(0,'/workspace/uwb-mmd-gap64-20261106')
from common import B,ready,read,write,sha
from stream_archive import build,verify
def create():
    ready(True);root=B/'run';assert not (B/'SUMMARY.json').exists() and not (B/'FINALIZE_ONCE.lock').exists()
    from controller import identity
    old=read(B/'CONTROLLER_IDENTITY.json');q=identity(old['pid'])
    assert q is None or q['state']=='Z' or q['boot_id']!=old['boot_id'] or q['startticks']!=old['startticks']
    assert read(B/'STATE.json')['state']=='pilot_audited_frozen_unscored_requires_complete_private_backup_before_once_gate'
    for a in ['train','replay','output']:
        t=read(root/'execution'/a/'TERMINATION.json');assert t['exit_code']==0 and t['resource_stop_reason'] is None
    f=read(root/'PREDICTIONS_FROZEN.json');assert f['full_training_replayed'] and not f['diagnostic_labels_read']
    for n,h in f['files'].items():assert sha(root/n)==h
    files={'study/'+n:B/n for n in ['ACTIVATION.json','SOURCE_REVIEW.json','ARCHIVE_DEPENDENCIES.json',*read(B/'FROZEN_HASHES.json'),'FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json','REAL_RELEASE.json','SYNTHETIC_RESULT.json','RESOURCE_RESULT.json','initial.pt','roles.npz','INITIAL_ASSET.json','STATE.json','RESOURCE_LEDGER.json','CONTROLLER_IDENTITY.json']}
    for d in ['run','operator','initial_backup']:
        for p in (B/d).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith(('.lock','.tmp')):
                files[d+'/'+str(p.relative_to(B/d))]=p
    c=ready()
    for n in read(B/'DEPENDENCIES.json')['files']:files['native_source/'+str(Path(n).relative_to(c['native_source']))]=Path(n)
    for ref in c['author_reference_files']:files['author_review/'+str(Path(ref).relative_to(B/'author_reference'))]=Path(ref)
    files['lineage/noMixup_reference_SUMMARY.json']=Path(c['source1_reference_summary'])
    prior=Path(c['prior_CORAL_summary']).parent
    for src,dst in [('SUMMARY.json','prior_CORAL_SUMMARY.json'),('STATE.json','prior_CORAL_POST_SCORE_STATE.json'),('FINALIZE_ONCE.lock','prior_CORAL_FINALIZE_ONCE.lock')]:files['lineage/'+dst]=prior/src
    layout=B/'FULL_SOURCE_LAYOUT.json';write(layout,dict(archive_to_expected_path={k:str(v) for k,v in files.items()},raw_data_and_maps_external=True,runtime_binaries_external=True,matched_noMixup_baseline_archives_external=True))
    files['SOURCE_LAYOUT.json']=layout
    return build(B/'full_backup','mmd_gap64_full_20261106',files,'fixed_MMD_GAP64_full_pilot_models_audits_outputs_before_once_gate',chunk_size=7340032,limit=536870912)
def verify_backup():
    d=B/'full_backup';m=read(d/'MANIFEST.json');rows=[read(d/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
    for r in rows:
        z=r['metadata'];assert not z['shared'] and z['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf'] and z['permissions']==[dict(role='owner',type='user')]
        assert r['raw_fetch'] and r['download_verified'] and not r['include_base64']
    v=verify(m,rows);write(d/'READBACK_INPUT.json',rows);write(d/'READBACK_VERIFIED.json',v)
    receipt=dict(state='complete_pilot_modelset_owner_only_raw_allmember_verified',scope=m['scope'],archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],members_manifest_sha256=m['members_manifest_sha256'],diagnostic_labels_read=False,complete_modelset_verified=True,chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows])
    write(d/'DRIVE_BACKUP_RECEIPT.json',receipt);write(d/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),**receipt));return v
if __name__=='__main__':print(json.dumps(create() if sys.argv[1]=='build' else verify_backup()))
