"""Initial and final private recovery, bounded streaming, no raw radar arrays."""
from pathlib import Path
from common import B,config,ready,require_initial,read,write,sha,privacy
from stream_archive import build,verify
from checks import verify_audits,member
def sources(c,p):
    files={}
    for n in [*read(B/'FROZEN_HASHES.json'),'FROZEN_HASHES.json','REMOTE_REAL_PREREGISTRATION.json']:
        files['study/'+n]=B/n
    for arm,row in read(B/'SEED_FILES.json').items():files['initial_models/'+arm+'.pt']=B/row['path']
    for n in read(B/'ASSET_FILES.json'):files['private_IDs/'+Path(n).name]=B/n
    tm=Path(p['TimeMIL']['source'])
    for path in tm.iterdir():
        if path.is_file() and path.suffix in ['.py','.json','.md','.txt','.log']:files['TimeMIL_source/'+path.name]=path
    for path in (tm/'vendor').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':files['TimeMIL_source/'+str(path.relative_to(tm))]=path
    roots=[Path('/workspace/uwb-shufflenetv2-x1-source-20261103'),Path('/workspace/uwb-resnet18-finetune-source-20261102'),Path('/workspace/uwb-resnet18-imagenet-source-20261102')]
    for root in roots:
        for path in root.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and 'weights' not in path.parts and path.suffix in ['.py','.json','.md','']:
                files['ShuffleNet_source/'+root.name+'/'+str(path.relative_to(root))]=path
    files['initial_encoder/shufflenetv2_x1-5666bf0f80.pth']=roots[0]/'weights/shufflenetv2_x1-5666bf0f80.pth'
    pilot=Path(p['reused_held1_root'])
    for n in ['SUMMARY.json','FROZEN_HASHES.json','private_backup/CLOUD_DOWNLOAD_INDEX.json']:
        files['held1_lineage/'+n]=pilot/n
    for baseline,items in c['baselines'].items():
        for n,row in items.items():files['strong_baseline_lineage/'+baseline+'/'+n]=Path(row['path'])
    for arm,row in c['held1_member_summaries'].items():files['held1_lineage/'+arm+'_SUMMARY.json']=Path(row['path'])
    return files
def all_frozen():
    c,p=require_initial();f=read(B/'ALL_PREDICTIONS_FROZEN.json')
    assert f['state']=='all5newfolds_3arms_frozen_unscored' and f['method_lock_sha256']==sha(B/'METHOD_LOCK.json')
    assert not f['diagnostic_labels_read'] and not f['users7_8_labels_used']
    for held in [2,3,4,5,6]:
        verify_audits(held);root=B/'folds'/f'held{held}';a=read(root/'fusion/FOLD_FROZEN.json')
        assert sha(root/'fusion/FOLD_FROZEN.json')==f['folds'][str(held)]
        assert a['state']=='three_arms_frozen_before_any_new_held_truth' and not a['diagnostic_labels_read']
        for n,h in a['files'].items():assert sha(root/'fusion'/n)==h
        t=read(root/'FUSION_TERMINATION.json');assert t['exit_code']==0 and t['resource_stop_reason'] is None and not t['wall_timeout'] and t['cgroup_OOM_kill_delta']==0
    ledger=read(B/'RESOURCE_LEDGER.json')
    assert sum(x['wall'] for x in ledger['folds'].values())<=p['total5fold_wall_upper_seconds']
    assert sum(x['cpu'] for x in ledger['folds'].values())<=p['total5fold_CPU_upper_seconds']
    for x in ledger['folds'].values():assert x['wall']<=p['per_fold_wall_seconds'] and x['cpu']<=p['per_fold_CPU_seconds']
    return c,p
def create(kind):
    assert kind in ['initial','full']
    c,p=ready() if kind=='initial' else all_frozen()
    if kind=='initial':assert not (B/'CONTROLLER_IDENTITY.json').exists()
    files=sources(c,p)
    if kind=='full':
        for name in ['STATE.json','RESOURCE_LEDGER.json','ALL_PREDICTIONS_FROZEN.json','INITIAL_PRIVATE_INSURANCE.json']:
            files['execution/'+name]=B/name
        for folder in ['folds','attempts']:
            for path in (B/folder).rglob('*'):
                if path.is_file() and not path.name.endswith('.lock') and not path.name.endswith('.tmp') and '__pycache__' not in path.parts:files[folder+'/'+str(path.relative_to(B/folder))]=path
        tm=Path(p['TimeMIL']['source'])/'run02/timemil/best.pt'
        sh=Path('/workspace/uwb-shufflenetv2-x1-adam100-recovery45-20261103/run01/final_model.pt')
        files['reused_held1_models/TimeMIL_best.pt']=tm;files['reused_held1_models/ShuffleNet_final.pt']=sh
    layout=B/(kind+'_SOURCE_LAYOUT.json');write(layout,dict(archive_to_expected_path={k:str(v) for k,v in files.items()},raw_radar_arrays_included=False,runtime_binaries_included=False,new_training_states_included=kind=='full'))
    files['SOURCE_LAYOUT.json']=layout
    return build(B/(kind+'_backup'),'softvote_source6_'+kind+'_20261103',files,c[kind+'_archive_scope'],chunk_size=c['chunk_bytes'],limit=c[kind+'_archive_upper_bytes'])
def restore_verify(kind):
    c,p=ready() if kind=='initial' else all_frozen();d=B/(kind+'_backup');m=read(d/'MANIFEST.json');rows=read(d/'READBACK_INPUT.json')
    assert m['scope']==c[kind+'_archive_scope'];privacy(rows);v=verify(m,rows);write(d/'READBACK_VERIFIED.json',v)
    r=dict(state='private_raw_all_members_verified',scope=m['scope'],archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],members_manifest_sha256=m['members_manifest_sha256'],complete_modelset_verified=True,diagnostic_labels_scored=False,chunks=[{k:v for k,v in row.items() if k!='path'} for row in rows])
    write(d/'DRIVE_BACKUP_RECEIPT.json',r);write(d/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),**r))
    if kind=='initial':
        members={x['path']:x for x in read(d/'MEMBERS.json')['members']}
        for arm,row in read(B/'SEED_FILES.json').items():assert members['initial_models/'+arm+'.pt']['sha256']==row['sha256']
        write(B/'INITIAL_PRIVATE_INSURANCE.json',dict(state='initial_source6_models_roles_sources_private_raw_allmember_verified',frozen_sha256=sha(B/'FROZEN_HASHES.json'),archive_sha256=m['archive_sha256'],before_new_UWB_fit=True))
    return v
if __name__=='__main__':
    import sys,json
    assert len(sys.argv)==3 and sys.argv[1] in ['build','verify'] and sys.argv[2] in ['initial','full']
    print(json.dumps(create(sys.argv[2]) if sys.argv[1]=='build' else restore_verify(sys.argv[2])))
