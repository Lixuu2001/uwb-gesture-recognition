"""Complete selected predictors and experiment evidence; no raw radar arrays."""
from pathlib import Path
from common import B,read,write,sha,complete
from stream_archive import build,verify
FOLDER='12AGMwT99yXCTYBQocQaX901Br-IelmOf'
def create():
    p,v,a=complete();files={}
    for n in [*read(B/'FROZEN_HASHES.json'),'FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json']:
        files['study/'+n]=B/n
    for directory in ['validation01','audit01','validate_supervisor01','audit_supervisor01']:
        for f in (B/directory).iterdir():
            if f.is_file():files[directory+'/'+f.name]=f
    tm=Path(p['timemil_root']);sh=Path(p['shuffle_root'])
    for n in ['best.pt','last.pt','initial.pt','epochs.jsonl','ARM_COMPLETE.json']:
        files['selected_models/TimeMIL/'+n]=tm/'run02/timemil'/n
    for n in ['final_model.pt','final_full.pt','initial12_head.pt','TRAIN_ENDPOINT.json','split.npz']:
        files['selected_models/ShuffleNet/'+n]=sh/'run01'/n
    files['models_lineage/TimeMIL_COMPLETE_RESULT_AUDIT.json']=tm/'COMPLETE_RESULT_AUDIT.json'
    files['models_lineage/TimeMIL_PRIVATE_RECEIPT.json']=Path('/workspace/uwb-cloud-current/uwb_direction1/results/cloud_20261030_timemil_final/FINAL_DRIVE_BACKUP_RECEIPT.json')
    for n in ['COMPLETE_AUDIT.json','FROZEN_HASHES.json','TRIAL_PROTOCOL.json']:
        files['models_lineage/ShuffleNet_'+n]=sh/n
    files['models_lineage/ShuffleNet_PRIVATE_INDEX.json']=sh/'complete_backup/CLOUD_DOWNLOAD_INDEX.json'
    for f in tm.iterdir():
        if f.is_file() and f.suffix in ['.py','.json','.md','.txt']:files['TimeMIL_source/'+f.name]=f
    for f in (tm/'vendor').rglob('*'):
        if f.is_file() and f.suffix!='.pyc' and '__pycache__' not in f.parts:files['TimeMIL_source/'+str(f.relative_to(tm))]=f
    roots=[Path('/workspace/uwb-shufflenetv2-x1-source-20261103'),Path('/workspace/uwb-resnet18-finetune-source-20261102'),Path('/workspace/uwb-resnet18-imagenet-source-20261102')]
    for root in roots:
        for f in root.rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and ('weights' not in f.parts) and f.suffix in ['.py','.json','.md','']:
                files['ShuffleNet_source/'+root.name+'/'+str(f.relative_to(root))]=f
    files['initial_encoder/shufflenetv2_x1-5666bf0f80.pth']=roots[0]/'weights/shufflenetv2_x1-5666bf0f80.pth'
    files['dependencies/source_ids.npz']=Path(p['ids'])
    write(B/'SOURCE_LAYOUT.json',dict(scope='both_selected_predictors_full_states_and_existing_training_lineage',new_training=False,archive_to_path={k:str(v) for k,v in files.items()}))
    files['SOURCE_LAYOUT.json']=B/'SOURCE_LAYOUT.json'
    return build(B/'private_backup','softvote_TimeMIL64_ShuffleNetAdam100_source1_20261103',files,p['private_scope'],chunk_size=p['chunk_bytes'],limit=p['archive_upper_bytes'])
def privacy(rows):
    assert rows and len({r['file_id'] for r in rows})==len(rows)
    for r in rows:
        assert r['raw_fetch'] is True and r['include_base64'] is False and r['download_verified'] is True
        m=r['metadata'];assert m['shared'] is False and m['parent_ids']==[FOLDER]
        assert m['permissions']==[dict(role='owner',type='user')] and m['size_bytes']==r['bytes']
def restore_verify():
    p,v,a=complete();b=B/'private_backup';m=read(b/'MANIFEST.json');rows=read(b/'READBACK_INPUT.json')
    assert m['scope']==p['private_scope'];privacy(rows);r=verify(m,rows);write(b/'READBACK_VERIFIED.json',r)
    result=dict(state='private_raw_all_members_verified',scope=p['private_scope'],archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],members_manifest_sha256=m['members_manifest_sha256'],complete_selected_modelset_verified=True,diagnostic_labels_scored=False,validation_gate_passed=v['validation_gate_passed'],all_inputs_and_models_preserved=True,chunks=[{k:x for k,x in r.items() if k!='path'} for r in rows])
    write(b/'DRIVE_BACKUP_RECEIPT.json',result)
    write(b/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id=FOLDER,max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),**result))
    return r
if __name__=='__main__':
    import sys,json
    assert len(sys.argv)==2 and sys.argv[1] in ['build','verify']
    print(json.dumps(create() if sys.argv[1]=='build' else restore_verify()))
