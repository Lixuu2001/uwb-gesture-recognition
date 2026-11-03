"""Prospective complete audited Adam100 source/model recovery; no activation files."""
from pathlib import Path
from trial_common import B, complete, read, write, sha
from stream_archive import build, verify

SCOPE = 'complete_audited_full_ShuffleNetV2_x1_Adam100_B20_checkpoint45_recovery_source_held1'
FOLDER = '12AGMwT99yXCTYBQocQaX901Br-IelmOf'

def scientific_files(c):
    files={'study/FROZEN_HASHES.json':B/'FROZEN_HASHES.json'}
    for name in read(B/'FROZEN_HASHES.json'):
        files['study/'+name]=B/name
    native=(B/c['source_module_directory']).resolve()
    for p in sorted(native.iterdir()):
        if p.is_file() and p.suffix in ('.py','.json','.log','.md'):
            files['native/'+p.name]=p
    official=Path('/workspace/uwb-shufflenetv2-x1-source-20261103')
    for name in ['native_shufflenet.py','local_author_sources/shufflenetv2.py',
                 'local_author_sources/LICENSE','weights/shufflenetv2_x1-5666bf0f80.pth']:
        files['official/'+name]=official/name
    files['support/native_finetune.py']=Path('/workspace/uwb-resnet18-finetune-source-20261102/native_finetune.py')
    base=Path('/workspace/uwb-resnet18-imagenet-source-20261102')
    for name in ['native_resnet18.py','local_author_sources/resnet.py','local_author_sources/LICENSE']:
        files['support/base/'+name]=base/name
    files['dependencies/source_ids.npz']=Path(c['source_manifest'])
    for key in ['reference_predictions','reference_summary']:
        files['reference/'+Path(c[key]).name]=Path(c[key])
    cost=Path(c['cost_workspace'])
    for name in [*read(cost/'COST_HASHES.json'),'COST_HASHES.json',
            'REMOTE_COST_PREREGISTRATION.json','VERIFIED_COST_RESULT.json',
            'profile01/COST_RESULT.json','supervisor01/TERMINATION.json']:
        files['cost/'+name]=cost/name
    old=Path(c['operational_recovery']['original_root'])
    for name in ['FROZEN_HASHES.json','REMOTE_REAL_PREREGISTRATION.json','INITIAL_PRIVATE_INSURANCE.json']:
        files['original/'+name]=old/name
    for name in ['final_full.pt','steps.jsonl','STATE.json','epoch045.json']:
        files['original/run01/'+name]=old/'run01'/name
    return files

def create():
    c=complete();files=scientific_files(c)
    for name in ['REMOTE_REAL_PREREGISTRATION.json','COMPLETE_AUDIT.json',
                 'STATE.json','INITIAL_PRIVATE_INSURANCE.json']:
        files['study/'+name]=B/name
    for directory in ['run01','supervisor01','audit_supervisor01','audit01']:
        for p in sorted((B/directory).iterdir()):
            if p.is_file(): files[directory+'/'+p.name]=p
    write(B/'SOURCE_LAYOUT.json',dict(
        state='scientific_source_model_layout_runtime_raw_data_restore_separately',
        archive_to_expected_path={k:str(v) for k,v in sorted(files.items())},
        whole_job_or_OS_resume=False))
    files['SOURCE_LAYOUT.json']=B/'SOURCE_LAYOUT.json'
    metadata=sum(p.stat().st_size for name,p in files.items()
                 if not name.endswith('.pt') and not name.startswith('official/weights/'))
    assert metadata<=c['backup_metadata_upper_bytes']
    return build(B/'complete_backup','shufflenetv2_x1_adam100_recovery45_complete_20261103',
                 files,SCOPE,chunk_size=c['streaming_chunk_bytes'],limit=c['archive_upper_bytes'])

def privacy(rows):
    assert rows and len({r['file_id'] for r in rows})==len(rows)
    for r in rows:
        assert r['raw_fetch'] is True and r['include_base64'] is False
        assert r['download_verified'] is True
        m=r['metadata']
        assert m['shared'] is False and m['parent_ids']==[FOLDER]
        assert m['permissions']==[dict(role='owner',type='user')]
        assert m['size_bytes']==r['bytes']

def restore_verify():
    complete();p=B/'complete_backup';m=read(p/'MANIFEST.json')
    assert m['scope']==SCOPE and m['complete_modelset'] is True
    rows=read(p/'READBACK_INPUT.json');privacy(rows)
    v=verify(m,rows);write(p/'READBACK_VERIFIED.json',v)
    receipt=dict(state='private_readback_verified',scope=SCOPE,
        archive_bytes=m['archive_bytes'],archive_sha256=m['archive_sha256'],
        members_manifest_sha256=m['members_manifest_sha256'],
        chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows],
        complete_modelset_verified=True,numeric_audit_certified=True,
        initial_encoder_head_final_full_Adam_RNG_included=True,
        diagnostic_labels_scored=False,whole_job_resume=False,
        raw_arrays_included=False,runtime_binaries_included=False)
    write(p/'DRIVE_BACKUP_RECEIPT.json',receipt)
    write(p/'CLOUD_DOWNLOAD_INDEX.json',dict(schema_version=1,
        receipt_sha256=sha(p/'DRIVE_BACKUP_RECEIPT.json'),folder_id=FOLDER,
        max_concurrency=3,folder_must_remain_owner_only=True,
        raw_fetch_options=dict(download_raw_file=True,include_base64=False),**receipt))
    return v

if __name__=='__main__':
    import sys,json
    assert len(sys.argv)==2 and sys.argv[1] in ['build','verify']
    print(json.dumps(create() if sys.argv[1]=='build' else restore_verify()))
