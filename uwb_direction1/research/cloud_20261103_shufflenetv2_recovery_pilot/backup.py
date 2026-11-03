from pathlib import Path
from common import B,complete,read,write,sha
from stream_archive import build,verify
SCOPE='complete_audited_full_ShuffleNetV2_x1_SGDM40_B15_recovery_20261103_source_held1'
def create():
 c=complete();files={}
 for n in ['PROTOCOL.json','FROZEN_HASHES.json','DEPENDENCIES.json','REMOTE_PREREGISTRATION.json','COMPLETE_AUDIT.json','STATE.json','OPERATIONAL_RECOVERY_PLAN.json','INITIAL_PRIVATE_INSURANCE.json',*read(B/'FROZEN_HASHES.json')]:files['study/'+n]=B/n
 for root in ['run01','supervisor01','audit_supervisor01']:
  for p in sorted((B/root).iterdir()):
   if p.is_file():files[root+'/'+p.name]=p
 s=Path(c['source_preparation'])
 for n in ['native_shufflenet.py','check_full_native.py','PROSPECTIVE_NEXT_CONFIG.json','SELECTION_RECORD.json','NATIVE_CHECK.json','NATIVE_CHECK.log','PAPER_REVIEW.json','WEIGHTS_SOURCE.json']:
  files['source/'+n]=s/n
 for n in ['local_author_sources/shufflenetv2.py','local_author_sources/LICENSE','weights/shufflenetv2_x1-5666bf0f80.pth']:files['official/'+n]=s/n
 # Shared adapter/SGD helpers are used unchanged, not a second backbone forward.
 files['support/native_finetune.py']=Path('/workspace/uwb-resnet18-finetune-source-20261102/native_finetune.py')
 base=Path('/workspace/uwb-resnet18-imagenet-source-20261102')
 for n in ['native_resnet18.py','local_author_sources/resnet.py','local_author_sources/LICENSE']:files['support/base/'+n]=base/n
 files['dependencies/source_ids.npz']=Path(c['source_manifest'])
 cost=Path(c['cost_workspace'])
 for n in ['POLICY.json','COST_HASHES.json','VERIFIED_COST_RESULT.json','profile01/COST_RESULT.json','supervisor01/TERMINATION.json','guard.py','cost_math.py','profile_cost.py','run_screen.py','test_cost_boundaries.py','DEPENDENCIES.json','COST_BOUNDARY_TESTS.log','NATIVE_CHECK.log']:files['cost/'+n]=cost/n
 for key in ['reference_predictions','reference_summary']:files['reference/'+Path(c[key]).name]=Path(c[key])
 layout=B/'SOURCE_LAYOUT.json';write(layout,dict(state='scientific_source_and_model_layout_runtime_raw_data_restore_separately',archive_to_expected_path={k:str(v) for k,v in sorted(files.items())},whole_job_or_OS_resume=False,strict_dependency_guard_also_requires_original_branch_history_and_hashed_runtime=True));files['SOURCE_LAYOUT.json']=layout
 metadata=sum(Path(p).stat().st_size for n,p in files.items() if not n.endswith('.pt') and not n.startswith('official/weights/'));assert metadata<=c['backup_metadata_upper_bytes']
 return build(B/'complete_backup','shufflenetv2_x1_sgdm40_b15_recovery_complete_20261103',files,SCOPE,chunk_size=c['streaming_chunk_bytes'],limit=c['archive_upper_bytes'])
def restore_verify():
 complete();p=B/'complete_backup';m=read(p/'MANIFEST.json');rows=read(p/'READBACK_INPUT.json');assert m['scope']==SCOPE and len(set(r['file_id'] for r in rows))==len(rows)
 for r in rows:
  assert r['raw_fetch'] and not r['include_base64'] and r['download_verified'];meta=r['metadata'];assert meta['shared']==False and meta['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf'] and meta['permissions']==[dict(role='owner',type='user')] and meta['size_bytes']==r['bytes']
 v=verify(m,rows);write(p/'READBACK_VERIFIED.json',v);receipt=dict(state='private_readback_verified',scope=SCOPE,archive_bytes=m['archive_bytes'],archive_sha256=m['archive_sha256'],members_manifest_sha256=m['members_manifest_sha256'],chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows],complete_modelset_verified=True,numeric_audit_certified=True,shared_initial_encoder_and_initial12_head_best_model_final_model_optimizer_rng_included=True,diagnostic_labels_scored=False,whole_job_resume=False,raw_arrays_included=False,runtime_binaries_included=False);write(p/'DRIVE_BACKUP_RECEIPT.json',receipt)
 write(p/'CLOUD_DOWNLOAD_INDEX.json',dict(schema_version=1,purpose='Complete full official ShuffleNetV2 x1 SGDM40 B15 audited model/ID/log/scientific-helper recovery',receipt_sha256=sha(p/'DRIVE_BACKUP_RECEIPT.json'),folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),folder_must_remain_owner_only=True,**receipt));return v
if __name__=='__main__':
 import sys,json
 assert len(sys.argv)==2 and sys.argv[1] in ['build','verify'];print(json.dumps(create() if sys.argv[1]=='build' else restore_verify()))
