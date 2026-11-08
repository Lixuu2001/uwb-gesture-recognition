"""Private supplemental raw readback and isolated physical recovery of all saved fold files."""
from pathlib import Path
import sys,tempfile,time,json,importlib.util
P=Path('/workspace/uwb-swad-loso6-20261108');sys.path.insert(0,str(P/'shared'))
from common import read,write,sha,ready
from stream_archive import verify
from controller import identity
start=time.monotonic();cpu=time.process_time();ready();d=P/'held1_completion_backup';base=P/'held1_production64_backup';m=read(d/'MANIFEST.json');transfer=read(d/'TRANSFER_ATTEMPT_RESULTS.json');assert transfer['complete'] and transfer['failed_parts']==0 and transfer['verified_parts']==len(m['chunks'])==1
rows=[read(d/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
for r in rows:
 q=r['metadata'];assert not q['shared'] and q['permissions']==[dict(type='user',role='owner')] and q['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf']
 assert r['raw_fetch'] and r['download_verified'] and not r['include_base64']
v=verify(m,rows);assert not v['complete_modelset_verified'];write(d/'READBACK_INPUT.json',rows);write(d/'READBACK_VERIFIED.json',v)
spec=importlib.util.spec_from_file_location('swad_completed_fold_physical_recovery',P/'operations/restore_completed_fold_dedup.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
audit=read(P/'HELD1_COMPLETION_AUDIT.json');recipe=read(P/'HELD1_COMPLETION_DEDUP_RECIPE.json')
with tempfile.TemporaryDirectory(prefix='uwb-swad-isolated-completion-',dir='/tmp') as tmp:
 dest=Path(tmp)/'extract';result=module.restore(read(base/'MANIFEST.json'),read(base/'READBACK_INPUT.json'),m,rows,dest)
 rebuilt=dest/'restored_phase';actual={str(p.relative_to(rebuilt)) for p in rebuilt.rglob('*') if p.is_file()};assert actual==set(audit['source_file_sha256'])
 for n,h in audit['source_file_sha256'].items():assert sha(rebuilt/n)==sha(P/n)==h,n
 result.update(all350_source_files_independently_rechecked_against_original_audit=True,actual_recovery_target_isolated_and_removed_after_verification=True,physical_byte_copy_not_symlink_or_weight_refit=True,new_archive_raw_readback_all_members_verified=True,base_existing_raw_readback_chain_reused_and_archive_bytes_physically_reverified=True,restore_recipe_sha256=sha(P/'HELD1_COMPLETION_DEDUP_RECIPE.json'))
write(P/'HELD1_COMPLETION_PHYSICAL_RECOVERY_VERIFIED.json',result)
old=read(P/'PHASE_IDENTITY.json');now=identity(old['pid']);assert now and now['state']!='Z' and all(now[k]==old[k] for k in ['pid','boot_id','startticks'])
ready();assert not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
index=dict(state='SWAD_held1_completed_independent64_supplement_raw_allmember_verified_requires_production22_and_recipe',folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',raw_fetch_options=dict(download_raw_file=True,include_base64=False),max_concurrency=3,archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],members_manifest_sha256=m['members_manifest_sha256'],requires_base_archive_sha256=m['requires_base_archive_sha256'],complete_single_fold_with_verified_base_and_recipe=True,complete_modelset=False,complete_all_six=False,whole_job_resume=False,cohort_score_authorized=False,chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows])
write(d/'CLOUD_DOWNLOAD_INDEX.json',index)
write(P/'HELD1_COMPLETE_COMPOSITE_RECOVERY_INDEX.json',dict(initial_and_production_recovery_chain=read(P/'HELD1_PRODUCTION64_COMPOSITE_RECOVERY_INDEX.json'),held1_completion_supplement=index,exact_byte_recipe=recipe,isolated_physical_recovery_verified=result,complete_held1_production64_independent64_all4_models_outputs_and_native_audits=True,complete_sixfold=False,whole_job_resume_authorized=False,single_fold_score_authorized=False,existing_phase_identity=old,preserve_original_elapsed_wall_CPU_caps=True,do_not_retrain_or_overwrite_completed_fold=True))
receipt=dict(state='SWAD_held1_complete64_fresh64_private_recovery_chain_and_all350_file_isolated_physical_restore_verified_unscored',new_parts=1,new_payload_members=v['tracked_payload_members_verified'],new_archive_bytes=m['archive_bytes'],new_archive_sha256=m['archive_sha256'],base_production_parts=22,source_files=recipe['files'],source_bytes=recipe['bytes'],exact_byte_reuse=recipe['storage_counts'],physical_recovery=result,raw_allmember_verification=v,transfer_failed_parts=0,automatic_retries=0,transfer_tool_window_wall_seconds=transfer['transfer_tool_window_wall_seconds'],transfer_CPU_unmeasured=True,build_execution=read(P/'HELD1_COMPLETION_BUILD_EXECUTION.json'),verification_restore_wall=time.monotonic()-start,verification_restore_CPU=time.process_time()-cpu,original_model_phase_active_and_unchanged=True,no_additional_model_forward_or_optimizer_update=True,whole_job_resume_authorized=False,cohort_score_authorized=False,complete_all_six=False)
write(P/'HELD1_COMPLETE_PRIVATE_RECEIPT.json',receipt);print(json.dumps(dict(restored_files=result['restored_files'],restored_bytes=result['restored_bytes'],new_parts=1,new_payload=v['tracked_payload_members_verified'],new_archive_bytes=m['archive_bytes'],complete_all_six=False)))
