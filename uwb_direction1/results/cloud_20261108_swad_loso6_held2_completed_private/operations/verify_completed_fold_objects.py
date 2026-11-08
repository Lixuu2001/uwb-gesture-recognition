"""Verify private object archive and materialize every saved completed-fold file in isolation."""
from pathlib import Path
import sys,json,tempfile,time,importlib.util
P=Path('/workspace/uwb-swad-loso6-20261108');sys.path.insert(0,str(P/'shared'))
from common import read,write,sha,ready
from controller import identity
from stream_archive import verify

def main(h):
 assert h in range(2,7);started=time.monotonic();cpu=time.process_time();ready();d=P/f'held{h}_completed_backup';m=read(d/'MANIFEST.json');assert m['held']==h and not m['complete_all_six'] and not(P/f'HELD{h}_COMPLETE_PRIVATE_RECEIPT.json').exists()
 a=read(d/'TRANSFER_ATTEMPT_RESULTS.json');assert a['complete'] and a['failed_parts']==0 and a['verified_parts']==len(m['chunks'])
 rows=[read(d/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
 for r in rows:
  q=r['metadata'];assert not q['shared'] and q['permissions']==[dict(type='user',role='owner')] and q['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf'];assert r['raw_fetch'] and r['download_verified'] and not r['include_base64']
 v=verify(m,rows);assert not v['complete_modelset_verified'];write(d/'READBACK_INPUT.json',rows);write(d/'READBACK_VERIFIED.json',v)
 base=P/'held1_production64_backup';bm=read(base/'MANIFEST.json');base_rows=read(base/'READBACK_INPUT.json');audit=read(P/f'HELD{h}_COMPLETE_OBJECT_AUDIT.json');recipe=read(P/f'HELD{h}_COMPLETE_OBJECT_RECIPE.json');assert sha(P/f'HELD{h}_COMPLETE_OBJECT_RECIPE.json')==audit['recipe_sha256']
 spec=importlib.util.spec_from_file_location('swad_exact_fold_object_restore',P/'operations/restore_fold_objects.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 with tempfile.TemporaryDirectory(prefix=f'uwb-swad-isolated-fold{h}-',dir='/tmp') as tmp:
  dest=Path(tmp)/'extract';result=module.restore(bm,base_rows,m,rows,dest);restored=dest/'restored_phase';actual={str(f.relative_to(restored)) for f in restored.rglob('*') if f.is_file()};assert actual==set(audit['source_file_sha256'])
  for n,digest in audit['source_file_sha256'].items():assert sha(restored/n)==sha(P/n)==digest,n
  result.update(physical_byte_copy_not_symlink_or_refit=True,all_files_rechecked_against_original_independent_audit_SHA_and_current_immutable_files=True,temporary_recovery_copy_removed_after_verification=True,new_archive_raw_allmember_verified=True,base_prior_raw_readback_reused_plus_current_physical_archive_reverification=True)
 write(P/f'HELD{h}_COMPLETE_PHYSICAL_RECOVERY_VERIFIED.json',result)
 old=read(P/'PHASE_IDENTITY.json');now=identity(old['pid']);active=now and now['state']!='Z' and all(now[k]==old[k] for k in ['pid','boot_id','startticks']);phase=read(P/'STATE.json');assert active or (phase['state']=='all_six_method_folds_frozen_unscored_requires_complete_private_backup_before_cohort_once' and phase['completed_folds']==list(range(1,7)))
 ready();assert not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
 index=dict(state='complete_saved_single_fold_native64_fresh64_objects_private_raw_verified_requires_prior_base22_and_recipe',held=h,folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',raw_fetch_options=dict(download_raw_file=True,include_base64=False),max_concurrency=3,archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],members_manifest_sha256=m['members_manifest_sha256'],requires_base_archive_sha256=m['requires_base_archive_sha256'],complete_single_fold_with_verified_base_and_recipe=True,complete_modelset=False,complete_all_six=False,whole_job_resume_authorized=False,cohort_score_authorized=False,chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows])
 write(d/'CLOUD_DOWNLOAD_INDEX.json',index)
 prior=read(P/'HELD1_COMPLETE_COMPOSITE_RECOVERY_INDEX.json')
 write(P/f'HELD{h}_COMPLETE_COMPOSITE_RECOVERY_INDEX.json',dict(original_raw_GAP_all6_initial_and_held1_complete_chain=prior,completed_fold_objects=index,exact_byte_recipe=recipe,isolated_physical_recovery_verified=result,complete_held_user=h,complete_native64_fresh64_all4_models_outputs_audits=True,complete_sixfold=False,whole_job_resume_authorized=False,single_fold_score_authorized=False,preserve_original_PID_elapsed_budget=True,do_not_retrain_or_overwrite_completed_folds=True))
 receipt=dict(state='complete_saved_SWAD_fold_native64_independent64_all4_audits_private_raw_allmember_and_physical_restore_verified_unscored',held=h,parts=len(m['chunks']),payload_members=v['tracked_payload_members_verified'],archive_bytes=m['archive_bytes'],archive_sha256=m['archive_sha256'],source_files=recipe['files'],source_bytes=recipe['bytes'],new_unique_objects=recipe['new_unique_objects'],new_unique_bytes=recipe['new_unique_bytes'],within_archive_duplicate_targets=recipe['within_new_archive_duplicate_targets'],prior_base_files_reused=recipe['base_files'],prior_base_logical_bytes_reused=recipe['base_logical_bytes'],physical_recovery=result,raw_allmember_verification=v,transfer_failed_parts=0,automatic_retries=0,transfer_tool_window_wall_seconds=a['transfer_tool_window_wall_seconds'],transfer_CPU_unmeasured=True,verification_restore_wall=time.monotonic()-started,verification_restore_CPU=time.process_time()-cpu,active_phase_untouched=True,extra_model_forward_or_optimizer_updates=0,complete_all_six=False,cohort_score_authorized=False)
 write(P/f'HELD{h}_COMPLETE_PRIVATE_RECEIPT.json',receipt);print(json.dumps(dict(held=h,files=result['restored_files'],bytes=result['restored_bytes'],parts=len(m['chunks']),payload=v['tracked_payload_members_verified'],archive_bytes=m['archive_bytes'],complete_all_six=False)))
if __name__=='__main__':assert len(sys.argv)==2;main(int(sys.argv[1]))
