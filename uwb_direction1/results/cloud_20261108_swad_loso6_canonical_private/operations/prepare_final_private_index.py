"""Assemble final recovery metadata only after all6 canonical raw/private verification. Never score."""
from pathlib import Path
import sys,json,time
P=Path('/workspace/uwb-swad-loso6-20261108');sys.path.insert(0,str(P/'shared'))
from common import ready,read,write,sha
from controller import identity
from stream_archive import verify

def preflight():
 ready();assert not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
 state=read(P/'STATE.json')
 if state['state']!='all_six_method_folds_frozen_unscored_requires_complete_private_backup_before_cohort_once' or state['completed_folds']!=list(range(1,7)):
  raise RuntimeError('incomplete_all6_no_final_index_or_release_or_score')
 old=read(P/'PHASE_IDENTITY.json');now=identity(old['pid']);assert now is None or now['state']=='Z' or any(now[k]!=old[k] for k in ['boot_id','startticks'])
 frozen=read(P/'ALL_SIX_PREDICTIONS_FROZEN.json');assert frozen['all4_arms_frozen'] and frozen['all_held_users']==list(range(1,7)) and not frozen['diagnostic_labels_read']
 for n,r in frozen['files'].items():
  path=Path(r['path']);assert path.resolve().is_relative_to(P.resolve()) and sha(path)==r['sha256'],n
 for h in range(1,7):
  fold=P/f'folds/held{h}';local=read(fold/'run/PREDICTIONS_FROZEN.json');cfg=read(fold/'PROTOCOL.json')
  assert read(fold/'PHASE_TERMINATION.json')['exit_code']==0 and local['full_training_replayed'] and local['native_reference_matches_original_GAP'] and local['epochs']==64 and local['updates']==cfg['expected_updates']
  for stage in ['train','replay','output']:
   t=read(fold/f'run/execution/{stage}/TERMINATION.json');assert t['exit_code']==0 and t['resource_stop_reason'] is None and not t['diagnostic_labels_read']
  for side in ['train','replay']:
   done=read(fold/f'run/{side}/COMPLETE.json');assert done['epochs']==64 and done['updates']==cfg['expected_updates'] and done['native_original_curve_and_selected_reference_verified']
   for arm in ['swad','tail','bn_only','native_reference']:
    for name in [f'selected_{arm}.pt',f'{arm}_validation_unscored.npz',f'{arm}_diagnostic_unscored.npz']:assert f'held{h}/{side}/{name}' in frozen['files']
 backup=P/'full_backup';m=read(backup/'MANIFEST.json');v=read(backup/'READBACK_VERIFIED.json');rows=read(backup/'READBACK_INPUT.json')
 assert m['complete_modelset'] and m['complete_all_six'] and v['complete_modelset_verified']
 for r in rows:
  d=r['metadata'];assert d['shared'] is False and d['permissions']==[dict(type='user',role='owner')] and d['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf'];assert r['raw_fetch'] and r['download_verified'] and r['include_base64'] is False
 assert verify(m,rows)==v
 idx=read(backup/'CLOUD_DOWNLOAD_INDEX.json');assert idx['archive_sha256']==m['archive_sha256'] and idx['complete_all_six']
 baseline=read(P/'operator/INFERENCE_PROTOCOL.json')['baseline_artifacts']
 for r in baseline.values():assert sha(r['model'])==r['model_sha256'] and sha(r['prediction'])==r['prediction_sha256']
 return state,frozen,m,v,idx,baseline

def main():
 started=time.monotonic();cpu=time.process_time();state,frozen,m,v,idx,baseline=preflight()
 assert not(P/'FULL_COMPOSITE_RECOVERY_INDEX.json').exists() and not(P/'FULL_PRIVATE_PREPARATION_RECEIPT.json').exists()
 initial=read(P/'INITIAL_COMPOSITE_RECOVERY_INDEX.json');layout=read(P/'FULL_SOURCE_LAYOUT.json')
 costs={str(h):{k:read(P/f'folds/held{h}/run/execution/{k}/TERMINATION.json')['cumulative'] for k in ['train','replay','output']} for h in range(1,7)}
 index=dict(state='complete_SWAD_all6_canonical_models_and_audits_private_raw_allmember_verified_unscored_NOT_score_release',initial_raw_strong_GAP_all6_initial_recovery_chain=initial,SWAD_complete_canonical_all6=idx,complete_all_six=True,complete_primary_tail_bn_only_and_native_reference_models_and_outputs=True,source_layout=layout,source_frozen_manifest_sha256=sha(P/'FROZEN_HASHES.json'),public_scientific_preregistration=read(P/'REMOTE_PREREGISTRATION.json'),all6_frozen_outputs_sha256=sha(P/'ALL_SIX_PREDICTIONS_FROZEN.json'),original_phase_identity=read(P/'PHASE_IDENTITY.json'),original_phase_must_not_restart=True,actual_main_phase_cost=state,fold_stage_costs_cumulative=costs,baseline_artifact_hashes_currently_rechecked=baseline,baseline_retrain_for_restore_forbidden=True,raw_arrays_and_maps_runtime_external=True,source_maps_restore_helper=dict(repository_path='uwb_direction1/results/cloud_20261108_swad_loso6_initial_private/restore_maps.py',sha256=sha(P/'restore_maps.py')),canonical_archive_full_extract_or_fresh_environment_restore_not_separately_exercised=True,perfold_physical_recovery_proofs_available=[str(p.relative_to(P)) for p in sorted(P.glob('HELD*_COMPLETE_PHYSICAL_RECOVERY_VERIFIED.json'))],raw_fetch_options=dict(download_raw_file=True,include_base64=False),max_complete_chunk_pipelines=3,all_canonical_parts_owner_only_observed=True,held_scores_computed=False,users7_8_labels_used=False,whole_job_resume_authorized=False,score_authorized_by_this_index=False,required_next_steps=['exact_Git_publication_of_this_index_and_proofs','FULL_PRIVATE_RELEASE_binding_exact_commit_archive_and_frozen_once_scorer_SHA','frozen_label_free24arm_inference','one_cohort_score_only'],old_scientific_sources_and_negative_conclusions_unchanged=True)
 write(P/'FULL_COMPOSITE_RECOVERY_INDEX.json',index)
 receipt=dict(state='all6_full_canonical_private_verification_metadata_assembled_still_unscored',archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],chunks=len(m['chunks']),payload_members=v['tracked_payload_members_verified'],raw_allmember_verification=v,all6_complete=True,all4_arms_frozen=True,baseline_hashes_rechecked_without_prediction_truth_decode=True,full_composite_index_sha256=sha(P/'FULL_COMPOSITE_RECOVERY_INDEX.json'),once_scorer_sha256=sha(P/'operator/finalize_cohort_once.py'),inference_operator_sha256=sha(P/'operator/inference_cost.py'),inference_protocol_sha256=sha(P/'operator/INFERENCE_PROTOCOL.json'),helper_source_sha256=sha(__file__),helper_did_not_load_models_or_predictions_or_gesture_labels=True,extra_training_or_model_forward=0,no_private_release_or_once_lock_created=True,wall=time.monotonic()-started,CPU=time.process_time()-cpu)
 write(P/'FULL_PRIVATE_PREPARATION_RECEIPT.json',receipt);print(json.dumps(dict(chunks=receipt['chunks'],payload=receipt['payload_members'],archive_bytes=receipt['archive_bytes'],held_scores_computed=False)))
if __name__=='__main__':
 if sys.argv[1:]==['--check-preconditions']:
  try:preflight()
  except RuntimeError as ex:
   assert str(ex)=='incomplete_all6_no_final_index_or_release_or_score';print(json.dumps(dict(state='blocked_incomplete_all6_before_backup_or_any_truth_loading',final_index_created=False,score_or_release_created=False)))
  else:print(json.dumps(dict(state='all6_private_preconditions_verified_no_index_or_score_created')))
 else:assert sys.argv[1:]==['--build-index'];main()
