"""Exact-byte storage dedup for independently produced replay; never fit or score."""
from pathlib import Path
import sys,json,copy,time
P=Path('/workspace/uwb-swad-loso6-20261108');sys.path.insert(0,str(P/'shared'))
from common import read,write,sha,ready
from controller import identity
from stream_archive import build,verify
from journal import load as journal_load
import engine as e

def main():
 started=time.monotonic();cpu=time.process_time();ready();assert not(P/'HELD1_COMPLETION_AUDIT.json').exists()
 old=read(P/'PHASE_IDENTITY.json');now=identity(old['pid']);assert now and now['state']!='Z' and all(now[k]==old[k] for k in ['pid','boot_id','startticks'])
 assert not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
 F=P/'folds/held1';R=F/'run';assert read(F/'PHASE_TERMINATION.json')['exit_code']==0
 local=read(R/'PREDICTIONS_FROZEN.json');assert local['full_training_replayed'] and local['native_reference_matches_original_GAP'] and local['epochs']==64 and local['updates']==4736
 for n,h in local['files'].items():assert sha(R/n)==h,n
 for k in range(65):assert read(R/f'train/epoch{k:03}.json')==read(R/f'replay/epoch{k:03}.json')
 assert (R/'train/steps.jsonl').read_bytes()==(R/'replay/steps.jsonl').read_bytes()
 assert len((R/'train/steps.jsonl').read_bytes().splitlines())==4736
 a,ap=journal_load(R/'train');b,bp=journal_load(R/'replay');assert e.semantic(a)==e.semantic(b)
 for side,st in [('train',a),('replay',b)]:
  c=read(R/side/'COMPLETE.json');assert c['epochs']==64 and c['updates']==4736 and c['native_original_curve_and_selected_reference_verified'] and c['final_state_sha256']==e.semantic(st)
  assert e.semantic(e.torch.load(R/side/'final_full.pt',map_location='cpu',weights_only=True))==e.semantic(st)
 for arm in ['swad','tail','bn_only','native_reference']:
  x=e.torch.load(R/f'train/selected_{arm}.pt',map_location='cpu',weights_only=True);y=e.torch.load(R/f'replay/selected_{arm}.pt',map_location='cpu',weights_only=True);assert e.semantic(x)==e.semantic(y)
  for role in ['validation','diagnostic']:
   with e.np.load(R/f'train/{arm}_{role}_unscored.npz') as x,e.np.load(R/f'replay/{arm}_{role}_unscored.npz') as y:
    assert set(x.files)==set(y.files)=={'ids','scores'}
    for name in x.files:e.np.testing.assert_array_equal(x[name],y[name])
 for stage in ['train','replay','output']:
  t=read(R/f'execution/{stage}/TERMINATION.json');assert t['exit_code']==0 and t['resource_stop_reason'] is None and not t['diagnostic_labels_read']
 base=P/'held1_production64_backup';manifest=read(base/'MANIFEST.json');oldproof=read(base/'READBACK_VERIFIED.json');assert verify(manifest,read(base/'READBACK_INPUT.json'))==oldproof
 lookup={(r['bytes'],r['sha256']):r['path'] for r in read(base/'MEMBERS.json')['members']}
 files={};layout=[];digests={};counts=dict(base_files=0,supplement_files=0,base_bytes=0,supplement_bytes=0)
 for f in sorted(F.rglob('*')):
  if not f.is_file() or f.name.endswith(('.tmp','.lock')):continue
  rel=str(f.relative_to(P));size=f.stat().st_size;digest=sha(f);pair=(size,digest);digests[rel]=digest
  if pair in lookup:
   row=dict(phase_relative_path=rel,bytes=size,sha256=digest,source='base',archive_member=lookup[pair]);counts['base_files']+=1;counts['base_bytes']+=size
  else:
   member='new/'+rel;files[member]=f;row=dict(phase_relative_path=rel,bytes=size,sha256=digest,source='supplement',archive_member=member);counts['supplement_files']+=1;counts['supplement_bytes']+=size
  layout.append(row)
 costs={k:read(R/f'execution/{k}/TERMINATION.json')['cumulative'] for k in ['train','replay','output']};increments={};previous=dict(wall=0.,cpu=0.)
 for k,v in costs.items():increments[k]={n:v[n]-previous[n] for n in ['wall','cpu']};previous=v
 result=dict(state='held1_complete_native64_independentfresh64_all4_models_outputs_and_third_sourceBN_audit_verified_unscored',epochs_per_production_and_replay=64,updates_per_production_and_replay=4736,all65_full_epoch_records_equal=True,all4736_step_rows_bitwise_equal=True,full_model_optimizer_SWAD_queues_RNG_modes_equal=True,all4_selected_models_and_all8_role_outputs_bitwise_equal=True,source_only_third_BN_output_audit_complete=True,original_GAP_reference_and_all64_curves_verified=True,earlier_production_audit_sha256=sha(P/'HELD1_PRODUCTION64_PROTECTION_AUDIT.json'),all_frozen_model_output_hashes_verified=True,actual_independent_execution_logs_and_identities_preserved=True,exact_byte_dedup_is_storage_only_not_replay_from_cached_weights=True,source_file_sha256=digests,storage_counts=counts,fold_stage_costs_cumulative=costs,fold_stage_costs_by_difference=increments,outer_fold_cost=read(F/'PHASE_TERMINATION.json')['cumulative_remaining_phase'],base_archive_sha256=manifest['archive_sha256'],base_prior_raw_readback_reused_plus_current_local_full_member_reverification=True,additional_model_forward_or_optimizer_updates=0,raw_UWB_values_or_gesture_truth_opened=False,only_already_frozen_unscored_scores_compared=True,cohort_scored=False,complete_all_six=False,does_not_exclude_all_shared_implementation_errors=True,observed_phase_resource=read(P/'RESOURCE_LEDGER.json'),wall=time.monotonic()-started,CPU=time.process_time()-cpu)
 write(P/'HELD1_COMPLETION_AUDIT.json',result)
 recipe=dict(state='lossless_full_completed_fold_exact_byte_dedup_recipe',base_archive_sha256=manifest['archive_sha256'],base_members_manifest_sha256=manifest['members_manifest_sha256'],base_private_index_sha256=sha(base/'CLOUD_DOWNLOAD_INDEX.json'),rows=layout,files=len(layout),bytes=sum(r['bytes'] for r in layout),storage_counts=counts,all_new_training_updates=0,weights_must_not_be_retrained_or_approximated=True,copy_after_sha_and_size_verification=True,actual_source_files_preserved=True,restore_only_under_empty_isolated_destination=True,locks_and_uncommitted_tmp_files_excluded=True,whole_job_resume_authorized=False,complete_all_six=False)
 write(P/'HELD1_COMPLETION_DEDUP_RECIPE.json',recipe)
 files['RECIPE.json']=P/'HELD1_COMPLETION_DEDUP_RECIPE.json';files['AUDIT.json']=P/'HELD1_COMPLETION_AUDIT.json'
 files['operators/build_held1_completion_supplement.py']=Path(__file__);files['operators/restore_completed_fold_dedup.py']=P/'operations/restore_completed_fold_dedup.py'
 m=build(P/'held1_completion_backup','swad_native_loso6_held1_completed_supplement_20261108',files,'SWAD_held1_completed_production64_plus_independent64_immutable_supplement_requires_prior_production22_exact_dedup_recipe_NOT_all6',chunk_size=7340032,limit=16777216)
 m.update(complete_modelset=False,complete_all_six=False,complete_single_fold_with_base=True,requires_base_archive_sha256=manifest['archive_sha256']);write(P/'held1_completion_backup/MANIFEST.json',m)
 for n,h in digests.items():assert sha(P/n)==h,n
 print(json.dumps(dict(state=result['state'],files=recipe['files'],bytes=recipe['bytes'],counts=counts,archive_bytes=m['archive_bytes'],chunks=len(m['chunks']),payload=m['tracked_payload_files'])))
if __name__=='__main__':main()
