"""Audit a completed fold and store unique exact bytes once; no fitting/scoring/retry."""
from pathlib import Path
import sys,json,time
P=Path('/workspace/uwb-swad-loso6-20261108');sys.path.insert(0,str(P/'shared'))
from common import read,write,sha,ready
from controller import identity
from stream_archive import build,verify
from journal import load as journal_load
import engine as e

def main(h):
 assert h in range(2,7);started=time.monotonic();cpu=time.process_time();ready();out=P/f'held{h}_completed_backup';assert not out.exists() and not(P/f'HELD{h}_COMPLETE_OBJECT_AUDIT.json').exists()
 old=read(P/'PHASE_IDENTITY.json');now=identity(old['pid']);active=now and now['state']!='Z' and all(now[k]==old[k] for k in ['pid','boot_id','startticks']);phase=read(P/'STATE.json');assert active or (phase['state']=='all_six_method_folds_frozen_unscored_requires_complete_private_backup_before_cohort_once' and phase['completed_folds']==list(range(1,7)))
 assert not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
 F=P/f'folds/held{h}';R=F/'run';cfg=read(F/'PROTOCOL.json');N=cfg['expected_updates'];B=N//64;train_n=cfg['counts'][0]
 assert read(F/'PHASE_TERMINATION.json')['exit_code']==0
 frozen=read(R/'PREDICTIONS_FROZEN.json');assert frozen['full_training_replayed'] and frozen['native_reference_matches_original_GAP'] and frozen['epochs']==64 and frozen['updates']==N
 for n,digest in frozen['files'].items():assert sha(R/n)==digest,n
 for k in range(65):assert read(R/f'train/epoch{k:03}.json')==read(R/f'replay/epoch{k:03}.json')
 assert (R/'train/steps.jsonl').read_bytes()==(R/'replay/steps.jsonl').read_bytes()
 steps=[json.loads(line) for line in (R/'train/steps.jsonl').read_bytes().splitlines()];assert len(steps)==N
 curves=[json.loads(x) for x in Path(cfg['baseline_curve']['path']).read_text().splitlines()];assert sha(cfg['baseline_curve']['path'])==cfg['baseline_curve']['sha256'] and len(curves)==64
 for k in range(1,65):
  records=steps[(k-1)*B:k*B];oldr=curves[k-1];r=read(R/f'train/epoch{k:03}.json');assert sum(a['forward_batch'] for a in records)==train_n
  assert sum(a['loss']*a['forward_batch'] for a in records)/train_n==oldr['train_bce'] and r['validation_macro']==oldr['validation_macro'] and r['validation_loss']==oldr.get('validation_loss',oldr.get('validation_bce'))
 a,_=journal_load(R/'train');b,_=journal_load(R/'replay');assert e.semantic(a)==e.semantic(b)
 for side,st in [('train',a),('replay',b)]:
  c=read(R/side/'COMPLETE.json');assert c['epochs']==64 and c['updates']==N and c['native_original_curve_and_selected_reference_verified'] and c['final_state_sha256']==e.semantic(st)
  assert e.semantic(e.torch.load(R/side/'final_full.pt',map_location='cpu',weights_only=True))==e.semantic(st)
 for arm in cfg['arms']:
  x=e.torch.load(R/f'train/selected_{arm}.pt',map_location='cpu',weights_only=True);y=e.torch.load(R/f'replay/selected_{arm}.pt',map_location='cpu',weights_only=True);assert e.semantic(x)==e.semantic(y)
  if arm!='native_reference':
   cal=x['metadata']['calibration'];assert cal['parameters_unchanged'] and cal['all_main_RNG_and_modes_and_momenta_restored'] and cal['only_BN_stat_buffers_changed'] and not cal['validation_or_held_inputs_used'];assert cal['seed']==21161030 and cal['passes']==1 and cal['n']==train_n and cal['source_users']==cfg['train_users']
  for role in ['validation','diagnostic']:
   with e.np.load(R/f'train/{arm}_{role}_unscored.npz') as x,e.np.load(R/f'replay/{arm}_{role}_unscored.npz') as y:
    assert set(x.files)==set(y.files)=={'ids','scores'}
    for key in x.files:e.np.testing.assert_array_equal(x[key],y[key])
 costs={k:read(R/f'execution/{k}/TERMINATION.json')['cumulative'] for k in ['train','replay','output']};increments={};prior=dict(wall=0.,cpu=0.)
 for k,v in costs.items():
  term=read(R/f'execution/{k}/TERMINATION.json');assert term['exit_code']==0 and term['resource_stop_reason'] is None and not term['diagnostic_labels_read'];increments[k]={n:v[n]-prior[n] for n in ['wall','cpu']};prior=v
 base=P/'held1_production64_backup';bm=read(base/'MANIFEST.json');assert verify(bm,read(base/'READBACK_INPUT.json'))==read(base/'READBACK_VERIFIED.json')
 base_map={(r['bytes'],r['sha256']):r['path'] for r in read(base/'MEMBERS.json')['members']};new_map={};files={};recipe_rows=[];digests={};base_files=base_bytes=internal_duplicates=0;physical_new_bytes=0
 for f in sorted(F.rglob('*')):
  if not f.is_file() or f.name.endswith(('.tmp','.lock')):continue
  n=str(f.relative_to(P));size=f.stat().st_size;digest=sha(f);pair=(size,digest);digests[n]=digest
  if pair in base_map:source='base';member=base_map[pair];base_files+=1;base_bytes+=size
  else:
   source='new'
   if pair in new_map:member=new_map[pair];internal_duplicates+=1
   else:member='objects/'+digest;assert member not in files;new_map[pair]=member;files[member]=f;physical_new_bytes+=size
  recipe_rows.append(dict(phase_relative_path=n,bytes=size,sha256=digest,source=source,archive_member=member))
 recipe=dict(state='exact_content_object_recipe_actual_independently_produced_completed_fold',held=h,base_archive_sha256=bm['archive_sha256'],base_members_manifest_sha256=bm['members_manifest_sha256'],rows=recipe_rows,files=len(recipe_rows),bytes=sum(r['bytes'] for r in recipe_rows),base_files=base_files,base_logical_bytes=base_bytes,new_unique_objects=len(new_map),new_unique_bytes=physical_new_bytes,within_new_archive_duplicate_targets=internal_duplicates,weights_must_not_be_refit_or_approximated=True,storage_only_not_cached_replay=True,original_local_files_unchanged=True,isolated_empty_restore_destination_required=True,whole_job_resume_authorized=False,complete_all_six=False)
 write(P/f'HELD{h}_COMPLETE_OBJECT_RECIPE.json',recipe)
 audit=dict(state='completed_fold_native64_fresh64_all4_output_audits_and_originalGAP_curves_verified_before_lossless_private_object_storage',held=h,epochs_each=64,updates_each=N,all65_full_epoch_records_equal=True,all_update_step_rows_bitwise_equal=True,full_model_optimizer_SWAD_RNG_and_modes_equal=True,all4_selected_models_and8_role_outputs_equal=True,third_sourceBN_output_audit_complete=True,all64_originalGAP_trainingBCE_validation_macro_BCE_exact=True,source_only_calibration_metadata_verified=True,first_independent_gradient_reference=read(R/'train/FIRST_INDEPENDENT_REFERENCE.json'),SWAD_selection=frozen['SWAD_selection'],native_reference_selected_epoch=frozen['native_reference_selected_epoch'],source_file_sha256=digests,recipe_sha256=sha(P/f'HELD{h}_COMPLETE_OBJECT_RECIPE.json'),fold_costs_cumulative=costs,fold_costs_by_difference=increments,outer_fold_cumulative=read(F/'PHASE_TERMINATION.json')['cumulative_remaining_phase'],storage_only_no_independent_training_replaced_by_cached_model=True,extra_model_forward_or_optimizer_updates=0,raw_UWB_values_or_gesture_truth_opened=False,only_frozen_unscored_scores_compared=True,does_not_exclude_all_shared_implementation_errors=True,complete_all_six=False,cohort_scored=False,observed_phase_resource=read(P/'RESOURCE_LEDGER.json'),wall=time.monotonic()-started,CPU=time.process_time()-cpu)
 write(P/f'HELD{h}_COMPLETE_OBJECT_AUDIT.json',audit)
 files['RECIPE.json']=P/f'HELD{h}_COMPLETE_OBJECT_RECIPE.json';files['AUDIT.json']=P/f'HELD{h}_COMPLETE_OBJECT_AUDIT.json'
 for name in ['archive_completed_fold_objects.py','restore_fold_objects.py','verify_completed_fold_objects.py']:files['operators/'+name]=P/'operations'/name
 m=build(out,f'swad_native_loso6_held{h}_completed_objects_20261108',files,f'complete_SWAD_held{h}_native64_fresh64_all4_audits_lossless_objects_requires_prior_held1production22_NOT_all6',chunk_size=7340032,limit=268435456)
 m.update(held=h,requires_base_archive_sha256=bm['archive_sha256'],complete_modelset=False,complete_single_fold_with_verified_base_and_recipe=True,complete_all_six=False);write(out/'MANIFEST.json',m)
 for n,digest in digests.items():assert sha(P/n)==digest,n
 print(json.dumps(dict(held=h,files=recipe['files'],logical_bytes=recipe['bytes'],new_unique_objects=len(new_map),new_unique_bytes=physical_new_bytes,chunks=len(m['chunks']),archive_bytes=m['archive_bytes'],payload=m['tracked_payload_files'])))
if __name__=='__main__':assert len(sys.argv)==2;main(int(sys.argv[1]))
