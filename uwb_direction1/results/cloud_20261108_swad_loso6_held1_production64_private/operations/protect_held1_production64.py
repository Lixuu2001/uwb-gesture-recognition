"""Read-only production64 insurance during existing replay; no actual data forwards or score."""
from pathlib import Path
import sys,json,time,copy,resource
P=Path('/workspace/uwb-swad-loso6-20261108');sys.path.insert(0,str(P/'shared'))
from common import ready,read,write,sha
from controller import identity
from journal import load as journal_load
from stream_archive import build
import engine as e

def main():
 started=time.monotonic();cpu=time.process_time();ready(True)
 old=read(P/'PHASE_IDENTITY.json');now=identity(old['pid']);assert now and now['state']!='Z' and all(now[k]==old[k] for k in ['pid','boot_id','startticks'])
 assert not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists() and not(P/'HELD1_PRODUCTION64_PROTECTION_AUDIT.json').exists()
 fold=P/'folds/held1';train=fold/'run/train';c=read(fold/'PROTOCOL.json');done=read(train/'COMPLETE.json');production_before={str(f.relative_to(P)):sha(f) for f in sorted(train.rglob('*')) if f.is_file() and not f.name.endswith(('.tmp','.lock'))}
 assert done['epochs']==64 and done['updates']==c['expected_updates']==4736 and done['native_original_curve_and_selected_reference_verified'] and not done['diagnostic_labels_read']
 state,progress=journal_load(train);assert progress['epoch']==state['epoch']==64 and state['updates']==4736 and e.semantic(state)==done['final_state_sha256']
 final=e.torch.load(train/'final_full.pt',map_location='cpu',weights_only=True);assert e.semantic(final)==e.semantic(state)
 m,o,g=e.make();e.restore(copy.deepcopy(state),m,o,g);assert o.native.count==o.swad.updates==4736 and o.swad.last_epoch==64
 assert e.semantic(e.capture(m,o,g,64,4736,state['best_key'],state['best_model'],state['best_epoch']))==e.semantic(state)
 records=[read(train/f'epoch{k:03}.json') for k in range(1,65)];best=max(records,key=lambda r:(r['validation_macro'],-r['validation_loss'],-r['epoch']));assert best['epoch']==state['best_epoch']==done['native_reference_selected_epoch']==60
 oldcurve=[json.loads(x) for x in Path(c['baseline_curve']['path']).read_text().splitlines()];assert len(oldcurve)==64 and sha(c['baseline_curve']['path'])==c['baseline_curve']['sha256']
 losses=[0.]*64;counts=[0]*64;rows=0
 for line in (train/'steps.jsonl').open():
  r=json.loads(line);rows+=1;epoch=r['epoch'];j=(rows-1)%74
  assert r['update']==r['completed_updates_after']==rows and r['completed_updates_before']==rows-1 and epoch==(rows-1)//74+1
  assert r['offset']==j*64 and r['forward_batch']==(64 if j<73 else 28) and r['loss']==r['global_BCE'] and r['segment_n']==j+1
  assert r['segment_start']==(epoch-1)*74+1 and r['segment_end']==rows
  assert r['tail_n']==(max(0,epoch-33)*74+j+1 if epoch>=33 else 0)
  losses[epoch-1]+=r['loss']*r['forward_batch'];counts[epoch-1]+=r['forward_batch']
 assert rows==4736
 for k,(r,oldr) in enumerate(zip(records,oldcurve)):
  assert counts[k]==4700 and losses[k]/4700==oldr['train_bce'] and r['validation_macro']==oldr['validation_macro'] and r['validation_loss']==oldr.get('validation_bce',oldr.get('validation_loss'))
 method=copy.deepcopy(o.swad);averaged=method.finalize();names=[n for n,p in m.named_parameters()]
 preBN=dict(swad=averaged.module.state_dict(),tail=state['optimizer']['swad']['tail']['model'],bn_only=state['best_model'],native_reference=state['best_model']);selections={}
 for arm in c['arms']:
  selected=e.torch.load(train/f'selected_{arm}.pt',map_location='cpu',weights_only=True);selections[arm]=selected['metadata']
  for n in names:e.torch.testing.assert_close(selected['model'][n],preBN[arm][n],rtol=0,atol=0)
  if arm=='native_reference':assert e.semantic(selected['model'])==e.semantic(state['best_model'])
  else:
   cal=selected['metadata']['calibration'];assert cal['parameters_unchanged'] and cal['all_main_RNG_and_modes_and_momenta_restored'] and cal['only_BN_stat_buffers_changed'] and not cal['validation_or_held_inputs_used']
   assert cal['source_users']==c['train_users'] and cal['seed']==21161030 and cal['passes']==1 and cal['n']==4700 and cal['forward_batches']==74 and len(cal['BN_batches'])==8 and set(cal['BN_batches'].values())=={74}
 assert selections['swad']['selection']==done['SWAD_selection']
 assert selections['tail']['selection']==dict(start_step=2369,end_step=4736,n_averaged=2368)
 assert all(selections[x]['selection']['native_reference_selected_epoch']==60 for x in ['bn_only','native_reference'])
 ref=c['baseline_artifacts'];cached=e.legacy_checkpoint(ref['model'],ref['model_sha256']);assert e.semantic(cached['model'])==e.semantic(state['best_model']) and cached['epoch']==60
 with e.np.load(ref['prediction']) as z,e.np.load(train/'native_reference_diagnostic_unscored.npz') as other:
  e.np.testing.assert_array_equal(z['ids'],other['ids']);e.np.testing.assert_array_equal(z['logits'],other['scores'])
 # No label-bearing entries in the old prediction archive are loaded.
 for n,key in [('selected.pt','selected_sha256'),('validation_unscored.npz','validation_outputs_sha256'),('diagnostic_unscored.npz','diagnostic_outputs_sha256')]:assert sha(train/n)==done[key]
 result=dict(state='held1_completed_SWAD_production64_readonly_fullstate_all64_originalGAP_curves_all4_parameters_and_sourceBN_metadata_verified_unscored',production_epochs=64,production_updates=4736,all65_journal_records_verified=True,full_native_optimizer_SWAD_queues_RNG_mode_deserialization_exact=True,all64_training_BCE_and_validation_macro_BCE_exact_original_GAP=True,all4736_step_counts_loss_and_segment_tail_counters_checked=True,independent_per_step_average_parameter_reconstruction_from_real_weights_not_available=True,primary_SWAD_range=selections['swad']['selection'],primary_SWAD_n_averaged_definition='pinned author AveragedModel final segment-merge count, including preserved queue corner repeats; not dense optimizer update count',fixed_tail=selections['tail']['selection'],native_reference_validation_selected_epoch=60,calibrated_arms_source_only_BN_metadata_and_parameters_exact=True,native_reference_model_epoch_and_held_unscored_logits_exact_original_GAP=True,raw_training_validation_or_held_UWB_values_or_gesture_truth_opened_by_this_operator=False,held_unscored_logits_loaded_only_for_bitwise_native_reference_comparison=True,additional_model_forwards_or_training_updates=0,first_gradient_reference=read(train/'FIRST_INDEPENDENT_REFERENCE.json'),production_sha256=production_before,snapshot_phase_resource=read(P/'RESOURCE_LEDGER.json'),replay_observation=read(fold/'run/replay/STATE.json'),independent64_replay_complete=(fold/'run/replay/COMPLETE.json').exists(),complete_all_six=False,does_not_exclude_all_shared_implementation_errors=True,users7_8_labels_used=False,wall=time.monotonic()-started,CPU=time.process_time()-cpu)
 write(P/'HELD1_PRODUCTION64_PROTECTION_AUDIT.json',result)
 files={'phase/'+n:P/n for n in [*read(P/'FROZEN_HASHES.json'),'FROZEN_HASHES.json','REAL_RELEASE.json','REMOTE_PREREGISTRATION.json','INITIAL_ASSET.json','INITIAL_COMPOSITE_RECOVERY_INDEX.json','PRELAUNCH_PROCESS_AND_RELEASE_CHECK.json','LAUNCH_OBSERVATION.json','HELD1_PRODUCTION64_PROTECTION_AUDIT.json']}
 for f in train.rglob('*'):
  if f.is_file() and not f.name.endswith(('.tmp','.lock')):files['phase/'+str(f.relative_to(P))]=f
 for n in ['initial.pt','roles.npz','PROTOCOL.json']:
  f=fold/n;files['phase/'+str(f.relative_to(P))]=f
 for n in read(P/'shared/DEPENDENCIES.json')['files']:files['native_source/'+str(Path(n).relative_to(c['native_source']))]=Path(n)
 write(P/'HELD1_PRODUCTION64_SOURCE_LAYOUT.json',dict(archive_to_expected_path={n:str(f) for n,f in files.items()},scope='held1 production64 only; replay active and not archived; not complete LOSO6 or whole job resume',preserve_existing_phase_identity_and_budget=True,no_automatic_relaunch=True,do_not_retrain_completed_production=True,cohort_scoring_requires_all6_final_private_release=True));files['SOURCE_LAYOUT.json']=P/'HELD1_PRODUCTION64_SOURCE_LAYOUT.json'
 a=build(P/'held1_production64_backup','swad_native_loso6_held1_production64_20261108',files,'partial_SWAD_held1_production64_all4_model_outputs_fullqueues_logs_unscored_NOT_all6',chunk_size=7340032,limit=268435456)
 a.update(complete_modelset=False,complete_all_six=False,complete_training_modelset=False,held1_production64_complete=True,held1_replay_included=False);write(P/'held1_production64_backup/MANIFEST.json',a)
 for n,h in production_before.items():assert sha(P/n)==h,n
 print(json.dumps(dict(state=result['state'],chunks=len(a['chunks']),bytes=a['archive_bytes'],payload=a['tracked_payload_files'],actual_model_phase_untouched=True)))
if __name__=='__main__':main()
