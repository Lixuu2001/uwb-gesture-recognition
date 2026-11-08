"""Read-only completed production audit and partial private archive; no held scoring/new fit."""
from pathlib import Path
import sys,json,time,copy,resource
P=Path('/workspace/uwb-groupdro-loso6-20261108');sys.path.insert(0,str(P/'shared'))
from common import ready,read,write,sha
from controller import identity
import engine as e
from journal import load as journal_load
from stream_archive import build

def main():
 started=time.monotonic();cpu=time.process_time();ready(True)
 old=read(P/'PHASE_IDENTITY.json');now=identity(old['pid']);assert now and now['state']!='Z' and all(now[k]==old[k] for k in ['pid','boot_id','startticks'])
 assert not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
 fold=P/'folds/held1';train=fold/'run/train';c=read(fold/'PROTOCOL.json');done=read(train/'COMPLETE.json')
 assert done['epochs']==64 and done['updates']==c['expected_updates']==4736 and not done['diagnostic_labels_read']
 state,progress=journal_load(train);assert progress['epoch']==state['epoch']==64 and state['updates']==4736
 assert e.semantic(state)==done['final_state_sha256']
 final=e.torch.load(train/'final_full.pt',map_location='cpu',weights_only=True);assert e.semantic(final)==e.semantic(state)
 q=state['optimizer']['groupDRO'];native=state['optimizer']['native_optimizer'];assert q['updates']==4736 and tuple(q['domain_users'])==tuple(c['train_users'])
 # Exercise full state deserialization without any model forward/optimizer update/UWB value load.
 m,o,g=e.make();e.restore(copy.deepcopy(state),m,o,g);assert o.native.count==o.robust.updates==4736 and e.semantic(e.capture(m,o,g,64,4736,state['best_key'],state['best_model'],state['best_epoch']))==e.semantic(state)
 selected=e.torch.load(train/'selected.pt',map_location='cpu',weights_only=True);assert selected['selected_epoch']==done['selected_epoch']==state['best_epoch'] and e.semantic(selected['model'])==e.semantic(state['best_model'])
 records=[read(train/f'epoch{k:03}.json') for k in range(1,65)]
 best=max(records,key=lambda r:(r['validation_macro'],-r['validation_loss'],-r['epoch']))
 assert best['epoch']==selected['selected_epoch']
 for n,key in [('selected.pt','selected_sha256'),('validation_unscored.npz','validation_outputs_sha256'),('diagnostic_unscored.npz','diagnostic_outputs_sha256')]:assert sha(train/n)==done[key]
 # Independent per-step risk/q arithmetic, using logged risk/current q only. This is not independent verification of risks against real data.
 previous=e.np.ones(4,dtype=e.np.float64);maxq=0.;maxloss=0.;rows=0;missing=0
 for line in (train/'steps.jsonl').open():
  r=json.loads(line);rows+=1;assert r['update']==r['completed_updates_after']==rows and r['completed_updates_before']==rows-1 and r['group_users']==c['train_users']
  counts=e.np.array(r['domain_counts']);risks=e.np.array(r['source_risks']);actual=e.np.array(r['group_q']);present=counts>0
  assert counts.sum()==r['forward_batch'] and list(present)==r['present'] and e.np.isfinite(risks).all() and e.np.isfinite(actual).all() and (actual>=0).all()
  expected=previous.copy();expected[present]*=e.np.exp(.01*risks[present]);expected/=expected.sum();maxq=max(maxq,float(abs(expected-actual).max()));assert maxq<=1.5e-7
  weights=actual.copy()
  if not present.all():weights[present]/=weights[present].sum();missing+=1
  loss=float(e.np.dot(weights[present],risks[present]));maxloss=max(maxloss,abs(loss-r['loss']));assert maxloss<=2e-6 and r['loss']==r['weighted_source_BCE']
  previous=actual
 assert rows==4736; e.np.testing.assert_array_equal(previous,q['q'].numpy())
 result=dict(state='held1_completed_production64_readonly_native_q_journal_validation_selection_and_output_hash_audit_passed_unscored',held=1,production_epochs=64,production_updates=4736,selected_epoch_validation_only=selected['selected_epoch'],all65_journal_records_verified=True,native_full_model_optimizer_q_RNG_mode_deserialization_exact=True,all4736_logged_q_transitions_independent_NumPy_one_step_from_previous_logged_float32_q=True,logged_risks_not_independently_verified_against_actual_data_by_this_audit=True,max_q_arithmetic_absolute_difference=maxq,max_weighted_loss_arithmetic_absolute_difference=maxloss,missing_group_batches=missing,diagnostic_label_values_or_prediction_values_decoded=False,additional_model_forward_or_training_updates=0,prior_first_gradient_reference_preserved=read(train/'FIRST_INDEPENDENT_REFERENCE.json'),production_sha256={str(f.relative_to(P)):sha(f) for f in sorted(train.rglob('*')) if f.is_file() and not f.name.endswith(('.tmp','.lock'))},snapshot_phase_resource=read(P/'RESOURCE_LEDGER.json'),replay_observation=read(fold/'run/replay/STATE.json'),independent64_replay_complete=(fold/'run/replay/COMPLETE.json').exists(),full6_complete=False,users7_8_labels_used=False,wall=time.monotonic()-started,CPU=time.process_time()-cpu)
 write(P/'HELD1_PRODUCTION64_PROTECTION_AUDIT.json',result)
 files={'phase/'+n:P/n for n in [*read(P/'FROZEN_HASHES.json'),'FROZEN_HASHES.json','REAL_RELEASE.json','REMOTE_PREREGISTRATION.json','INITIAL_ASSET.json','INITIAL_COMPOSITE_RECOVERY_INDEX.json','OPERATOR_LAUNCH.json','LAUNCH_PREFLIGHT.json','HELD1_PRODUCTION64_PROTECTION_AUDIT.json']}
 for f in train.rglob('*'):
  if f.is_file() and not f.name.endswith(('.tmp','.lock')):files['phase/'+str(f.relative_to(P))]=f
 for n in ['initial.pt','roles.npz','PROTOCOL.json']:
  f=fold/n;files['phase/'+str(f.relative_to(P))]=f
 for n in read(P/'shared/DEPENDENCIES.json')['files']:files['native_source/'+str(Path(n).relative_to(c['native_source']))]=Path(n)
 layout=dict(archive_to_expected_path={n:str(f) for n,f in files.items()},scope='held1 production64 only; replay actively running and not archived; not complete sixfold modelset',preserve_existing_phase_identity_and_budget=True,no_automatic_relaunch=True,do_not_retrain_completed_production=True,cohort_scoring_requires_all6_final_private_release=True)
 write(P/'HELD1_PRODUCTION64_SOURCE_LAYOUT.json',layout);files['SOURCE_LAYOUT.json']=P/'HELD1_PRODUCTION64_SOURCE_LAYOUT.json'
 wall=time.monotonic();cp=time.process_time()
 a=build(P/'held1_production64_backup','groupdro_native_loso6_held1_production64_20261108',files,'partial_GroupDRO_held1_production64_native_q_full_state_logs_outputs_unscored_NOT_all6',chunk_size=7340032,limit=268435456)
 a.update(complete_modelset=False,complete_all_six=False,complete_training_modelset=False,held1_production64_complete=True,held1_replay_included=False);write(P/'held1_production64_backup/MANIFEST.json',a)
 # Confirm immutable source bytes still match the audited hashes after archive construction.
 for n,h in result['production_sha256'].items():assert sha(P/n)==h,n
 write(P/'HELD1_PRODUCTION64_PROTECTION_COST.json',dict(readonly_audit_wall=result['wall'],readonly_audit_CPU=result['CPU'],archive_build_wall=time.monotonic()-wall,archive_build_parent_CPU=time.process_time()-cp,archive_build_child_CPU_not_measured=True,operator_total_wall=time.monotonic()-started,operator_parent_CPU=time.process_time()-cpu,operator_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,not_part_of_training_process_family_CPU=True,main_phase_wall_keeps_running=True))
 print(json.dumps(dict(state=result['state'],q_maxdiff=maxq,weighted_loss_maxdiff=maxloss,chunks=len(a['chunks']),bytes=a['archive_bytes'],payload=len(a.get('members',[])),phase_untouched=True)))
if __name__=='__main__':main()
