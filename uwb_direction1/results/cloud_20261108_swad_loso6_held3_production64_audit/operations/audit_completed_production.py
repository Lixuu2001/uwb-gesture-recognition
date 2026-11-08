"""Read-only native64 production audit and committed replay prefix; no new forward or score."""
from pathlib import Path
import sys,json,copy,time
P=Path('/workspace/uwb-swad-loso6-20261108');sys.path.insert(0,str(P/'shared'))
from common import read,write,sha,ready
from controller import identity
from journal import load as journal_load
import engine as e

def main(h):
 assert h in range(1,7);started=time.monotonic();cpu=time.process_time();ready();F=P/f'folds/held{h}';R=F/'run';T=R/'train';cfg=read(F/'PROTOCOL.json');N=cfg['expected_updates'];B=N//64;n=cfg['counts'][0];assert not(P/f'HELD{h}_PRODUCTION64_READONLY_AUDIT.json').exists()
 old=read(P/'PHASE_IDENTITY.json');now=identity(old['pid']);assert now and now['state']!='Z' and all(now[k]==old[k] for k in ['pid','boot_id','startticks']);assert not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
 done=read(T/'COMPLETE.json');assert done['epochs']==64 and done['updates']==N and done['native_original_curve_and_selected_reference_verified'] and not done['diagnostic_labels_read']
 hashes={str(f.relative_to(P)):sha(f) for f in sorted(T.rglob('*')) if f.is_file() and not f.name.endswith(('.tmp','.lock'))}
 st,progress=journal_load(T);assert st['epoch']==progress['epoch']==64 and st['updates']==N and e.semantic(st)==done['final_state_sha256']
 assert e.semantic(e.torch.load(T/'final_full.pt',map_location='cpu',weights_only=True))==e.semantic(st)
 m,o,g=e.make();e.restore(copy.deepcopy(st),m,o,g);assert o.native.count==o.swad.updates==N and o.swad.last_epoch==64
 assert e.semantic(e.capture(m,o,g,64,N,st['best_key'],st['best_model'],st['best_epoch']))==e.semantic(st)
 rows=[json.loads(x) for x in (T/'steps.jsonl').read_bytes().splitlines()];assert len(rows)==N
 oldcurve=[json.loads(x) for x in Path(cfg['baseline_curve']['path']).read_text().splitlines()];assert sha(cfg['baseline_curve']['path'])==cfg['baseline_curve']['sha256'] and len(oldcurve)==64
 records=[read(T/f'epoch{k:03}.json') for k in range(1,65)]
 for k,(r,q) in enumerate(zip(records,oldcurve),1):
  step=rows[(k-1)*B:k*B];assert sum(a['forward_batch'] for a in step)==n
  assert sum(a['loss']*a['forward_batch'] for a in step)/n==q['train_bce'] and r['validation_macro']==q['validation_macro'] and r['validation_loss']==q.get('validation_loss',q.get('validation_bce'))
  for j,a in enumerate(step):
   update=(k-1)*B+j+1;assert a['update']==a['completed_updates_after']==update and a['completed_updates_before']==update-1 and a['epoch']==k and a['loss']==a['global_BCE']
   assert a['offset']==j*64 and a['segment_n']==j+1 and a['segment_start']==(k-1)*B+1 and a['segment_end']==update and a['tail_n']==((k-33)*B+j+1 if k>=33 else 0)
 best=max(records,key=lambda r:(r['validation_macro'],-r['validation_loss'],-r['epoch']));assert best['epoch']==st['best_epoch']==done['native_reference_selected_epoch']
 final=copy.deepcopy(o.swad).finalize();params=[n for n,p in m.named_parameters()];states=dict(swad=final.module.state_dict(),tail=st['optimizer']['swad']['tail']['model'],bn_only=st['best_model'],native_reference=st['best_model']);meta={}
 for arm in cfg['arms']:
  selected=e.torch.load(T/f'selected_{arm}.pt',map_location='cpu',weights_only=True);meta[arm]=selected['metadata']
  for name in params:e.torch.testing.assert_close(selected['model'][name],states[arm][name],rtol=0,atol=0)
  if arm=='native_reference':assert e.semantic(selected['model'])==e.semantic(st['best_model'])
  else:
   c=meta[arm]['calibration'];assert c['n']==n and c['source_users']==cfg['train_users'] and c['seed']==21161030 and c['passes']==1 and len(c['BN_batches'])==8 and set(c['BN_batches'].values())=={B} and not c['validation_or_held_inputs_used']
   assert c['parameters_unchanged'] and c['only_BN_stat_buffers_changed'] and c['all_main_RNG_and_modes_and_momenta_restored']
 assert meta['swad']['selection']==done['SWAD_selection'] and meta['tail']['selection']==dict(start_step=32*B+1,end_step=N,n_averaged=32*B)
 ref=cfg['baseline_artifacts'];old_selected=e.legacy_checkpoint(ref['model'],ref['model_sha256']);assert old_selected['epoch']==st['best_epoch'] and e.semantic(old_selected['model'])==e.semantic(st['best_model'])
 with e.np.load(ref['prediction']) as x,e.np.load(T/'native_reference_diagnostic_unscored.npz') as y:
  e.np.testing.assert_array_equal(x['ids'],y['ids']);e.np.testing.assert_array_equal(x['logits'],y['scores'])
 rp=read(R/'replay/STATE.json');epochs=rp['epoch'];updates=rp['updates']
 for k in range(epochs+1):assert read(T/f'epoch{k:03}.json')==read(R/f'replay/epoch{k:03}.json')
 replay_rows=(R/'replay/steps.jsonl').read_bytes().splitlines(keepends=True);production_rows=(T/'steps.jsonl').read_bytes().splitlines(keepends=True);assert len(replay_rows)>=updates and production_rows[:updates]==replay_rows[:updates]
 for name,digest in hashes.items():assert sha(P/name)==digest,name
 result=dict(state='completed_native64_production_all_originalGAP_curves_fullstate_models_sourceBN_metadata_and_fresh_replay_prefix_verified_unscored',held=h,epochs=64,updates=N,all65_production_journal_records_verified=True,all64_originalGAP_train_BCE_and_validation_macro_BCE_exact=True,native_model_optimizer_SWAD_queues_RNG_modes_restore_exact=True,all_step_loss_and_segment_tail_counters_checked=True,all4_selected_parameters_exact_precalibration=True,source_only_BN_metadata_and_all8_counts_verified=True,native_selected_epoch=st['best_epoch'],native_reference_model_and_unscored_logits_exact_original_GAP=True,SWAD_selection=meta['swad']['selection'],tail_selection=meta['tail']['selection'],replay_committed_prefix=dict(epochs=epochs,updates=updates,all_epoch_records_equal=True,all_step_rows_bitwise_equal=True),all64_fresh_replay_complete=(R/'replay/COMPLETE.json').exists(),all4_final_outputs_frozen=(R/'PREDICTIONS_FROZEN.json').exists(),production_file_sha256=hashes,first_independent_gradient_reference=read(T/'FIRST_INDEPENDENT_REFERENCE.json'),source_frozen_sha256=sha(P/'FROZEN_HASHES.json'),source_method_configs_unchanged=True,extra_model_forward_or_optimizer_updates=0,raw_UWB_values_or_gesture_truth_opened=False,only_frozen_unscored_native_reference_scores_compared=True,does_not_exclude_all_shared_implementation_errors=True,all6_complete=False,test_scores_computed=False,observed_phase_resource=read(P/'RESOURCE_LEDGER.json'),wall=time.monotonic()-started,CPU=time.process_time()-cpu)
 write(P/f'HELD{h}_PRODUCTION64_READONLY_AUDIT.json',result);print(json.dumps(dict(held=h,production_epochs=64,production_updates=N,replay_epochs=epochs,replay_updates=updates,native_curves_exact_all64=True,extra_training_updates=0)))
if __name__=='__main__':assert len(sys.argv)==2;main(int(sys.argv[1]))
