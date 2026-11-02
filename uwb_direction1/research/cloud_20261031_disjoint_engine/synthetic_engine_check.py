"""Independent original fit comparison plus full native replay; synthetic only."""
import tempfile,time,json,copy
from source_engine import *
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic()
 g=np.random.RandomState(131);x=g.randn(25,90,567,1).astype('float32');y=np.arange(25,dtype='int64')%12;vx=g.randn(3,90,567,1).astype('float32');vy=np.array([0,1,2],dtype='int64');identity={'fixture':'synthetic25_plus3','classes':12};states=[];records=[]
 with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b,tempfile.TemporaryDirectory() as c:
  h0,m0=make(a,7);original=m0.fit
  def fixed(*args,**kw):kw['shuffle']=False;return original(*args,**kw)
  m0.fit=fixed;h0.fit(x,labels(y,25),vx,labels(vy,3),2,8);baseline=native_capture(m0);baseline_cb=callback_state(h0)
  h,m,cb,cfg=setup(b,7,25,2,8)
  try:
   for ep in range(2):
    records.append(epoch(h,m,cb,cfg,x,y,vx,vy,ep,np.arange(25,dtype='int64')));states.append(snapshot(h,m,cfg,identity,ep+1,int(m.optimizer.iterations.numpy())))
   cb.on_train_end()
   for k in ['weights','optimizer','metrics']:equal(baseline[k],native_capture(m)[k])
   equal(baseline_cb,callback_state(h));assert all(r['last_batch_size']==1 for r in records);assert states[-1]['total_steps']==26
   expected_predictions=predict(m,vx,2);after_prediction=snapshot(h,m,cfg,identity,2,26)
  finally:close(cb)
  h2,m2,cb2,cfg2=setup(c,7,25,2,8)
  try:
   torch.save(states[0],Path(c)/'synthetic_epoch1.pt');saved=torch.load(Path(c)/'synthetic_epoch1.pt',weights_only=False)
   restore(saved,h2,m2,cfg2,identity);equal(saved,snapshot(h2,m2,cfg2,identity,1,13))
   second=epoch(h2,m2,cb2,cfg2,x,y,vx,vy,1,np.arange(25,dtype='int64'));equal(records[1],second);equal(states[1],snapshot(h2,m2,cfg2,identity,2,26))
   replay_predictions=predict(m2,vx,2);equal(expected_predictions,replay_predictions);equal(after_prediction,snapshot(h2,m2,cfg2,identity,2,26))
   # Restore verifies identity before assigning a native variable.
   unchanged=native_capture(m2)
   try:restore(saved,h2,m2,cfg2,{'fixture':'wrong'})
   except ValueError:pass
   else:raise AssertionError('Wrong identity accepted')
   equal(unchanged,native_capture(m2))
   cb2.on_train_end()
  finally:close(cb2)
  for name in ['best_model.keras']:
   orig=keras.models.load_model(Path(a)/name);manual=keras.models.load_model(Path(b)/name);replayed=keras.models.load_model(Path(c)/name)
   equal(orig.get_weights(),manual.get_weights());equal(manual.get_weights(),replayed.get_weights());equal([v.numpy() for v in orig.optimizer.variables],[v.numpy() for v in manual.optimizer.variables]);equal([v.numpy() for v in manual.optimizer.variables],[v.numpy() for v in replayed.optimizer.variables])
 # Dedicated per-epoch sampler: full coverage, repeatable, no global RNG draw.
 before=torch.get_rng_state().clone();o0=permutation(25,0,7);o1=permutation(25,1,7);equal(o0,permutation(25,0,7));assert not np.array_equal(o0,o1);equal(torch.sort(torch.tensor(o0)).values,torch.arange(25));equal(before,torch.get_rng_state())
 rejected=0
 for bad in [np.ones((3,90,567,1),dtype='float64'),np.full((3,90,567,1),np.nan,dtype='float32'),np.ones((3,90,566,1),dtype='float32')]:
  try:values(bad,3)
  except ValueError:rejected+=1
 for bad in [np.array([0.,1.,2.]),np.array([0,1,12]),np.array([0,1,-1]),np.array([True,False,True])]:
  try:labels(bad,3)
  except ValueError:rejected+=1
 assert rejected==7
 result=dict(state='pass_synthetic_explicit_source_engine_original_fit_and_replay',architecture='complete_default_DCNN_2L',native_original_fit_vs_engine_weights_Adam_BN_metrics_callbacks_exact=True,source_weighted_training_unweighted_validation_preserved=True,source_minibatch_rule25_to2_and_last1=True,synthetic_epochs=2,synthetic_updates=26,restore_epoch1_next_epoch_and_all_states_modes_RNG_exact=True,selected_native_keras_weights_optimizer_match_original_and_replay=True,labelblind_predictions_fixed_partition_and_postprediction_states_exact=True,identity_mismatch_rejected_before_mutation=True,dedicated_sampler_full_coverage_repeatable_and_no_global_Torch_rng_draw=True,input_label_numeric_rejections=7,native_fit_DataAdapter_RNG_parity_asserted=False,actual_UWB_runner_or_preprocessing_selected=False,formal_cost_or_real_UWB_values_labels_fit=False,wall_seconds=time.monotonic()-start)
 (Path(__file__).parent/'ENGINE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
