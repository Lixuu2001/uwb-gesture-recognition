"""Synthetic full-size model/callback/epoch/prediction exact restoration tests."""
from pathlib import Path
import copy,json,tempfile,time
from engine import *
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic()
 g=np.random.RandomState(931);x=(.1*g.randn(5,567,90)).astype('float32');x[0,13,:]=0
 y=np.array([0,1,2,0,1],dtype='int64');vx=(.1*g.randn(3,567,90)).astype('float32');vy=np.array([0,1,2],dtype='int64');identity={'scope':'synthetic5train3validation','classes':12}
 with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
  h,m,cb,settings=setup(a,7,5,3,2);assert m.count_params()==860364;states=[];records=[];first=None
  try:
   initial=snapshot(h,m,settings,identity,0,0)
   def batchcheck(r):
    nonlocal first
    if first is None:first=snapshot(h,m,settings,identity,1,r['total_steps'],r['batch'])
   for e in range(3):
    records.append(epoch(h,m,cb,settings,x,y,vx,vy,e,permutation(5,e,7),batchcheck));states.append(snapshot(h,m,settings,identity,e+1,(e+1)*3,2))
   cb.on_train_end();assert int(m.optimizer.iterations.numpy())==9 and all(r['last_batch_size']==1 for r in records)
   assert len(records[0]['batch_records'])==3 and set(records[0]['class_weights'])=={0,1,2}
   assert records[0]['class_weights']=={0:5/6,1:5/6,2:5/3}
   expected=predict(m,vx,2);expected_state=snapshot(h,m,settings,identity,3,9,2)
   torch.save(states[0],Path(a)/'synthetic_epoch1.pt');saved=torch.load(Path(a)/'synthetic_epoch1.pt',weights_only=False)
  finally:close(cb)
  h2,m2,cb2,s2=setup(b,7,5,3,2)
  try:
   equal(initial,snapshot(h2,m2,s2,identity,0,0))
   restore(first,h2,m2,s2,identity);equal(first,snapshot(h2,m2,s2,identity,1,1,0))
   restore(saved,h2,m2,s2,identity);equal(saved,snapshot(h2,m2,s2,identity,1,3,2))
   for e in [1,2]:
    got=epoch(h2,m2,cb2,s2,x,y,vx,vy,e,permutation(5,e,7));equal(records[e],got);equal(states[e],snapshot(h2,m2,s2,identity,e+1,(e+1)*3,2))
   cb2.on_train_end();equal(expected,predict(m2,vx,2));equal(expected_state,snapshot(h2,m2,s2,identity,3,9,2))
   before=capture_model(m2)
   for mutation in ['identity','settings','cursor']:
    bad=copy.deepcopy(saved)
    if mutation=='identity':bad['identity']={'wrong':True}
    elif mutation=='settings':bad['settings']['seed']=8
    else:bad['total_steps']=4
    try:restore(bad,h2,m2,s2,identity)
    except ValueError:pass
    else:raise AssertionError('Invalid restore accepted')
    equal(before,capture_model(m2))
  finally:close(cb2)
  # Native file read is observational; isolate RNG changes from deserialization.
  for folder in [a,b]:
   saved_rng=capture_model(m2)
   loaded=keras.models.load_model(Path(folder)/'best_model.keras')
   assert loaded.count_params()==860364 and len(loaded.optimizer.variables)==44
   restore_model(saved_rng,m2)
  # Source-level independent semantics: checkpoint selected strict minimum
  # weighted native training loss; validation is observed but not selected.
  best_index=min(range(3),key=lambda i:records[i]['logs']['loss'])
  selected=keras.models.load_model(Path(a)/'best_model.keras');equal(selected.get_weights(),states[best_index]['model_state']['native']['weights'])
  equal([v.numpy() for v in selected.optimizer.variables],states[best_index]['model_state']['native']['optimizer'])
 rejected=0
 for bad in [x.astype('float64'),np.full_like(x,np.nan),np.zeros((5,90,567),dtype='float32')]:
  try:values(bad,5)
  except ValueError:rejected+=1
 for bad in [y.astype('float32'),np.array([0,1,12]),np.array([-1,0]),np.array([True,False])]:
  try:labels(bad,len(bad))
  except ValueError:rejected+=1
 assert rejected==7
 before=torch.get_rng_state().clone();order=permutation(5,1,7);equal(order,permutation(5,1,7));assert set(order.tolist())==set(range(5));equal(before,torch.get_rng_state())
 result={'state':'pass_synthetic_full_MLSTM_epoch_callbacks_fullstate_prediction_replay','synthetic_epochs':3,'synthetic_updates':9,'nondivisible_batch5_B2_last1':True,'weighted_training_unweighted_validation':True,'class_weights_independent_counts_checked':True,'epoch1_restore_next2_all_model_Adam_BN_metrics_callbacks_global_layer_RNG_modes_exact':True,'labelblind_predictions_and_postprediction_states_exact':True,'native_best_strict_minimum_train_loss_weights_optimizer_exact':True,'identity_settings_cursor_rejections_before_model_mutation':3,'numeric_input_label_rejections':rejected,'sampler_coverage_repeatability_no_global_RNG_draw':True,'native_fit_DataAdapter_RNG_parity_claim':False,'historical_TF_numeric_parity_claim':False,'real_UWB_values_labels_fit':False,'wall_seconds':time.monotonic()-start}
 (Path(__file__).resolve().parent/'ENGINE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
