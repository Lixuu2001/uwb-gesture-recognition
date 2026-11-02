"""Full architecture, synthetic only; native fit comparison and observer replay."""
import copy,json,tempfile,time
from source_engine import *
B=Path(__file__).resolve().parent
class Recorder(keras.callbacks.Callback):
 def __init__(self):super().__init__();self.epochs=[];self.batches=[]
 def on_epoch_end(self,epoch,logs=None):self.epochs.append(copy.deepcopy(logs))
 def on_train_batch_end(self,batch,logs=None):self.batches.append(copy.deepcopy(logs))
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();g=np.random.RandomState(37)
 x=g.randn(6,90,567).astype('float32');x[:,:,0]=3.;x[:,:,1]=0.;raw=x.copy();z=source_znormalization(x);assert np.isfinite(z).all();np.testing.assert_array_equal(x,raw)
 # Independent float64 per-sample/channel formula, including zero variance.
 ref=(x.astype('float64')-x.astype('float64').mean(1,keepdims=True));sd=x.astype('float64').std(1,keepdims=True);ref/=np.where(sd==0,1.,sd);np.testing.assert_allclose(z,ref,rtol=2e-5,atol=2e-6);np.testing.assert_array_equal(z[:,:,:2],np.zeros_like(z[:,:,:2]))
 y=np.arange(6,dtype='int64');vx=g.randn(3,90,567).astype('float32');vy=np.array([0,1,2]);ids=np.arange(6)+100;vids=np.arange(3)+500;order=np.array([5,2,1,4,0,3]);vo=np.array([2,0,1]);identity=dict(synthetic_only=True,fixture='full_author_LITEMV_engine_no_UWB')
 def create(d):return Engine(output=d,length=90,channels=567,classes=12,epochs=3,batch_size=4,seed=19,identity=identity)
 def sequence(d,observe=False,resume=False):
  e=create(d);states=[];rows=[];seen=[];first=[]
  def observer(row):
   seen.append(copy.deepcopy(row))
   if row['total_steps']==1:first.append(e.checkpoint(epoch=1,batch=0,total_steps=1,sampler=dict(order=order.copy(),next_batch=1)))
  for ep in range(1,4):
   result=e.epoch(epoch=ep,values=x,labels=y,ids=ids,order=order,validation=(vx,vy,vids,vo),observer=observer if observe else None);rows.append(result);states.append(e.checkpoint(epoch=ep,batch=1,total_steps=2*ep,sampler=dict(order=order.copy(),next_batch=0)))
   if resume and ep==1:
    torch.save(states[-1],Path(d)/'resume.pt');saved=torch.load(Path(d)/'resume.pt',weights_only=False);e.close();e=create(d);position=e.restore(saved,expected_identity=identity);equal(saved,e.checkpoint(**{k:position[k] for k in ['epoch','batch','total_steps','sampler']}))
  best=keras.models.load_model(Path(d)/'best_model.keras');weights=best.get_weights();opt=[v.numpy() for v in best.optimizer.variables];pred_ids,p=e.predict(values=vx,ids=vids,order=vo);np.testing.assert_array_equal(pred_ids,vids[vo]);assert p.shape==(3,12)
  before=e.checkpoint(epoch=3,batch=1,total_steps=6,sampler={});wrong=copy.deepcopy(before);wrong['identity']['fixture']='wrong';wrongsettings=copy.deepcopy(before);wrongsettings['engine_settings']['batch_size']=99
  for bad in [wrong,wrongsettings]:
   try:e.restore(bad,expected_identity=identity)
   except ValueError:pass
   else:raise AssertionError('Wrong identity/config should fail before mutation')
   equal(before,e.checkpoint(epoch=3,batch=1,total_steps=6,sampler={}))
  e.close();return rows,states,weights,opt,seen,first
 with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b,tempfile.TemporaryDirectory() as c,tempfile.TemporaryDirectory() as d:
  baseline=sequence(a);observed=sequence(b,observe=True);resumed=sequence(c,observe=True,resume=True)
  equal(baseline[:4],observed[:4]);equal(observed,resumed)
  assert len(observed[4])==6 and len(observed[5])==1
  for i,r in enumerate(observed[4]):np.testing.assert_array_equal(r['training_ids'],ids[order[i%2*4:(i%2+1)*4]]);assert r['total_steps']==i+1
  keras.backend.clear_session();keras.utils.set_random_seed(19)
  # Recreate after clearing/setting the same seed, as the engine constructor.
  h=author.LITEMV(output_directory=d+'/',length_TS=90,n_channels=567,n_classes=12,n_epochs=3,batch_size=4,verbose=False);m=h.model;m._symbolic_build(data_batch=(np.zeros((1,90,567),dtype='float32'),np.zeros((1,12),dtype='float32'),None));author.capture(m);rec=Recorder();m.fit(x[order],fixed_one_hot(y,12)[order],epochs=3,batch_size=4,shuffle=False,validation_data=(vx[vo],fixed_one_hot(vy,12)[vo]),callbacks=h.callbacks+[rec],verbose=0)
  for key in ['weights','optimizer','metrics']:equal(author.capture(m)[key],baseline[1][-1]['native'][key])
  for native,engine in zip(rec.epochs,baseline[0]):
   assert set(native)==set(engine['logs'])|{'learning_rate'}
   for k,v in engine['logs'].items():assert native[k]==v
   assert native['learning_rate']==engine['learning_rate']
  equal(rec.batches,[r['logs'] for ep in baseline[0] for r in ep['batches']]);equal(callback_state(h),baseline[0][-1]['callbacks']);selected=keras.models.load_model(Path(d)/'best_model.keras');equal(selected.get_weights(),baseline[2]);equal([v.numpy() for v in selected.optimizer.variables],baseline[3])
 for labels in [np.array([12]),np.array([-1]),np.array([1.])]:
  try:fixed_one_hot(labels,12)
  except ValueError:pass
  else:raise AssertionError('Bad taxonomy should fail')
 result=dict(state='pass_synthetic_engine_only',full_author_LITEMV_architecture=True,source_commit=SOURCE_COMMIT,synthetic_epochs=3,synthetic_updates=6,synthetic_batch_size=4,last_batch=2,synthetic_seed=19,synthetic_training_rows=6,synthetic_validation_rows=3,native_fit_vs_engine_weights_Adam_BN_metrics_callbackstate_selectedcheckpoint_exact=True,all_native_batch_epoch_metrics_exact=True,explicit_sample_ID_order_and_short_final_batch_verified=True,observer_vs_noobserver_all_states_RNG_exact=True,epoch1_restore_next2epochs_model_Adam_BN_metrics_callbacks_modulemodes_all_RNG_exact=True,wrong_identity_or_settings_rejected_before_state_mutation=True,unmodified_source_znormalization_vs_independent_NumPy_and_zero_variance=True,fixed12_taxonomy_validation_missing_classes=True,prediction_interface_labelblind_and_ID_order_verified=True,native_fit_DataAdapter_global_RNG_parity_asserted=False,original_TensorFlow_execution_parity_asserted=False,formal_CPU_cost_gate_run=False,real_UWB_configuration_selected=False,real_UWB_values_labels_or_fit=False,wall_seconds=time.monotonic()-start)
 (B/'ENGINE_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
