"""Synthetic full LITEMV: native fit semantics and explicit epoch-state replay.

Fixture B4/3epochs/seed7 is engineering only, not a UWB configuration. No
TensorFlow parity claim. Global native DataAdapter RNG is not reproduced by
the explicit loop. Full explicit-loop state, including callbacks, is restored.
"""
import copy,json,tempfile,time
from native_hinception import *
import native_hinception as native_source
def make(output,seed):
 h,m,_=native_source.make(output,seed,567);return h,m
class Recorder(keras.callbacks.Callback):
 def __init__(self):super().__init__();self.epochs=[];self.batches=[]
 def on_train_batch_end(self,batch,logs=None):self.batches.append({k:float(v) for k,v in logs.items()})
 def on_epoch_end(self,epoch,logs=None):self.epochs.append({k:float(v) for k,v in logs.items()})
def callbacks_state(holder):
 keys=['best','min_delta','wait','cooldown_counter','_current_epoch','_batches_seen_since_last_saving','_last_batch_seen']
 return [dict(values={k:copy.deepcopy(getattr(c,k)) for k in keys if hasattr(c,k)},monitor_initialized=c.monitor_op is not None) for c in holder.callbacks]
def state(holder,model):return dict(native=capture(model),callbacks=callbacks_state(holder))
def restore_state(saved,holder,model):
 for c,s in zip(holder.callbacks,saved['callbacks']):
  if s['monitor_initialized'] and c.monitor_op is None:c._set_monitor_op()
  if not s['monitor_initialized']:assert c.monitor_op is None
  for k,v in s['values'].items():setattr(c,k,copy.deepcopy(v))
 restore(saved['native'],model)
def close(cb):
 if cb._executor is not None:cb._executor.shutdown(wait=True);cb._executor=None
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();rng=np.random.RandomState(31)
 x=rng.randn(6,90,567).astype('float32');y=np.eye(12,dtype='float32')[np.arange(6)];vx=rng.randn(3,90,567).astype('float32');vy=np.eye(12,dtype='float32')[[0,1,2]]
 def explicit(d,resume=False):
  def build():
   h,m=make(d,7);rec=Recorder();cb=keras.callbacks.CallbackList(h.callbacks+[rec],add_history=False,add_progbar=False,model=m,epochs=3,steps=2,verbose=0);cb.on_train_begin();return h,m,rec,cb
  h,m,rec,cb=build();epoch_logs=[];batch_logs=[];states=[]
  for ep in range(3):
   cb.on_epoch_begin(ep);m.reset_metrics()
   for j,lo in enumerate(range(0,6,4)):
    cb.on_train_batch_begin(j);logs=m.train_on_batch(x[lo:lo+4],y[lo:lo+4],return_dict=True);cb.on_train_batch_end(j,logs);batch_logs.append(copy.deepcopy(logs))
   logs={k:float(v) for k,v in logs.items()};m.reset_metrics();val=m.test_on_batch(vx,vy,return_dict=True);logs.update({'val_'+k:float(v) for k,v in val.items()});cb.on_epoch_end(ep,logs);epoch_logs.append(copy.deepcopy(rec.epochs[-1]));states.append(state(h,m))
   if resume and ep==0:
    torch.save(states[-1],Path(d)/'epoch.pt');saved=torch.load(Path(d)/'epoch.pt',weights_only=False);close(cb);h,m,rec,cb=build();restore_state(saved,h,m);equal(saved,state(h,m))
  cb.on_train_end();close(cb);best=keras.models.load_model(Path(d)/'best_model.keras');return h,m,epoch_logs,batch_logs,states,best
 with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b,tempfile.TemporaryDirectory() as c:
  h,m=make(a,7);rec=Recorder();m.fit(x,y,batch_size=4,epochs=3,shuffle=False,validation_data=(vx,vy),callbacks=h.callbacks+[rec],verbose=0);native=capture(m);native_best=keras.models.load_model(Path(a)/'best_model.keras')
  h2,m2,epochs,batches,states,best=explicit(b);h3,m3,e3,b3,s3,best3=explicit(c,True)
  for k in ['weights','optimizer','metrics']:equal(native[k],capture(m2)[k])
  equal(rec.epochs,epochs);equal(rec.batches,batches);equal(native_best.get_weights(),best.get_weights());equal([v.numpy() for v in native_best.optimizer.variables],[v.numpy() for v in best.optimizer.variables]);equal(callbacks_state(h),callbacks_state(h2))
  equal(epochs,e3);equal(batches,b3);equal(states,s3);equal(best.get_weights(),best3.get_weights());equal([v.numpy() for v in best.optimizer.variables],[v.numpy() for v in best3.optimizer.variables])
  # Independent plateau schedule, including floor. Does not update the model.
  s=h2.callbacks[0];s.on_train_begin();m2.optimizer.learning_rate.assign(.001)
  for ep in range(251):
   s.on_epoch_end(ep,{'loss':.1});expected=max(.0001,.001*.5**(ep//50));assert abs(float(m2.optimizer.learning_rate.numpy())-expected)<1e-9
  # Strict train-loss minimum, including tied loss and opposing validation loss.
  h4,m4=make(Path(c)/'selection',7);Path(h4.output_directory).mkdir();cb=keras.callbacks.CallbackList(h4.callbacks,add_history=False,add_progbar=False,model=m4,epochs=4,steps=1,verbose=0);cb.on_train_begin();saved_hash=None;head=m4.layers[-1].bias
  for ep,(loss,vl) in enumerate([(3.,1.),(2.,2.),(2.,.1),(2.5,.01)]):
   head.assign(np.full(head.shape,ep,dtype='float32'));cb.on_epoch_end(ep,{'loss':loss,'val_loss':vl});p=Path(h4.output_directory)/'best_model.keras';current=hashlib.sha256(p.read_bytes()).hexdigest()
   if ep==1:saved_hash=current
   if ep>1:assert current==saved_hash
  cb.on_train_end();close(cb);selected=keras.models.load_model(p);np.testing.assert_array_equal(selected.layers[-1].bias.numpy(),np.ones(12,dtype='float32'))
 result=dict(state='pass_synthetic_native_fit_and_callback_replay_only',full_author_HInception_width_depth_multivariate_input_adaptation=True,synthetic_epochs=3,synthetic_updates=6,synthetic_batch_size=4,last_batch=2,synthetic_seed=7,native_fit_vs_explicit_native_batches_weights_optimizer_BN_metrics_exact=True,all_epoch_and_batch_metrics_exact=True,original_trainloss_checkpoint_weights_optimizer_exact=True,explicit_restore_next_two_epochs_model_Adam_BN_metrics_callbacks_all_RNG_exact=True,plateau_251_constant_loss_epochs_independent_rule_exact=True,checkpoint_trainloss_strictmin_tiesfirst_and_validation_ignored=True,fixed12_label_mapping_handles_missing_validation_classes=True,native_fit_DataAdapter_global_RNG_parity_asserted=False,original_TensorFlow_execution_parity_asserted=False,formal_CPU_cost_gate_run=False,real_UWB_configuration_selected=False,real_UWB_values_labels_or_fit=False,wall_seconds=time.monotonic()-start)
 (B/'FIT_SEMANTICS_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
