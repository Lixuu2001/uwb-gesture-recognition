"""Original weighted fit versus explicit native steps, synthetic order fixed.

No actual UWB data/config/cost. Only native fit shuffling is disabled by a local
fixture wrapper to expose the same known order; original author fit AST stays
unchanged apart from named artifact suffixes. Full DataAdapter RNG parity is
not claimed; explicit-loop checkpoint/RNG recovery is checked separately.
"""
import copy,json,tempfile,time
from native_disjoint import *
class Recorder(keras.callbacks.Callback):
 def __init__(self):super().__init__();self.epochs=[];self.batches=[]
 def on_epoch_end(self,epoch,logs=None):self.epochs.append(copy.deepcopy(logs))
 def on_train_batch_end(self,batch,logs=None):self.batches.append(copy.deepcopy(logs))
def cs(h):
 keys=['best','min_delta','wait','cooldown_counter','_current_epoch','_batches_seen_since_last_saving','_last_batch_seen'];return [dict(values={k:copy.deepcopy(getattr(c,k)) for k in keys if hasattr(c,k)},monitor_initialized=c.monitor_op is not None) for c in h.callbacks]
def snap(h,m):return dict(native=capture(m),callbacks=cs(h))
def recover(s,h,m):
 for c,v in zip(h.callbacks,s['callbacks']):
  if v['monitor_initialized'] and c.monitor_op is None:c._set_monitor_op()
  for k,x in v['values'].items():setattr(c,k,copy.deepcopy(x))
 restore(s['native'],m)
def close(cb):
 if cb._executor is not None:cb._executor.shutdown(wait=True);cb._executor=None
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();g=np.random.RandomState(73);x=g.randn(25,90,567,1).astype('float32');y=np.eye(12,dtype='float32')[np.arange(25)%12];vx=g.randn(3,90,567,1).astype('float32');vy=np.eye(12,dtype='float32')[[0,1,2]];cw=create_class_weight(y);weights=np.array([cw[int(c)] for c in y.argmax(1)],dtype='float32')
 def explicit(d,resume=False):
  def build():
   h,m=make(d,7);mini=configure_from_source(h,25,2,8);assert mini==2;r=Recorder();cb=keras.callbacks.CallbackList(h.callbacks+[r],add_history=False,add_progbar=False,model=m,epochs=2,steps=13,verbose=0);cb.on_train_begin();return h,m,r,cb
  h,m,r,cb=build();ep_logs=[];batch_logs=[];states=[]
  for ep in range(2):
   cb.on_epoch_begin(ep);m.reset_metrics()
   for j,i in enumerate(range(0,25,2)):
    cb.on_train_batch_begin(j);logs=m.train_on_batch(x[i:i+2],y[i:i+2],sample_weight=weights[i:i+2],return_dict=True);cb.on_train_batch_end(j,logs);batch_logs.append(copy.deepcopy(r.batches[-1]))
   logs={k:float(v) for k,v in logs.items()};m.reset_metrics()
   for i in range(0,3,2):val=m.test_on_batch(vx[i:i+2],vy[i:i+2],return_dict=True)
   logs.update({'val_'+k:float(v) for k,v in val.items()});cb.on_epoch_end(ep,logs);ep_logs.append(copy.deepcopy(r.epochs[-1]));states.append(snap(h,m))
   if resume and ep==0:
    torch.save(states[-1],Path(d)/'resume.pt');saved=torch.load(Path(d)/'resume.pt',weights_only=False);close(cb);h,m,r,cb=build();recover(saved,h,m);equal(saved,snap(h,m))
  cb.on_train_end();close(cb);best=keras.models.load_model(Path(d)/'best_model.keras');return h,m,ep_logs,batch_logs,states,best
 with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b,tempfile.TemporaryDirectory() as c:
  h,m=make(a,7);rec=Recorder();original=m.fit
  def fixed_order_fit(*args,**kw):kw['shuffle']=False;kw['callbacks']=kw['callbacks']+[rec];return original(*args,**kw)
  m.fit=fixed_order_fit;h.fit(x,y,vx,vy,2,8);native=capture(m);native_cb=cs(h);native_best=keras.models.load_model(Path(a)/'best_model.keras');native_last=keras.models.load_model(Path(a)/'model.keras')
  h2,m2,e2,b2,s2,best2=explicit(b);h3,m3,e3,b3,s3,best3=explicit(c,True)
  for k in ['weights','optimizer','metrics']:equal(native[k],capture(m2)[k])
  equal(native_cb,cs(h2));equal(rec.epochs,e2);equal(rec.batches,b2);equal(e2,e3);equal(b2,b3);equal(s2,s3);equal(native_best.get_weights(),best2.get_weights());equal([v.numpy() for v in native_best.optimizer.variables],[v.numpy() for v in best2.optimizer.variables]);equal(best2.get_weights(),best3.get_weights());equal([v.numpy() for v in best2.optimizer.variables],[v.numpy() for v in best3.optimizer.variables]);equal(native_last.get_weights(),native['weights']);equal([v.numpy() for v in native_last.optimizer.variables],native['optimizer'])
  # Independent scheduler follows val_loss; opposing train loss cannot select.
  reduce=h2.callbacks[0];reduce.on_train_begin();m2.optimizer.learning_rate.assign(.001)
  for ep in range(251):
   reduce.on_epoch_end(ep,{'loss':1000-ep,'val_loss':.1});expected=max(.0001,.001*.5**(ep//50));assert abs(float(m2.optimizer.learning_rate.numpy())-expected)<1e-9
 result=dict(state='pass_synthetic_original_weighted_fit_and_callback_replay_only',full_default_author_DCNN2L_architecture=True,original_fit_body_invoked_with_only_fixture_shuffle_false_override=True,synthetic_train_count=25,synthetic_validation_count=3,source_requested_batch8_actual_minibatch2=True,last_train_batch1=True,synthetic_epochs=2,synthetic_updates=26,native_original_class_weight_fit_vs_explicit_sample_weight_all_model_Adam_BN_metrics_exact=True,all_weighted_native_batch_epoch_metrics_and_callback_states_exact=True,validation_unweighted_and_val_loss_selection_preserved=True,native_selected_and_final_keras_weights_optimizer_exact=True,epoch1_restore_next_epoch_model_Adam_BN_metrics_callbacks_all_RNG_exact=True,independent_val_loss_plateau251_steps_ignores_opposing_trainloss=True,source_class_weights_not_normalized_to_unit_mean=True,native_DataAdapter_global_RNG_parity_asserted=False,original_TensorFlow_execution_parity_asserted=False,formal_CPU_cost_gate_run=False,real_UWB_configuration_selected=False,real_UWB_values_labels_or_fit=False,wall_seconds=time.monotonic()-start)
 (B/'FIT_SEMANTICS_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
