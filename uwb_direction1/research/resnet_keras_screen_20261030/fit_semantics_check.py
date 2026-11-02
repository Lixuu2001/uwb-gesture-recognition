"""Synthetic native fit versus explicit batch loop, including short final batch.

The loop is an auditable loader adaptation, not TensorFlow fit/RNG reproduction.
Native train/test primitives, sample-weighted epoch metrics and original callbacks
are checked against one uninterrupted current-Keras fit call with fixed order.
"""
import copy,json,tempfile,time
from author_adapter import *
class Recorder(keras.callbacks.Callback):
 def __init__(self):super().__init__();self.epochs=[];self.batches=[]
 def on_train_batch_end(self,batch,logs=None):self.batches.append({k:float(v) for k,v in logs.items()})
 def on_epoch_end(self,epoch,logs=None):self.epochs.append({k:float(v) for k,v in logs.items()})
def callbacks(holder,model,rec):
 cb=keras.callbacks.CallbackList(holder.callbacks+[rec],add_history=False,add_progbar=False,model=model,epochs=3,steps=2,verbose=0)
 return cb
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();rng=np.random.RandomState(31)
 x=rng.randn(70,90,567).astype('float32');y=np.eye(12,dtype='float32')[np.arange(70)%12];vx=rng.randn(17,90,567).astype('float32');vy=np.eye(12,dtype='float32')[np.arange(17)%12]
 with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
  h,m=make(a);r=Recorder();m.fit(x,y,batch_size=64,epochs=3,shuffle=False,validation_data=(vx,vy),callbacks=h.callbacks+[r],verbose=0);native=copy.deepcopy(snapshot(m));native_best=keras.models.load_model(Path(a)/'best_model.keras')
  h2,m2=make(b);r2=Recorder();cb=callbacks(h2,m2,r2);cb.on_train_begin()
  for ep in range(3):
   cb.on_epoch_begin(ep);m2.reset_metrics()
   for j,lo in enumerate(range(0,70,64)):
    cb.on_train_batch_begin(j);logs=m2.train_on_batch(x[lo:lo+64],y[lo:lo+64],return_dict=True);cb.on_train_batch_end(j,logs)
   logs={k:float(v) for k,v in logs.items()};m2.reset_metrics();val=m2.test_on_batch(vx,vy,return_dict=True);logs.update({'val_'+k:float(v) for k,v in val.items()});cb.on_epoch_end(ep,logs)
  cb.on_train_end();explicit=copy.deepcopy(snapshot(m2));explicit_best=keras.models.load_model(Path(b)/'best_model.keras')
  for key in ['weights','optimizer','metrics']:equal(native[key],explicit[key])
  equal(r.epochs,r2.epochs);equal(r.batches,r2.batches);equal(native_best.get_weights(),explicit_best.get_weights());equal([v.numpy() for v in native_best.optimizer.variables],[v.numpy() for v in explicit_best.optimizer.variables])
  assert h.callbacks[0].best==h2.callbacks[0].best and h.callbacks[0].wait==h2.callbacks[0].wait and h.callbacks[1].best==h2.callbacks[1].best
  # Independent constant-loss plateau expectation, no optimizer/model updates.
  s=h2.callbacks[0];s.on_train_begin();m2.optimizer.learning_rate.assign(.001);rates=[]
  for ep in range(251):
   s.on_epoch_end(ep,{'loss':.1});expected=max(.0001,.001*.5**(ep//50));actual=float(m2.optimizer.learning_rate.numpy());assert abs(actual-expected)<1e-9,(ep,actual,expected);rates.append(actual)
  result=dict(state='pass',synthetic_epochs=3,synthetic_updates=6,batch_size=64,last_batch=6,full_native_architecture=True,native_fit_vs_explicit_native_batches_weights_optimizer_BN_metrics_exact=True,all_epoch_and_batch_metrics_exact=True,original_training_loss_checkpoint_weights_optimizer_exact=True,source_ReduceLROnPlateau_251_constant_loss_epochs_matches_independent_rule=True,native_fit_dataadapter_global_RNG_parity_asserted=False,explicit_loader_adaptation_required=True,original_TensorFlow_reproduction=False,real_UWB_values_labels_or_fit=False,wall_seconds=time.monotonic()-start)
 (B/'FIT_SEMANTICS_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
