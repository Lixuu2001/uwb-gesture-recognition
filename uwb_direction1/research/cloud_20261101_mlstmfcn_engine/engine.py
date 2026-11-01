"""Full native MLSTM epoch/callback/state API, no UWB loaders or run defaults.

GPL-3.0-only. Native helper imports source AST without changing original files.
The manual dedicated sampler is a disclosed protocol adaptation, not a claim
of old TF or native .fit DataAdapter RNG/shuffle equivalence.
"""
from pathlib import Path
from types import SimpleNamespace
import copy,sys
SOURCE=Path('/workspace/uwb-mlstmfcn-source-probe-20261031')
sys.path.insert(0,str(SOURCE))
from native_mlstm import make,np,torch,keras,capture as capture_model,restore as restore_model,equal

def labels(y,n):
 if not isinstance(y,np.ndarray) or y.dtype.kind not in 'iu' or y.shape!=(n,) or n<1 or np.any((y<0)|(y>=12)):raise ValueError('Need fixed12 nonempty integer labels')
 return np.eye(12,dtype='float32')[y]
def values(x,n):
 if not isinstance(x,np.ndarray) or x.dtype!=np.float32 or x.shape!=(n,567,90) or n<1 or not np.isfinite(x).all():raise ValueError('Need finite float32[N,567,90]')
def class_weights(y):
 labels(y,len(y));present,counts=np.unique(y,return_counts=True)
 return {int(c):len(y)/(len(present)*int(n)) for c,n in zip(present,counts)}
def permutation(n,epoch,seed):
 if not all(type(x) is int for x in [n,epoch,seed]) or n<1 or epoch<0 or not 0<=seed<2**32:raise ValueError('Invalid explicit sampler settings')
 return torch.randperm(n,generator=torch.Generator().manual_seed(seed+epoch)).numpy()
def setup(out,seed,train_count,epochs,batch_size):
 if not all(type(x) is int for x in [seed,train_count,epochs,batch_size]) or train_count<1 or epochs<1 or batch_size<1:raise ValueError('Explicit fit settings required')
 m=make(seed)
 # Source helper callback order, loss monitor, patience/factor/minlr retained.
 # Full native checkpoint replaces weights-only filename for audit evidence;
 # custom checkpoints, not native files alone, retain complete RNG state.
 callbacks=[keras.callbacks.ModelCheckpoint(str(Path(out)/'best_model.keras'),monitor='loss',mode='min',save_best_only=True,verbose=0),keras.callbacks.ReduceLROnPlateau(monitor='loss',mode='min',patience=100,factor=float(1/np.cbrt(2)),cooldown=0,min_lr=1e-4,verbose=0)]
 h=SimpleNamespace(callbacks=callbacks)
 cb=keras.callbacks.CallbackList(callbacks,add_history=False,add_progbar=False,model=m,epochs=epochs,steps=(train_count+batch_size-1)//batch_size,verbose=0)
 cb.on_train_begin();capture_model(m)
 settings={'seed':seed,'train_count':train_count,'epochs':epochs,'batch_size':batch_size,'input_shape':[567,90],'class_count':12,'native_gate':'sigmoid','source_model_blob':'6d425bda53e84b0c9d6c620a016f29b903807ae6','dedicated_sampler':'TorchGenerator(seed+zero_based_epoch); no global RNG draw'}
 return h,m,cb,settings
def callback_state(h):
 keys=['best','min_delta','wait','cooldown_counter','_current_epoch','_batches_seen_since_last_saving','_last_batch_seen']
 return [{'values':{k:copy.deepcopy(getattr(c,k)) for k in keys if hasattr(c,k)},'monitor_initialized':c.monitor_op is not None} for c in h.callbacks]
def snapshot(h,m,settings,identity,epoch,total_steps,batch=-1):
 if type(epoch) is not int or not 0<=epoch<=settings['epochs'] or type(total_steps) is not int or total_steps!=int(m.optimizer.iterations.numpy()):raise ValueError('Invalid snapshot cursor')
 return {'model_state':capture_model(m),'callbacks':callback_state(h),'settings':copy.deepcopy(settings),'identity':copy.deepcopy(identity),'epoch':epoch,'batch':batch,'total_steps':total_steps}
def restore(s,h,m,settings,identity):
 if s['settings']!=settings or s['identity']!=identity or s['total_steps']!=int(s['model_state']['native']['optimizer'][0]):raise ValueError('Checkpoint identity/settings/cursor mismatch')
 if len(s['callbacks'])!=len(h.callbacks):raise ValueError('Wrong callback topology')
 for c,state in zip(h.callbacks,s['callbacks']):
  if state['monitor_initialized'] and c.monitor_op is None:c._set_monitor_op()
  if not state['monitor_initialized'] and c.monitor_op is not None:raise ValueError('Unexpected initialized callback')
  for k,v in state['values'].items():setattr(c,k,copy.deepcopy(v))
 restore_model(s['model_state'],m)
def epoch(h,m,cb,settings,x,y,vx,vy,index,order,after_batch=None):
 n=settings['train_count'];values(x,n);hot=labels(y,n);values(vx,len(vx));vhot=labels(vy,len(vx));batch=settings['batch_size']
 if type(index) is not int or not 0<=index<settings['epochs']:raise ValueError('Epoch outside explicit budget')
 if not isinstance(order,np.ndarray) or order.dtype.kind not in 'iu' or order.shape!=(n,) or not np.array_equal(np.sort(order),np.arange(n)):raise ValueError('Order must cover each training row exactly once')
 cw=class_weights(y);weights=np.array([cw[int(c)] for c in y],dtype='float32');cb.on_epoch_begin(index);m.reset_metrics();records=[]
 for j,i in enumerate(range(0,n,batch)):
  rows=order[i:i+batch];cb.on_train_batch_begin(j)
  tracker=m.metrics[0];old_total=float(tracker.total.numpy());old_count=float(tracker.count.numpy())
  logs=m.train_on_batch(x[rows],hot[rows],sample_weight=weights[rows],return_dict=True)
  logs={k:float(v) for k,v in logs.items()}
  if not all(np.isfinite(v) for v in logs.values()):raise FloatingPointError('Nonfinite training metrics')
  if not all(v.value.grad is None or torch.isfinite(v.value.grad).all() for v in m.trainable_variables):raise FloatingPointError('Nonfinite gradients')
  delta=float(tracker.count.numpy())-old_count;assert delta==len(rows)
  batch_loss=(float(tracker.total.numpy())-old_total)/delta
  if not np.isfinite(batch_loss):raise FloatingPointError('Nonfinite batch loss')
  cb.on_train_batch_end(j,logs)
  record={'batch':j,'indices':rows.copy(),'batch_size':len(rows),'total_steps':int(m.optimizer.iterations.numpy()),'native_cumulative_metrics':logs,'batch_loss_from_tracker_delta':batch_loss}
  records.append(record)
  if after_batch is not None:after_batch(record)
 epochlogs=dict(logs);m.reset_metrics()
 for i in range(0,len(vx),batch):vlogs=m.test_on_batch(vx[i:i+batch],vhot[i:i+batch],return_dict=True)
 epochlogs.update({'val_'+k:float(v) for k,v in vlogs.items()})
 if not all(np.isfinite(v) for v in epochlogs.values()):raise FloatingPointError('Nonfinite validation metrics')
 previous=h.callbacks[0].best;improved=previous is None or epochlogs['loss']<float(previous)
 cb.on_epoch_end(index,epochlogs)
 return {'logs':epochlogs,'batch_records':records,'order':order.copy(),'class_weights':cw,'improved':improved,'total_steps':int(m.optimizer.iterations.numpy()),'last_batch_size':n%batch or batch}
def predict(m,x,batch_size):
 values(x,len(x))
 if type(batch_size) is not int or batch_size<1:raise ValueError('Explicit inference batch required')
 p=np.concatenate([m.predict_on_batch(x[i:i+batch_size]) for i in range(0,len(x),batch_size)])
 if p.shape!=(len(x),12) or not np.isfinite(p).all():raise FloatingPointError('Invalid probabilities')
 np.testing.assert_allclose(p.sum(1),1,rtol=1e-5,atol=1e-6);return p
def close(cb):
 if cb._executor is not None:cb._executor.shutdown(wait=True);cb._executor=None
