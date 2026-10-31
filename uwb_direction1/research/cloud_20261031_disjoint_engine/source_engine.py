"""GPL-3.0-only. Explicit author-weighted native Disjoint-CNN epoch/state API.

No real-data configuration/default budget/loader or automatic training launch.
Published original MIT model bytes and prior GPL state helper stay unchanged.
The caller must register actual settings and input/preprocessing first.
"""
from pathlib import Path
import sys,copy,hashlib,json
SOURCE=Path('/workspace/uwb-disjoint-source-probe-20261031')
sys.path.insert(0,str(SOURCE))
from native_disjoint import make,configure_from_source,create_class_weight,np,torch,keras,capture as native_capture,restore as native_restore,equal

def labels(y,count):
 if not isinstance(y,np.ndarray) or y.dtype.kind not in 'iu' or y.shape!=(count,) or count<1 or np.any(y<0) or np.any(y>=12):raise ValueError('Need nonempty fixed12 integer labels')
 return np.eye(12,dtype='float32')[y]

def values(x,count):
 if not isinstance(x,np.ndarray) or x.dtype!=np.float32 or x.shape!=(count,90,567,1) or count<1 or not np.isfinite(x).all():raise ValueError('Need finite float32 [N,90,567,1] inputs')

def permutation(count,epoch,seed):
 if not all(type(x) is int for x in [count,epoch,seed]) or count<1 or epoch<0 or seed<0:raise ValueError('Invalid explicit sampler settings')
 return torch.randperm(count,generator=torch.Generator().manual_seed(seed+epoch)).numpy()

def setup(output,seed,train_count,epochs,batch_size):
 if not all(type(x) is int for x in [seed,train_count,epochs,batch_size]) or seed<0 or train_count<10 or epochs<1 or batch_size<1:raise ValueError('Unset/invalid explicit source-fit settings')
 h,m=make(output,seed);mini=configure_from_source(h,train_count,epochs,batch_size)
 assert mini==int(min(train_count/10,batch_size)) and mini>0
 cb=keras.callbacks.CallbackList(h.callbacks,add_history=False,add_progbar=False,model=m,epochs=epochs,steps=(train_count+mini-1)//mini,verbose=0)
 cb.on_train_begin();native_capture(m)
 settings=dict(seed=seed,train_count=train_count,epochs=epochs,requested_batch=batch_size,actual_batch=mini,source_commit='38180ccc1f050a3fee6961f5b4ea3bc132016ab3',input_shape=[90,567,1],class_count=12)
 return h,m,cb,settings

def callback_state(h):
 keys=['best','min_delta','wait','cooldown_counter','_current_epoch','_batches_seen_since_last_saving','_last_batch_seen']
 return [dict(values={k:copy.deepcopy(getattr(c,k)) for k in keys if hasattr(c,k)},monitor_initialized=c.monitor_op is not None) for c in h.callbacks]

def snapshot(h,m,settings,identity,epoch,total_steps):
 if type(epoch) is not int or not 0<=epoch<=settings['epochs'] or type(total_steps) is not int or total_steps!=int(m.optimizer.iterations.numpy()):raise ValueError('Invalid checkpoint cursor')
 return dict(native=native_capture(m),callbacks=callback_state(h),settings=copy.deepcopy(settings),identity=copy.deepcopy(identity),epoch=epoch,total_steps=total_steps,module_training_modes={n:bool(v.training) for n,v in m.named_modules()})

def restore(s,h,m,settings,identity):
 if s['identity']!=identity or s['settings']!=settings or s['total_steps']!=int(s['native']['optimizer'][0]):raise ValueError('Checkpoint identity/settings/cursor mismatch')
 if len(s['callbacks'])!=len(h.callbacks):raise ValueError('Wrong callback topology')
 modes=dict(m.named_modules())
 if set(modes)!=set(s['module_training_modes']):raise ValueError('Wrong module topology')
 for c,state in zip(h.callbacks,s['callbacks']):
  if state['monitor_initialized'] and c.monitor_op is None:c._set_monitor_op()
  if not state['monitor_initialized'] and c.monitor_op is not None:raise ValueError('Unexpected initialized callback')
  for k,v in state['values'].items():setattr(c,k,copy.deepcopy(v))
 native_restore(s['native'],m)
 for n,v in s['module_training_modes'].items():modes[n].training=v

def epoch(h,m,cb,settings,x,y,vx,vy,epoch_index,order):
 n=settings['train_count'];values(x,n);onehot=labels(y,n);values(vx,len(vx));val_y=labels(vy,len(vx));mini=settings['actual_batch']
 if type(epoch_index) is not int or not 0<=epoch_index<settings['epochs']:raise ValueError('Epoch outside explicit budget')
 if not isinstance(order,np.ndarray) or order.dtype.kind not in 'iu' or order.shape!=(n,) or not np.array_equal(np.sort(order),np.arange(n)):raise ValueError('Training order must cover every row once')
 cw=create_class_weight(onehot);weights=np.array([cw[int(c)] for c in y],dtype='float32');cb.on_epoch_begin(epoch_index);m.reset_metrics();batch_logs=[]
 for j,i in enumerate(range(0,n,mini)):
  rows=order[i:i+mini];cb.on_train_batch_begin(j);logs=m.train_on_batch(x[rows],onehot[rows],sample_weight=weights[rows],return_dict=True)
  if not all(np.isfinite(float(v)) for v in logs.values()):raise FloatingPointError('Nonfinite native training metrics')
  if not all(v.value.grad is None or torch.isfinite(v.value.grad).all() for v in m.trainable_variables):raise FloatingPointError('Nonfinite gradients')
  cb.on_train_batch_end(j,logs);batch_logs.append({k:float(v) for k,v in logs.items()})
 logs={k:float(v) for k,v in logs.items()};m.reset_metrics()
 for i in range(0,len(vx),mini):vlogs=m.test_on_batch(vx[i:i+mini],val_y[i:i+mini],return_dict=True)
 logs.update({'val_'+k:float(v) for k,v in vlogs.items()})
 if not all(np.isfinite(v) for v in logs.values()):raise FloatingPointError('Nonfinite native validation metrics')
 cb.on_epoch_end(epoch_index,logs)
 return dict(logs=logs,batch_logs=batch_logs,order=order.copy(),class_weights=cw,last_batch_size=n%mini or mini,total_steps=int(m.optimizer.iterations.numpy()))

def predict(m,x,batch_size):
 """Label-blind; caller must freeze this partition and output before labels."""
 values(x,len(x))
 if type(batch_size) is not int or batch_size<1:raise ValueError('Unset inference batch')
 p=np.concatenate([m.predict_on_batch(x[i:i+batch_size]) for i in range(0,len(x),batch_size)])
 if p.shape!=(len(x),12) or not np.isfinite(p).all():raise FloatingPointError('Invalid native probabilities')
 np.testing.assert_allclose(p.sum(1),1,rtol=1e-5,atol=1e-6);return p

def close(cb):
 if cb._executor is not None:cb._executor.shutdown(wait=True);cb._executor=None
