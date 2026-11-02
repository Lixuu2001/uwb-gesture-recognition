"""Native pinned author architecture plus explicit auditable loader/state handling."""
import sys,copy,ast
from pathlib import Path
SCREEN=Path('/workspace/uwb-hinception-source-probe-20261031');sys.path.insert(0,str(SCREEN))
import native_hinception as author
from native_hinception import np,torch,keras,equal
def make(output,seed=20261029):
 h,m,_=author.make(output,seed,567);h.epochs=100
 # Keras shape-only compile/optimizer build, no real forward or optimizer update.
 cb=keras.callbacks.CallbackList(h.callbacks,add_history=False,add_progbar=False,model=m,epochs=100,steps=74,verbose=0);cb.on_train_begin()
 author.capture(m)  # materialize global generator before all saved RNG instants
 return h,m,cb
def training_order(epoch,seed=20261029):
 assert 1<=epoch<=100
 return torch.randperm(4700,generator=torch.Generator().manual_seed(seed+epoch)).numpy()
def evaluation_order(count,role,seed=20261029):
 assert (count,role) in [(1000,'validation'),(1200,'diagnostic')]
 return np.random.RandomState(seed+(101 if role=='validation' else 102)).permutation(count)
def callback_state(holder):
 keys=['best','min_delta','wait','cooldown_counter','_current_epoch','_batches_seen_since_last_saving','_last_batch_seen']
 return [dict(values={k:copy.deepcopy(getattr(c,k)) for k in keys if hasattr(c,k)},monitor_initialized=c.monitor_op is not None) for c in holder.callbacks]
def snapshot(holder,model,epoch,total,batch,identity):
 assert total==int(model.optimizer.iterations.numpy())
 return dict(native=author.capture(model),callbacks=callback_state(holder),epoch=epoch,total_steps=total,sampler=dict(epoch=epoch,batch=batch,order_seed=20261029+epoch),identity=identity,settings=dict(source='cf4d0e55ee11b7393ae3d317cdf5f22dcc51ba3e',depth=holder.depth,n_filters=holder.n_filters,epochs=holder.epochs,channels=567),module_training_modes={n:bool(m.training) for n,m in model.named_modules()})
def restore(saved,holder,model,*,expected_identity):
 if saved['identity']!=expected_identity or saved['settings']!=dict(source='cf4d0e55ee11b7393ae3d317cdf5f22dcc51ba3e',depth=holder.depth,n_filters=holder.n_filters,epochs=holder.epochs,channels=567):raise ValueError('Checkpoint identity/settings mismatch')
 for c,s in zip(holder.callbacks,saved['callbacks']):
  if s['monitor_initialized'] and c.monitor_op is None:c._set_monitor_op()
  if not s['monitor_initialized']:assert c.monitor_op is None
  for k,v in s['values'].items():setattr(c,k,copy.deepcopy(v))
 author.restore(saved['native'],model)
 modes=dict(model.named_modules());assert set(modes)==set(saved['module_training_modes'])
 for n,v in saved['module_training_modes'].items():modes[n].training=v
def train_step(model,x,y):
 before=float(model._loss_tracker.total.numpy());logs=model.train_on_batch(x,y,return_dict=True)
 assert all(v.value.grad is None or torch.isfinite(v.value.grad).all() for v in model.trainable_variables)
 return {k:float(v) for k,v in logs.items()},(float(model._loss_tracker.total.numpy())-before)/len(x)
def validation(model,values,y):
 model.reset_metrics();onehot=np.eye(12,dtype='float32')[y]
 for i in range(0,len(values),64):logs=model.test_on_batch(values[i:i+64],onehot[i:i+64],return_dict=True)
 return {k:float(v) for k,v in logs.items()}
def predict(model,values):
 p=np.concatenate([model.predict_on_batch(values[i:i+64]) for i in range(0,len(values),64)])
 assert p.shape==(len(values),12) and np.isfinite(p).all();np.testing.assert_allclose(p.sum(1),1,rtol=1e-5,atol=1e-6)
 return p

_tree=ast.parse((SCREEN/'vendor/CF-4-TSC/utils/utils.py').read_bytes());_fn=next(n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name=='znormalisation');_ns=dict(np=np);exec(compile(ast.fix_missing_locations(ast.Module(body=[_fn],type_ignores=[])),'author_HCCF_znormalisation','exec'),_ns);_norm=_ns['znormalisation']
def preprocess(values):
 result=_norm(np.array(values,dtype='float32',copy=True));assert result.shape[1:]==(90,567) and np.isfinite(result).all();return result

def close(cb):
 if cb._executor is not None:cb._executor.shutdown(wait=True);cb._executor=None
