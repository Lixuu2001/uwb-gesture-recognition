"""Pinned author LITEMV/native Keras primitives; no real-data entry or defaults.

GPL-3.0-only, as the retained author source. Source architecture is executed by
the previous explicit native binding, not reimplemented or reduced. Epoch,
batch, seed, shape, preprocessing and identities must be supplied by a future
registered caller. This module never loads UWB arrays or labels.
"""
from pathlib import Path
import ast,copy,json,hashlib,sys
SCREEN=Path('/workspace/uwb-litemv-source-probe-20261031');sys.path.insert(0,str(SCREEN))
import native_litemv as author
from native_litemv import np,torch,keras,equal
SOURCE_COMMIT='e62f64393104be8eda415b4d974987028ca68754'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_dependencies():
 runtime=json.loads((SCREEN/'dependencies/RUNTIME.json').read_bytes())
 if runtime['python']!=sys.version or runtime['numpy']!=np.__version__ or runtime['torch']!=torch.__version__ or runtime['keras']!=keras.__version__ or keras.backend.backend()!='torch':raise RuntimeError('Different base Python/NumPy/Torch/native Keras backend')
 for n,h in json.loads((SCREEN/'PREPARATION_HASHES.json').read_bytes()).items():
  if sha(SCREEN/n)!=h:raise RuntimeError('Changed source preparation: '+n)
 for n,h in json.loads((SCREEN/'dependencies/DEPENDENCIES.json').read_bytes())['files'].items():
  if sha(author.DEPS/n)!=h:raise RuntimeError('Changed isolated runtime: '+n)
 return True
def bind_normalization():
 author.check_source();tree=ast.parse((SCREEN/'vendor/LITE/src/utils/utils.py').read_bytes());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='znormalisation');ns=dict(np=np);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'pinned_author_znormalisation_unmodified','exec'),ns);return ns['znormalisation']
source_znormalization=bind_normalization()
def fixed_one_hot(labels,classes):
 labels=np.asarray(labels)
 if labels.dtype.kind not in 'iu' or labels.ndim!=1 or classes!=12 or (labels.size and (labels.min()<0 or labels.max()>=classes)):raise ValueError('Requires fixed 12-class task taxonomy and integer class IDs')
 return np.eye(classes,dtype='float32')[labels]
def callback_state(holder):
 keys=['best','min_delta','wait','cooldown_counter','_current_epoch','_batches_seen_since_last_saving','_last_batch_seen']
 return [dict(class_name=type(c).__name__,values={k:copy.deepcopy(getattr(c,k)) for k in keys if hasattr(c,k)},monitor_initialized=c.monitor_op is not None) for c in holder.callbacks]
class Engine:
 def __init__(self,*,output,length,channels,classes,epochs,batch_size,seed,identity):
  for n in [length,channels,classes,epochs,batch_size,seed]:
   if type(n) is not int:raise TypeError('Explicit integer settings required')
  if min(length,channels,classes,epochs,batch_size)<=0 or classes!=12:raise ValueError('Invalid dimensions/budget/fixed task taxonomy')
  if not isinstance(identity,dict) or not identity:raise ValueError('Explicit nonempty identity required')
  verify_dependencies();keras.backend.clear_session();keras.utils.set_random_seed(seed)
  self.identity=copy.deepcopy(identity);self.batch_size=batch_size;self.epochs=epochs;self.classes=classes;self.length=length;self.channels=channels;self.seed=seed;self.settings=dict(length=length,channels=channels,classes=classes,epochs=epochs,batch_size=batch_size,seed=seed,source_commit=SOURCE_COMMIT,source_preparation_manifest_sha256=sha(SCREEN/'PREPARATION_HASHES.json'),dependency_manifest_sha256=sha(SCREEN/'dependencies/DEPENDENCIES.json'))
  self.holder=author.LITEMV(output_directory=str(output)+'/',length_TS=length,n_channels=channels,n_classes=classes,n_epochs=epochs,batch_size=batch_size,verbose=False)
  self.model=self.holder.model;self.model._symbolic_build(data_batch=(np.zeros((1,length,channels),dtype='float32'),np.zeros((1,classes),dtype='float32'),None))
  self.callbacks=keras.callbacks.CallbackList(self.holder.callbacks,add_history=False,add_progbar=False,model=self.model,epochs=epochs,verbose=0);self.callbacks.on_train_begin();author.capture(self.model)
 def checkpoint(self,*,epoch,batch,total_steps,sampler):
  if total_steps!=int(self.model.optimizer.iterations.numpy()):raise ValueError('Checkpoint position/Adam iteration mismatch')
  return copy.deepcopy(dict(identity=self.identity,engine_settings=self.settings,native=author.capture(self.model),callbacks=callback_state(self.holder),module_training_modes={n:bool(m.training) for n,m in self.model.named_modules()},position=dict(epoch=epoch,batch=batch,total_steps=total_steps,sampler=sampler)))
 def restore(self,saved,*,expected_identity):
  # Identity mismatch must reject before changing any model/callback/RNG state.
  if expected_identity!=self.identity or saved.get('identity')!=expected_identity:raise ValueError('Checkpoint identity mismatch')
  if saved.get('engine_settings')!=self.settings:raise ValueError('Checkpoint engine settings/source/runtime mismatch')
  states=saved['callbacks']
  if len(states)!=len(self.holder.callbacks) or any(s['class_name']!=type(c).__name__ for c,s in zip(self.holder.callbacks,states)):raise ValueError('Checkpoint callback structure mismatch')
  modes=dict(self.model.named_modules())
  if set(modes)!=set(saved['module_training_modes']):raise ValueError('Checkpoint module structure mismatch')
  for c,s in zip(self.holder.callbacks,states):
   if s['monitor_initialized'] and c.monitor_op is None:c._set_monitor_op()
   if not s['monitor_initialized'] and c.monitor_op is not None:raise ValueError('Restore requires fresh callback monitor')
   for k,v in s['values'].items():setattr(c,k,copy.deepcopy(v))
  author.restore(saved['native'],self.model)
  for n,training in saved['module_training_modes'].items():modes[n].training=training
  return copy.deepcopy(saved['position'])
 def epoch(self,*,epoch,values,labels,ids,order,validation=None,observer=None):
  if not 1<=epoch<=self.epochs:raise ValueError('Epoch outside explicit budget')
  values=np.asarray(values);labels=np.asarray(labels);ids=np.asarray(ids);order=np.asarray(order)
  if values.shape!=(len(labels),self.length,self.channels) or ids.ndim!=1 or len(ids)!=len(labels) or len(np.unique(ids))!=len(ids):raise ValueError('Training shape/identity mismatch')
  if order.dtype.kind not in 'iu' or order.ndim!=1 or not np.array_equal(np.sort(order),np.arange(len(ids))) or not len(order):raise ValueError('Explicit permutation of all training rows required')
  if not np.isfinite(values).all():raise ValueError('Nonfinite training input')
  if validation is not None and np.intersect1d(ids,np.asarray(validation[2])).size:raise ValueError('Training/validation sample ID overlap')
  if validation is not None:self.validate_evaluation(*validation)
  y=fixed_one_hot(labels,self.classes);self.callbacks.on_epoch_begin(epoch-1);self.model.reset_metrics();batches=[]
  for batch,lo in enumerate(range(0,len(order),self.batch_size)):
   ix=order[lo:lo+self.batch_size];self.callbacks.on_train_batch_begin(batch);logs=self.model.train_on_batch(values[ix],y[ix],return_dict=True)
   if not all(np.isfinite(float(v)) for v in logs.values()):raise FloatingPointError('Nonfinite native train metrics')
   if not all(v.value.grad is None or torch.isfinite(v.value.grad).all() for v in self.model.trainable_variables):raise FloatingPointError('Nonfinite native train gradient')
   self.callbacks.on_train_batch_end(batch,logs);row=dict(epoch=epoch,batch=batch,total_steps=int(self.model.optimizer.iterations.numpy()),training_ids=ids[ix].copy(),logs={k:float(v) for k,v in logs.items()});batches.append(copy.deepcopy(row))
   if observer is not None:observer(copy.deepcopy(row))
  logs={k:float(v) for k,v in logs.items()}
  if validation is not None:
   vx,vy,vids,vorder=validation;vlogs=self.evaluate(values=vx,labels=vy,ids=vids,order=vorder);logs.update({'val_'+k:v for k,v in vlogs.items()})
  self.callbacks.on_epoch_end(epoch-1,logs)
  # ReduceLR injects learning_rate into CallbackList's COPY of the logs. Expose
  # the actual current LR separately; no claim it mutates this input dictionary.
  return dict(epoch=epoch,training_records=len(ids),batches=batches,logs=logs,learning_rate=float(self.model.optimizer.learning_rate.numpy()),callbacks=callback_state(self.holder))
 def evaluate(self,*,values,labels,ids,order):
  self.validate_evaluation(values,labels,ids,order)
  values=np.asarray(values);labels=np.asarray(labels);ids=np.asarray(ids);order=np.asarray(order)
  y=fixed_one_hot(labels,self.classes);self.model.reset_metrics()
  for lo in range(0,len(order),self.batch_size):
   ix=order[lo:lo+self.batch_size];logs=self.model.test_on_batch(values[ix],y[ix],return_dict=True)
  return {k:float(v) for k,v in logs.items()}
 def validate_evaluation(self,values,labels,ids,order):
  values=np.asarray(values);labels=np.asarray(labels);ids=np.asarray(ids);order=np.asarray(order)
  if values.shape!=(len(labels),self.length,self.channels) or ids.ndim!=1 or len(ids)!=len(labels) or len(np.unique(ids))!=len(ids) or order.dtype.kind not in 'iu' or order.ndim!=1 or not np.array_equal(np.sort(order),np.arange(len(ids))) or not len(ids):raise ValueError('Evaluation shape/order/identity mismatch')
  if not np.isfinite(values).all():raise ValueError('Nonfinite evaluation input')
  fixed_one_hot(labels,self.classes)
 def predict(self,*,values,ids,order):
  values=np.asarray(values);ids=np.asarray(ids);order=np.asarray(order)
  if values.shape!=(len(ids),self.length,self.channels) or ids.ndim!=1 or len(np.unique(ids))!=len(ids) or order.dtype.kind not in 'iu' or order.ndim!=1 or not np.array_equal(np.sort(order),np.arange(len(ids))) or not len(ids):raise ValueError('Prediction shape/order/identity mismatch')
  if not np.isfinite(values).all():raise ValueError('Nonfinite prediction input')
  probabilities=np.concatenate([self.model.predict_on_batch(values[order[lo:lo+self.batch_size]]) for lo in range(0,len(order),self.batch_size)])
  if probabilities.shape!=(len(ids),self.classes) or not np.isfinite(probabilities).all():raise FloatingPointError('Invalid predictions')
  np.testing.assert_allclose(probabilities.sum(1),1,rtol=1e-5,atol=1e-6);return ids[order].copy(),probabilities
 def close(self):
  self.callbacks.on_train_end()
  if self.callbacks._executor is not None:self.callbacks._executor.shutdown(wait=True);self.callbacks._executor=None
