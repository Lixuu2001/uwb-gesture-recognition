"""Native Keras author build_model AST, with explicit runtime/artifact compatibility."""
import os
os.environ['KERAS_BACKEND']='torch'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import ast,copy,hashlib,json,random,sys
from types import SimpleNamespace
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'deps'))
import numpy as np
import torch
import keras
def source_check():
 for row in json.loads((B/'SOURCE_INDEX.json').read_text())['files']:
  assert hashlib.sha256((B/'vendor/dl4tsc'/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
def source_build():
 source_check();tree=ast.parse((B/'vendor/dl4tsc/classifiers/resnet.py').read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Classifier_RESNET');fn=copy.deepcopy(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='build_model'))
 changes=[]
 for n in ast.walk(fn):
  if isinstance(n,ast.Constant) and n.value=='best_model.hdf5':n.value='best_model.keras';changes.append('artifact extension only')
 assert changes==['artifact extension only'];ns=dict(keras=keras);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'native_author_build_model_AST_artifact_compatibility','exec'),ns);return ns['build_model']
_BUILD=source_build()
def make(output,seed=20261029):
 keras.backend.clear_session();keras.utils.set_random_seed(seed);assert keras.backend.backend()=='torch'
 holder=SimpleNamespace(output_directory=str(output)+'/');model=_BUILD(holder,(90,567),12)
 return holder,model
def step(model,x,y):
 model.reset_metrics();return model.train_on_batch(x,y,return_dict=True)
def snapshot(model):
 from keras.src.random.seed_generator import global_seed_generator
 return dict(weights=[x.copy() for x in model.get_weights()],optimizer=[x.numpy().copy() for x in model.optimizer.variables],metrics=[[v.numpy().copy() for v in m.variables] for m in model.metrics],torch_rng=torch.get_rng_state().clone(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),keras_global_rng=global_seed_generator().state.numpy().copy())
def restore(saved,model):
 from keras.src.random.seed_generator import global_seed_generator
 model.set_weights(saved['weights']);model.optimizer.build(model.trainable_variables);assert len(model.optimizer.variables)==len(saved['optimizer'])
 for v,x in zip(model.optimizer.variables,saved['optimizer']):v.assign(x)
 # Materialize metric state before restoration without a forward or optimizer update.
 for m,vals in zip(model.metrics,saved['metrics']):
  if len(m.variables)==len(vals):
   for v,x in zip(m.variables,vals):v.assign(x)
 global_seed_generator().state.assign(saved['keras_global_rng']);torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
def equal(a,b):
 if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b
