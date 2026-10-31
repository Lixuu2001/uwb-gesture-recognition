"""Full author LITEMV AST, explicit tf.keras -> native Keras binding only.

No TensorFlow stub, no fake module, no replacement Torch architecture. This is
native Keras3/Torch compatibility preparation, not original TF execution parity.
GPL-3.0-only source and change notice accompany the extracted architecture.
"""
import os
os.environ['KERAS_BACKEND']='torch'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import ast,copy,json,hashlib,sys,random
B=Path(__file__).resolve().parent;DEPS=Path('/workspace/uwb-resnet-keras-screen-20261030/deps');sys.path.insert(0,str(DEPS))
import numpy as np
import torch
import keras
def check_source():
 for r in json.loads((B/'SOURCE_INDEX.json').read_text())['files']:assert hashlib.sha256((B/'vendor/LITE'/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
def bind_class():
 check_source();tree=ast.parse((B/'vendor/LITE/src/classifiers/litemv.py').read_text());original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LITEMV');methods=['__init__','hybird_layer','_inception_module','_fcn_module','build_model'];cls=copy.deepcopy(original);cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in methods];assert len(cls.body)==5
 class Bind(ast.NodeTransformer):
  count=0
  def visit_Attribute(self,node):
   if isinstance(node.value,ast.Name) and node.value.id=='tf' and node.attr=='keras':self.count+=1;return ast.copy_location(ast.Name(id='keras',ctx=ast.Load()),node)
   return self.generic_visit(node)
 t=Bind();cls=t.visit(cls);assert not any(isinstance(n,ast.Name) and n.id=='tf' for n in ast.walk(cls));ns=dict(np=np,keras=keras);exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'author_LITEMV_AST_only_native_keras_binding','exec'),ns);return ns['LITEMV'],t.count
LITEMV,NATIVE_BINDINGS=bind_class()
def make(output,seed):
 keras.backend.clear_session();keras.utils.set_random_seed(seed);assert keras.__version__=='3.11.3' and keras.backend.backend()=='torch'
 h=LITEMV(output_directory=str(output)+'/',length_TS=90,n_channels=567,n_classes=12,verbose=False)
 m=h.model;m._symbolic_build(data_batch=(np.zeros((1,90,567),dtype='float32'),np.zeros((1,12),dtype='float32'),None));capture(m)
 return h,m
def capture(model):
 from keras.src.random.seed_generator import global_seed_generator
 kg=global_seed_generator().state.numpy().copy()
 return copy.deepcopy(dict(weights=model.get_weights(),optimizer=[v.numpy().copy() for v in model.optimizer.variables],metrics=[[v.numpy().copy() for v in metric.variables] for metric in model.metrics],torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),keras_global_rng=kg))
def restore(saved,model):
 from keras.src.random.seed_generator import global_seed_generator
 model.set_weights(saved['weights']);model.optimizer.build(model.trainable_variables);assert len(model.optimizer.variables)==len(saved['optimizer'])
 for v,x in zip(model.optimizer.variables,saved['optimizer']):v.assign(x)
 assert len(model.metrics)==len(saved['metrics'])
 for m,xs in zip(model.metrics,saved['metrics']):
  assert len(m.variables)==len(xs)
  for v,x in zip(m.variables,xs):v.assign(x)
 global_seed_generator().state.assign(saved['keras_global_rng']);torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
def equal(a,b):
 if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b
