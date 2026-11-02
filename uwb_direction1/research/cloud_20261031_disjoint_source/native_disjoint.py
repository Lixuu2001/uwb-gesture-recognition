"""Full default author DCNN_2L; explicit native Keras imports/artifact suffixes.

Original MIT source is retained unedited. This adapter/check uses the existing
GPL-3.0-only native-state helper and is offered under GPL-3.0-only, retaining
the MIT source notice. No TensorFlow stub/replacement Torch architecture or
width/kernel/input-literal reduction. This is not original TF execution parity.
"""
from pathlib import Path
import ast,copy,hashlib,json,sys,math,time
STATE=Path('/workspace/uwb-litemv-source-probe-20261031');sys.path.insert(0,str(STATE))
from native_litemv import np,torch,keras,capture,restore,equal
B=Path(__file__).resolve().parent
def check_source():
 for r in json.loads((B/'SOURCE_INDEX.json').read_bytes())['files']:assert hashlib.sha256((B/'vendor/Disjoint-CNN'/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
def bind_class():
 check_source();tree=ast.parse((B/'vendor/Disjoint-CNN/classifiers/DCNN_2L.py').read_bytes());cls=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Classifier_DCNN_2L'));cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['__init__','build_model','fit']];assert len(cls.body)==3
 class Bind(ast.NodeTransformer):
  renamed=[]
  def visit_Constant(self,n):
   replacements={'model_init.h5':'model_init.weights.h5','best_model.h5':'best_model.keras','model.h5':'model.keras'}
   if isinstance(n.value,str) and n.value in replacements:self.renamed.append(n.value);return ast.copy_location(ast.Constant(value=replacements[n.value]),n)
   return n
 transform=Bind();cls=transform.visit(cls);assert sorted(transform.renamed)==['best_model.h5','model.h5','model_init.h5']
 helper=ast.parse((B/'vendor/Disjoint-CNN/utils/classifier_tools.py').read_bytes());fn=copy.deepcopy(next(n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name=='create_class_weight'));ns=dict(np=np,math=math,time=time,keras=keras)
 for n in ['Input','Conv2D','BatchNormalization','ELU','Reshape','Permute','MaxPooling2D']:ns[n]=getattr(keras.layers,n)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn,cls],type_ignores=[])),'author_DCNN2L_native_layer_bindings_and_artifact_suffixes','exec'),ns);return ns['Classifier_DCNN_2L'],ns['create_class_weight'],transform.renamed
CLASS,create_class_weight,ARTIFACT_SUFFIX_CHANGES=bind_class()
def configure_from_source(holder,train_count,epochs,batch_size):
 """Original fit prefix through self.callbacks; no training or label access."""
 from types import SimpleNamespace
 tree=ast.parse((B/'vendor/Disjoint-CNN/classifiers/DCNN_2L.py').read_bytes());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='fit'));prefix=[]
 for node in fn.body:
  prefix.append(node)
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr=='callbacks' for t in node.targets):break
 else:raise AssertionError('No original callback setup found')
 fn.name='_original_fit_setup';fn.body=prefix+[ast.Return(value=ast.Name(id='mini_batch_size',ctx=ast.Load()))]
 class Suffix(ast.NodeTransformer):
  def visit_Constant(self,n):
   return ast.copy_location(ast.Constant(value='best_model.keras'),n) if n.value=='best_model.h5' else n
 fn=Suffix().visit(fn);ns=dict(keras=keras);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'original_DCNN2L_fit_setup_with_suffix_only','exec'),ns);mini=ns['_original_fit_setup'](holder,SimpleNamespace(shape=(train_count,90,567,1)),None,None,None,epochs,batch_size);assert mini>0
 holder.model._symbolic_build(data_batch=(np.zeros((1,90,567,1),dtype='float32'),np.zeros((1,12),dtype='float32'),None));capture(holder.model);return mini
def make(output,seed):
 keras.backend.clear_session();keras.utils.set_random_seed(seed);assert keras.__version__=='3.11.3' and keras.backend.backend()=='torch';h=CLASS(output_directory=str(output)+'/',input_shape=(90,567,1),nb_classes=12,verbose=False);m=h.model;m.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(),metrics=['accuracy']);m._symbolic_build(data_batch=(np.zeros((1,90,567,1),dtype='float32'),np.zeros((1,12),dtype='float32'),None));capture(m);return h,m
