"""Full author H-Inception width/depth, explicit multivariate input adaptation.

GPL-3.0-only. Unedited vendor source retained. Five architecture methods use
native Keras binding, a single input-channel literal adaptation, and .keras
checkpoint suffix. No smaller model or TensorFlow stub. Not original TF parity
or an unmodified multivariate paper reproduction (paper uses univariate UCR).
"""
from pathlib import Path
import ast,copy,hashlib,json,sys
SCREEN=Path('/workspace/uwb-litemv-source-probe-20261031');sys.path.insert(0,str(SCREEN))
from native_litemv import np,torch,keras,capture,restore,equal
B=Path(__file__).resolve().parent
def check_source():
 for r in json.loads((B/'SOURCE_INDEX.json').read_bytes())['files']:assert hashlib.sha256((B/'vendor/CF-4-TSC'/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
def bind_class(channels):
 assert channels in (1,567);check_source();tree=ast.parse((B/'vendor/CF-4-TSC/classifiers/H_Inception.py').read_bytes());original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HINCEPTION');cls=copy.deepcopy(original);names=['__init__','hybrid_layer','_inception_module','_shortcut_layer','build_model'];cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(cls.body)==5
 class Bind(ast.NodeTransformer):
  keras_bindings=0;input_adaptations=0;suffix_adaptations=0
  def visit_Attribute(self,n):
   if isinstance(n.value,ast.Name) and n.value.id=='tf' and n.attr=='keras':self.keras_bindings+=1;return ast.copy_location(ast.Name(id='keras',ctx=ast.Load()),n)
   return self.generic_visit(n)
  def visit_Assign(self,n):
   if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='input_shape':
    assert isinstance(n.value,ast.Tuple) and isinstance(n.value.elts[1],ast.Constant) and n.value.elts[1].value==1;n.value.elts[1]=ast.copy_location(ast.Constant(value=channels),n.value.elts[1]);self.input_adaptations+=1
   return self.generic_visit(n)
  def visit_Constant(self,n):
   if n.value=='best_model.hdf5':self.suffix_adaptations+=1;return ast.copy_location(ast.Constant(value='best_model.keras'),n)
   return n
 trans=Bind();cls=trans.visit(cls);assert trans.input_adaptations==trans.suffix_adaptations==1 and not any(isinstance(n,ast.Name) and n.id=='tf' for n in ast.walk(cls));ns=dict(np=np,keras=keras);exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'author_HINCEPTION_explicit_native_and_input_binding','exec'),ns);return ns['HINCEPTION'],dict(native_keras_bindings=trans.keras_bindings,input_channel_literal_adaptations=trans.input_adaptations,artifact_suffix_adaptations=trans.suffix_adaptations)
def make(output,seed,channels):
 keras.backend.clear_session();keras.utils.set_random_seed(seed);assert keras.__version__=='3.11.3' and keras.backend.backend()=='torch';cls,changes=bind_class(channels);h=cls(output_directory=str(output)+'/',length_TS=90,n_classes=12);m=h.model;m._symbolic_build(data_batch=(np.zeros((1,90,channels),dtype='float32'),np.zeros((1,12),dtype='float32'),None));capture(m);return h,m,changes
