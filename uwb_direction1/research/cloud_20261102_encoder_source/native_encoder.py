"""GPL-3.0 full review-author Encoder binding; original sources stay private.

Keras3/Torch mathematical compatibility, not historical TensorFlow parity.
Preserves the review's CHANNEL softmax and elementwise attention, which differ
from Serra et al.'s temporal attention / multi-dataset pretraining method.
"""
import ast,copy,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,'/workspace/uwb-mlstmfcn-source-probe-20261031')
from native_mlstm import np,torch,keras,capture,restore,equal
B=Path(__file__).resolve().parent

@keras.saving.register_keras_serializable(package='UWBReviewEncoder')
class InstanceNormalization(keras.layers.Layer):
 """Rank3 default TFA instance norm: population moments over time, eps=.001."""
 def build(self,input_shape):
  if len(input_shape)!=3 or input_shape[-1] is None:raise ValueError('Fixed rank3 channels-last required')
  self.gamma=self.add_weight(name='gamma',shape=(input_shape[-1],),initializer='ones')
  self.beta=self.add_weight(name='beta',shape=(input_shape[-1],),initializer='zeros')
  super().build(input_shape)
 def call(self,x):
  mu=keras.ops.mean(x,axis=1,keepdims=True);var=keras.ops.var(x,axis=1,keepdims=True)
  return (x-mu)*keras.ops.rsqrt(var+0.001)*self.gamma+self.beta

@keras.saving.register_keras_serializable(package='UWBReviewEncoder')
class ChannelSlice(keras.layers.Layer):
 """Serializable exact replacement for the two source channel-slice Lambdas."""
 def __init__(self,start,stop=None,**kwargs):super().__init__(**kwargs);self.start=start;self.stop=stop
 def call(self,x):return x[:,:,self.start:self.stop]
 def get_config(self):return dict(super().get_config(),start=self.start,stop=self.stop)

def check_source():
 for r in json.loads((B/'SOURCE_INDEX.json').read_bytes())['files']:
  raw=(B/'local_author_sources'/r['local_path']).read_bytes()
  assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
  assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==r['git_blob_sha1']

def bind():
 check_source();tree=ast.parse((B/'local_author_sources/review_encoder.py').read_bytes())
 cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Classifier_ENCODER')
 fn=copy.deepcopy(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='build_model'))
 class Native(ast.NodeTransformer):
  norm=0;slices=0;extension=0
  def visit_Call(self,node):
   if ast.unparse(node.func)=='tfa.layers.InstanceNormalization':
    assert not node.args and not node.keywords;node.func=ast.Name(id='InstanceNormalization',ctx=ast.Load());self.norm+=1
   elif ast.unparse(node.func)=='keras.layers.Lambda':
    assert not node.keywords and len(node.args)==1
    s=ast.unparse(node.args[0]);assert s in ['lambda x: x[:, :, :256]','lambda x: x[:, :, 256:]']
    node.func=ast.Name(id='ChannelSlice',ctx=ast.Load());node.args=[ast.Constant(0),ast.Constant(256)] if ':256]' in s else [ast.Constant(256),ast.Constant(None)];self.slices+=1
   return self.generic_visit(node)
  def visit_Constant(self,node):
   if node.value=='best_model.hdf5':node.value='best_model.keras';self.extension+=1
   return node
 t=Native();fn=t.visit(fn);assert (t.norm,t.slices,t.extension)==(4,2,1)
 ns=dict(keras=keras,InstanceNormalization=InstanceNormalization,ChannelSlice=ChannelSlice)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'full_review_Encoder_build_model_native_compatibility','exec'),ns)
 return ns['build_model']

def make(output,seed):
 if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Explicit seed required')
 keras.backend.clear_session();keras.utils.set_random_seed(seed)
 assert keras.__version__=='3.11.3' and keras.backend.backend()=='torch'
 h=SimpleNamespace(output_directory=str(output)+'/');m=bind()(h,(90,567),12)
 m._symbolic_build(data_batch=(np.zeros((1,90,567),dtype='float32'),np.zeros((1,12),dtype='float32'),None));capture(m)
 return h,m
