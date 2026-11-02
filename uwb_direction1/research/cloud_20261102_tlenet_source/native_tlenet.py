"""GPL-3.0 complete review t-LeNet, source-policy/exact streaming adapter.

Preserves full1000/B256, metadata-triggered warp[1]/slice.9 and ten windows.
Native Keras3 optimizer schedule maps legacy Adam(lr,decay) explicitly; no TF
parity claim. Original source private; original fit's per-parent vote retained.
"""
import ast,copy,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,'/workspace/uwb-mlstmfcn-source-probe-20261031')
from native_mlstm import np,torch,keras,capture,restore,equal
B=Path(__file__).resolve().parent
def check_source():
 for r in json.loads((B/'SOURCE_INDEX.json').read_bytes())['files']:
  raw=(B/'local_author_sources'/r['path']).read_bytes();assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
  assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==r['git_blob_sha1']
def original_class():
 check_source();t=ast.parse((B/'local_author_sources/review_tlenet.py').read_bytes());return next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='Classifier_TLENET')
def choose_policy(h,ntrain,ntest,length):
 cls=original_class();fit=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='fit')
 decisions=[copy.deepcopy(x) for x in fit.body if isinstance(x,ast.If) and 'x_train.shape' in ast.unparse(x.test)];assert len(decisions)==2
 ns=dict(self=h,x_train=SimpleNamespace(shape=(ntrain,length,567)),x_test=SimpleNamespace(shape=(ntest,length,567)))
 exec(compile(ast.fix_missing_locations(ast.Module(body=decisions,type_ignores=[])),'unchanged_review_tlenet_metadata_policy','exec'),ns)
 return dict(warping_ratios=list(h.warping_ratios),slice_ratio=h.slice_ratio,length_sliced=int(h.slice_ratio*length),windows_per_parent=length-int(h.slice_ratio*length)+1)
def bind():
 cls=copy.deepcopy(original_class());wanted=['__init__','slice_data','window_warping','build_model','pre_processing'];cls.body=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in wanted];assert len(cls.body)==5
 class Compat(ast.NodeTransformer):
  adam=0;ext=0
  def visit_Call(self,node):
   if ast.unparse(node.func)=='keras.optimizers.Adam':
    assert not node.args and {k.arg:ast.literal_eval(k.value) for k in node.keywords}==dict(lr=.01,decay=.005)
    node.keywords=[ast.keyword(arg='learning_rate',value=ast.Call(func=ast.Name(id='InverseTimeDecay',ctx=ast.Load()),args=[ast.Constant(.01),ast.Constant(1),ast.Constant(.005)],keywords=[]))];self.adam+=1
   return self.generic_visit(node)
  def visit_Constant(self,node):
   if node.value=='best_model.hdf5':node.value='best_model.keras';self.ext+=1
   return node
 t=Compat();cls=t.visit(cls);assert (t.adam,t.ext)==(1,1);ns=dict(keras=keras,np=np,InverseTimeDecay=keras.optimizers.schedules.InverseTimeDecay)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'full_review_tlenet_native_compat','exec'),ns);return ns['Classifier_TLENET']
def make(output,seed):
 if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Explicit seed required')
 keras.backend.clear_session();keras.utils.set_random_seed(seed);assert keras.__version__=='3.11.3' and keras.backend.backend()=='torch'
 h=bind()(str(output)+'/',False);policy=choose_policy(h,4700,1000,90);assert policy==dict(warping_ratios=[1],slice_ratio=.9,length_sliced=81,windows_per_parent=10)
 m=h.build_model((81,567),12);m._symbolic_build(data_batch=(np.zeros((1,81,567),dtype='float32'),np.zeros((1,12),dtype='float32'),None));capture(m);return h,m
def windows(x,augmented_rows):
 rows=np.asarray(augmented_rows)
 if not isinstance(x,np.ndarray) or x.dtype!=np.float32 or x.ndim!=3 or x.shape[1:]!=(90,567) or rows.dtype.kind not in 'iu' or rows.ndim!=1 or len(rows)<1 or np.any((rows<0)|(rows>=len(x)*10)):raise ValueError('Wrong source shape/window indices')
 parents=rows//10;starts=rows%10;return np.array(x[parents[:,None],starts[:,None]+np.arange(81),:],dtype='float32',copy=True,order='C')
def vote(crop_probabilities):
 p=np.asarray(crop_probabilities)
 if p.ndim!=3 or p.shape[1:]!=(10,12) or len(p)<1 or not np.isfinite(p).all():raise ValueError('Need parent x10x12 finite probabilities')
 decisions=p.argmax(2)
 # Equivalent to source fit's np.unique(sorted labels) + argmax(count),
 # including the smallest-class tie rule. Broken standalone predict() is not used.
 return np.asarray([np.bincount(r,minlength=12).argmax() for r in decisions],dtype='int64')
