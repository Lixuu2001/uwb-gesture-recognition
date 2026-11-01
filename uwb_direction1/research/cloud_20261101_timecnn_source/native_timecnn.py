"""GPL-3.0 native binding of full pinned review Time-CNN build_model AST."""
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
def bind():
 check_source();t=ast.parse((B/'local_author_sources/review_cnn.py').read_bytes());cls=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='Classifier_CNN')
 fn=copy.deepcopy(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='build_model'));changed=0
 for x in ast.walk(fn):
  if isinstance(x,ast.Constant) and x.value=='best_model.hdf5':x.value='best_model.keras';changed+=1
 assert changed==1;ns={'keras':keras};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'full_review_TimeCNN_build_model_artifact_extension_only','exec'),ns);return ns['build_model']
def make(output,seed):
 if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Explicit seed required')
 keras.backend.clear_session();keras.utils.set_random_seed(seed);assert keras.__version__=='3.11.3' and keras.backend.backend()=='torch'
 h=SimpleNamespace(output_directory=str(output)+'/');m=bind()(h,(90,567),12)
 m._symbolic_build(data_batch=(np.zeros((1,90,567),dtype='float32'),np.zeros((1,12),dtype='float32'),None));capture(m);return h,m
