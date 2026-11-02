"""Pinned MIT TSLib DLinear classification, unmodified model/decomposition AST.

Forecasting paper is the architectural source; classification is an explicit
library migration baseline, not a claim of paper classification reproduction.
"""
import ast,copy,hashlib,json,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn,optim
from torch.nn import functional as F
B=Path(__file__).resolve().parent
def check_source():
 for r in json.loads((B/'SOURCE_INDEX.json').read_bytes())['files']:
  raw=(B/'local_author_sources'/r['path']).read_bytes();assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
  assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==r['git_blob_sha1']
def module(path):return ast.parse((B/'local_author_sources'/path).read_bytes())
def cli_defaults():
 check_source();wanted=['train_epochs','batch_size','patience','learning_rate','moving_avg'];found={}
 for n in ast.walk(module('run.py')):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant):
   key=str(n.args[0].value).removeprefix('--')
   if key in wanted:found[key]=ast.literal_eval(next(k.value for k in n.keywords if k.arg=='default'))
 assert found==dict(train_epochs=10,batch_size=32,patience=3,learning_rate=.0001,moving_avg=25)
 return found
def bind():
 cli_defaults();ns=dict(torch=torch,nn=nn,F=F,np=np,optim=optim)
 for path,names in [('layers/Autoformer_EncDec.py',['moving_avg','series_decomp']),('models/DLinear.py',['Model']),('utils/tools.py',['EarlyStopping'])]:
  nodes=[copy.deepcopy(n) for n in module(path).body if isinstance(n,ast.ClassDef) and n.name in names];assert [n.name for n in nodes]==names
  exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'unchanged_TSLib_'+path,'exec'),ns)
 exp=next(n for n in module('exp/exp_classification.py').body if isinstance(n,ast.ClassDef) and n.name=='Exp_Classification')
 fn=copy.deepcopy(next(n for n in exp.body if isinstance(n,ast.FunctionDef) and n.name=='_select_optimizer'))
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'unchanged_TSLib_classification_optimizer','exec'),ns)
 clips=[n for n in ast.walk(exp) if isinstance(n,ast.Call) and ast.unparse(n.func)=='nn.utils.clip_grad_norm_'];assert len(clips)==1 and len(clips[0].keywords)==1 and clips[0].keywords[0].arg=='max_norm' and ast.literal_eval(clips[0].keywords[0].value)==4.
 return ns['Model'],ns['EarlyStopping'],ns['_select_optimizer']
def make(seed):
 if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Explicit seed required')
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);assert torch.__version__=='2.15.0+cpu' and np.__version__=='2.3.5'
 cls,early,optimizer=bind();c=SimpleNamespace(task_name='classification',seq_len=90,pred_len=0,enc_in=567,num_class=12,moving_avg=25,learning_rate=.0001)
 m=cls(c).cpu();opt=optimizer(SimpleNamespace(model=m,args=c));assert isinstance(opt,torch.optim.RAdam)
 return m,opt,early
def forward(m,x):return m(x,None,None,None)
def step(m,opt,x,y):
 opt.zero_grad();out=forward(m,x);loss=F.cross_entropy(out,y)
 if not torch.isfinite(loss):raise FloatingPointError('Nonfinite CE')
 loss.backward();norm=nn.utils.clip_grad_norm_(m.parameters(),max_norm=4.)
 if not torch.isfinite(norm) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()):raise FloatingPointError('Nonfinite gradients')
 opt.step();return out.detach(),float(loss.detach()),float(norm)
def capture(m,opt):return copy.deepcopy(dict(model=m.state_dict(),optimizer=opt.state_dict(),modes={n:v.training for n,v in m.named_modules()},torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate()))
def restore(s,m,opt):
 m.load_state_dict(s['model'],strict=True);opt.load_state_dict(s['optimizer']);mods=dict(m.named_modules());assert set(mods)==set(s['modes'])
 for n,v in s['modes'].items():mods[n].training=v
 torch.set_rng_state(s['torch_rng']);np.random.set_state(s['numpy_rng']);random.setstate(s['python_rng'])
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
