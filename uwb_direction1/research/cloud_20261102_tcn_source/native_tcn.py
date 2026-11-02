"""MIT CMU Locus Lab TCN classes unchanged; private original source pinned.

Full released sequential-classification CLI architecture, not the old axial
dilation experiment or a claim of paper-table / historical PyTorch parity.
"""
import ast,copy,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
B=Path(__file__).resolve().parent
def check_source():
 for r in json.loads((B/'SOURCE_INDEX.json').read_bytes())['files']:
  raw=(B/'local_author_sources'/r['path']).read_bytes()
  assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
  assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==r['git_blob_sha1']
def bind():
 check_source();ns=dict(torch=torch,nn=nn,F=F,weight_norm=weight_norm)
 for name,wanted in [('TCN/tcn.py',['Chomp1d','TemporalBlock','TemporalConvNet']),('TCN/mnist_pixel/model.py',['TCN'])]:
  tree=ast.parse((B/'local_author_sources'/name).read_bytes());classes=[copy.deepcopy(x) for x in tree.body if isinstance(x,ast.ClassDef)]
  assert [x.name for x in classes]==wanted
  exec(compile(ast.fix_missing_locations(ast.Module(body=classes,type_ignores=[])),'full_pinned_author_'+name,'exec'),ns)
 return ns['TCN']
def make(seed):
 if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Explicit seed required')
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 assert torch.__version__=='2.15.0+cpu' and np.__version__=='2.3.5'
 m=bind()(567,12,[25]*8,kernel_size=7,dropout=.05).cpu()
 opt=torch.optim.Adam(m.parameters(),lr=.002)
 return m,opt
def capture(m,opt):
 return copy.deepcopy(dict(model=m.state_dict(),optimizer=opt.state_dict(),modes={n:v.training for n,v in m.named_modules()},torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate()))
def restore(s,m,opt):
 modes=dict(m.named_modules());assert set(modes)==set(s['modes'])
 m.load_state_dict(s['model'],strict=True);opt.load_state_dict(s['optimizer'])
 for n,t in s['modes'].items():modes[n].training=t
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
