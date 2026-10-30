"""Unmodified selected ConvTran AST bodies; no dataset or training driver imports.

64-dimensional paper-width setting executed through pinned author code, not an
exact paper experiment: code uses 4*embedding temporal filters, ReLU/GAP and
embedding-wide attention scaling. CLI default embedding16 differs explicitly.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import ast,hashlib,json,math,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'vendor/site-packages'))
from einops import rearrange

def selected(path,names,namespace):
 nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
def definitions():
 ns=dict(torch=torch,nn=nn,F=F,math=math,rearrange=rearrange,Optimizer=Optimizer)
 root=BASE/'vendor/ConvTran'
 selected(root/'Models/AbsolutePositionalEncoding.py',('tAPE','AbsolutePositionalEncoding','LearnablePositionalEncoding'),ns)
 selected(root/'Models/Attention.py',('Attention','Attention_Rel_Scl','Attention_Rel_Vec'),ns)
 selected(root/'Models/model.py',('ConvTran',),ns)
 selected(root/'Models/loss.py',('NoFussCrossEntropyLoss',),ns)
 selected(root/'Models/optimizers.py',('RAdam',),ns)
 return ns
def make_model(seed=20261029):
 torch.manual_seed(seed);ns=definitions()
 cfg=dict(Data_shape=(16,567,90),emb_size=64,dim_ff=256,num_heads=8,Fix_pos_encode='tAPE',Rel_pos_encode='eRPE',dropout=.01)
 return ns['ConvTran'](cfg,12),ns,cfg
def optimizer(model,namespace): return namespace['RAdam'](model.parameters(),lr=.001,weight_decay=0)
def source_check():
 index=json.loads((BASE/'SOURCE_INDEX.json').read_text())
 for path,row in index['files'].items():
  data=(BASE/'vendor/ConvTran'/path).read_bytes()
  assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
  assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['git_blob_sha']
 for path,expected in index['dependencies'].items(): assert hashlib.sha256((BASE/path).read_bytes()).hexdigest()==expected,path
 return index
