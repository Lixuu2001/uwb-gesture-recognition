"""Full official ResNet18 ImageNet1K_V1 frozen encoder + fixed linear SVM.

Exact original class/function AST,not a reduced network. torchvision extension
is unnecessary for these pure torch classes;only API-usage logging is stubbed.
This is a published transfer baseline,not the radar papers' fine-tuning protocol
or a claimed original contribution. No UWB labels are accessed by this module.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='4'
from pathlib import Path
import ast,hashlib,json,sys
from typing import Any,Callable,Optional,Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from sklearn.svm import LinearSVC
B=Path(__file__).resolve().parent
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
_log_api_usage_once=lambda *a,**k:None
_raw=(B/'local_author_sources/resnet.py').read_bytes()
assert hashlib.sha1(b'blob '+str(len(_raw)).encode()+b'\0'+_raw).hexdigest()=='47067ec83175a97cc6f6a8721b342128a434d440'
_names=['conv3x3','conv1x1','BasicBlock','Bottleneck','ResNet'];_tree=ast.parse(_raw.decode());_nodes=[n for n in _tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in _names];assert [n.name for n in _nodes]==_names
exec(compile(ast.Module(body=_nodes,type_ignores=[]),str(B/'local_author_sources/resnet.py'),'exec'),globals())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4194304),b''):h.update(block)
 return h.hexdigest()
def make(keep_original_head=False):
 meta=json.loads((B/'WEIGHTS_SOURCE.json').read_bytes());p=B/'weights/resnet18-f37072fd.pth';assert p.stat().st_size==meta['bytes'] and sha(p)==meta['sha256']
 torch.manual_seed(20261029);m=ResNet(BasicBlock,[2,2,2,2]);assert sum(p.numel() for p in m.parameters())==11689512
 state=torch.load(p,map_location='cpu',weights_only=True);m.load_state_dict(state,strict=True)
 if not keep_original_head:m.fc=nn.Identity()
 m.requires_grad_(False);m.eval();assert not any(p.requires_grad for p in m.parameters())
 return m
def preprocess(x):
 assert x.ndim==4 and tuple(x.shape[1:])==(3,90,189)
 t=torch.as_tensor(np.ascontiguousarray(x),dtype=torch.float32);assert torch.isfinite(t).all()
 lo=t.amin(dim=(-2,-1),keepdim=True);hi=t.amax(dim=(-2,-1),keepdim=True);t=(t-lo)/(hi-lo).clamp_min(1e-8)
 t=F.interpolate(t,size=(224,224),mode='bilinear',align_corners=False,antialias=False)
 mean=torch.tensor([.485,.456,.406]).reshape(1,3,1,1);std=torch.tensor([.229,.224,.225]).reshape(1,3,1,1)
 return (t-mean)/std
def extract(m,x,batch_size=16):
 assert not m.training and not any(p.requires_grad for p in m.parameters());features=[];hashes=[]
 with torch.inference_mode():
  for start in range(0,len(x),batch_size):
   z=m(preprocess(x[start:start+batch_size])).cpu().numpy();assert z.shape==(len(z),512) and np.isfinite(z).all();features.append(z);hashes.append(hashlib.sha256(z.tobytes()).hexdigest())
 return np.concatenate(features),hashes
def make_head():
 return LinearSVC(C=1.0,penalty='l2',loss='squared_hinge',dual=False,tol=1e-4,max_iter=10000,class_weight=None,fit_intercept=True,intercept_scaling=1,random_state=20261029)
def encoder_digest(m):
 h=hashlib.sha256()
 for n,t in m.state_dict().items():h.update(n.encode());h.update(str((tuple(t.shape),str(t.dtype))).encode());h.update(t.detach().contiguous().cpu().numpy().tobytes())
 return h.hexdigest()
