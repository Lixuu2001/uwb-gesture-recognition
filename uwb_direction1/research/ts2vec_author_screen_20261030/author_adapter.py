"""Pinned TS2Vec imports/defaults; no dataset loader or real fitting launcher.

Original model/loss/training source remains unchanged. CPU and seed are explicit
task adaptations. The helper full-context step is used only for a conservative
synthetic cost profile, not a replacement real-data training protocol.
"""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import hashlib,json,random,sys
from pathlib import Path
import numpy as np
import torch
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'vendor/ts2vec'))
from ts2vec import TS2Vec
from models.losses import hierarchical_contrastive_loss,instance_contrastive_loss,temporal_contrastive_loss
def source_check():
 index=json.loads((BASE/'SOURCE_INDEX.json').read_text())
 for n,row in index['files'].items():
  data=(BASE/'vendor/ts2vec'/n).read_bytes();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],n
  assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['git_blob_sha'],n
 return index
def make_model(seed=20261029):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 return TS2Vec(input_dims=567,device='cpu')
def full_context_update(model,opt,data):
 # This is the author's crop_l=90 case, preserving independent masks/dropout.
 model._net.train();opt.zero_grad();z1=model._net(data.clone());z2=model._net(data.clone())
 loss=hierarchical_contrastive_loss(z1,z2,temporal_unit=model.temporal_unit);assert torch.isfinite(loss);loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model._net.parameters())
 opt.step();model.net.update_parameters(model._net);return float(loss.detach())
