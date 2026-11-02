"""Pinned author training body; explicit multichannel/NumPy2/axis adaptation."""
import hashlib,json,random,sys
from pathlib import Path
import numpy as np
import torch
SCREEN=Path('/workspace/uwb-tstcc-screen-20261030')
sys.path.insert(0,str(SCREEN))
from author_adapter import make_models,source_check
from trainer.trainer import model_train
def make(seed=20261029,phase='ssl',pretrained=None):
 cfg,m,t,o,to=make_models(seed)
 if phase in ('linear','finetune'):
  assert pretrained is not None
  state=m.state_dict();state.update({k:v for k,v in pretrained.items() if 'logits' not in k});m.load_state_dict(state)
  if phase=='linear':
   for name,p in m.named_parameters():p.requires_grad_('logits' in name)
 assert phase in ('ssl','linear','finetune','scratch')
 scheduler=None if phase=='ssl' else torch.optim.lr_scheduler.ReduceLROnPlateau(o,'min')
 return cfg,m,t,o,to,scheduler
def step(m,t,o,to,cfg,x,y,weak=None,strong=None,phase='ssl'):
 assert len(x)==128
 if phase=='ssl':assert weak is not None and strong is not None
 else:weak=strong=x
 return model_train(m,t,o,to,torch.nn.CrossEntropyLoss(),[(x,y,weak,strong)],cfg,torch.device('cpu'),{'ssl':'self_supervised','linear':'train_linear','finetune':'fine_tune','scratch':'supervised'}[phase])
def augmented(x,seed=20261029,chunk=64):
 """Original distribution/RNG order and post-float cast, with all channels kept.

 Only changes: ragged-list indices; pat[:,warp] instead of pat[0,warp].
 Stream full-dataset fixed views to avoid full float64 temporary allocations.
 Includes original empty segments and exclusive max_seg8 ->1..7 segments.
 """
 assert x.ndim==3 and x.dtype==np.float32
 n,c,length=x.shape;r=np.random.RandomState(seed)
 factors=r.normal(loc=2.,scale=1.1,size=(n,length));weak=np.empty_like(x)
 for start in range(0,n,chunk):weak[start:start+chunk]=(x[start:start+chunk].astype(np.float64)*factors[start:start+chunk,None,:]).astype(np.float32)
 segments=r.randint(1,8,size=n);orders=np.tile(np.arange(length),(n,1))
 for i,count in enumerate(segments):
  if count>1:
   cuts=r.choice(length-2,count-1,replace=False);cuts.sort();parts=np.split(np.arange(length),cuts);order=r.permutation(len(parts));orders[i]=np.concatenate([parts[j] for j in order])
 strong=np.empty_like(x)
 for start in range(0,n,chunk):
  stop=min(start+chunk,n);permuted=np.take_along_axis(x[start:stop],orders[start:stop,None,:],axis=2)
  noise=r.normal(loc=0.,scale=.8,size=permuted.shape);strong[start:stop]=(permuted.astype(np.float64)+noise).astype(np.float32)
 return weak,strong,dict(seed=seed,factors=factors,segments=segments,orders=orders,final_numpy_rng=r.get_state())
def indices(epoch,batch,seed=20261029,n=4700):
 assert 1<=epoch<=40 and 0<=batch<36 and n==4700
 order=torch.randperm(n,generator=torch.Generator().manual_seed(seed+epoch)).numpy();return order[batch*128:(batch+1)*128]
def save_state(m,t,o,to,sched,phase,epoch,batch,total,identity):
 return dict(model=m.state_dict(),temporal=t.state_dict(),optimizer=o.state_dict(),temporal_optimizer=to.state_dict(),scheduler=None if sched is None else sched.state_dict(),phase=phase,epoch=epoch,batch=batch,total_steps=total,identity=identity,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),model_training=m.training,temporal_training=t.training)
def restore(saved,parts):
 cfg,m,t,o,to,sched=parts;m.load_state_dict(saved['model']);t.load_state_dict(saved['temporal']);o.load_state_dict(saved['optimizer']);to.load_state_dict(saved['temporal_optimizer'])
 if sched is not None:sched.load_state_dict(saved['scheduler'])
 m.train(saved['model_training']);t.train(saved['temporal_training']);torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
 return parts
def equal(a,b):
 if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and torch.equal(a,b)
 elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert type(a)==type(b) and len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b
