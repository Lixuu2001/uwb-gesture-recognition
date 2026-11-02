"""Known MC-dropout inference with the existing model's learned dropout rates.

Gal & Ghahramani ICML2016 section4: average stochastic predictive outputs.
The reused TimeMIL was trained with OvR BCE, so preserve its sigmoid link;
these scores are not claimed to be a calibrated categorical distribution.
"""
import contextlib,copy
import numpy as np
import torch
from torch import nn
@contextlib.contextmanager
def dropout_only(model):
 modules=dict(model.named_modules());before={n:m.training for n,m in modules.items()}
 try:
  model.eval();active=[]
  for n,m in modules.items():
   if isinstance(m,nn.Dropout) and m.p>0:m.train();active.append(dict(name=n,p=m.p,inplace=m.inplace))
  if not active:raise ValueError('No learned positive-rate dropout')
  assert not any(m.training for m in modules.values() if isinstance(m,nn.modules.batchnorm._BatchNorm))
  yield active
 finally:
  for n,m in modules.items():m.training=before[n]
def infer(model,x,samples,seed,batch_size,after_sample=None):
 if type(samples) is not int or samples<1 or type(seed) is not int or not 0<=seed<2**32 or type(batch_size) is not int or batch_size<1:raise ValueError('Explicit MC count/seed/batch required')
 if not isinstance(x,np.ndarray) or x.dtype!=np.float32 or x.ndim!=3 or len(x)<1 or not np.isfinite(x).all():raise ValueError('Finite float32 sequences required')
 before=copy.deepcopy(model.state_dict());logits=np.empty((samples,len(x),12),dtype='float32');rng=[]
 with torch.random.fork_rng(devices=[],enabled=True),dropout_only(model) as active,torch.no_grad():
  torch.manual_seed(seed)
  for t in range(samples):
   for i in range(0,len(x),batch_size):
    z=model(torch.from_numpy(x[i:i+batch_size]))
    if z.shape!=(min(batch_size,len(x)-i),12) or not torch.isfinite(z).all():raise FloatingPointError('Invalid MC logits')
    logits[t,i:i+len(z)]=z.numpy()
   rng.append(torch.get_rng_state().clone())
   if after_sample is not None:after_sample(t+1,logits[t],rng[-1])
 linked=torch.sigmoid(torch.from_numpy(logits)).numpy();mean=linked.mean(axis=0,dtype='float64')
 for n,v in model.state_dict().items():assert torch.equal(v,before[n]),'Inference changed '+n
 return dict(logits=logits,mean_sigmoid=mean,predictions=mean.argmax(1),torch_rng_after_each_sample=rng,dropout_modules=active)
