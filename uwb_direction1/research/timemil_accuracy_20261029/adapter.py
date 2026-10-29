"""Author TimeMIL core with an explicit CPU device adapter and resumable optimizer.

No project arrays, labels or training protocol are read here.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import sys
from pathlib import Path
B=Path(__file__).resolve().parent
sys.path[:0]=[str(B/'vendor/site-packages'),str(B/'vendor/TimeMIL')]
import torch
from torch import nn
from models import timemil as author

def cpu_compatible_wavelet(size,scale,shift):
    # Same arithmetic/grid as the author's function; replace hard-coded .cuda().
    x=torch.linspace(-(size[1]-1)//2,(size[1]-1)//2,size[1],device=scale.device,dtype=scale.dtype)
    x=x.reshape(1,-1).repeat(size[0],1)-shift
    c=2/(3**.5*torch.pi**.25)
    return c*(1-(x/scale)**2)*torch.exp(-(x/scale)**2/2)/(torch.abs(scale)**.5)

author.mexican_hat_wavelet=cpu_compatible_wavelet

def make_model(arm='timemil',seed=20261029):
    torch.manual_seed(seed)
    model=author.TimeMIL(in_features=567,n_classes=12,mDim=128,max_seq_len=90,dropout=.2)
    if arm=='timemil':return model
    assert arm=='gap'
    return GAP(model.feature_extractor,model._fc2)

class GAP(nn.Module):
    def __init__(self,backbone,head):
        super().__init__();self.feature_extractor=backbone;self.head=head
    def forward(self,x,warmup=False):
        return self.head(self.feature_extractor(x.transpose(1,2)).mean(-1))

class StableLookahead:
    """Author update rule, including lazy first slow buffer, with stable index keys.

    A plain wrapper avoids obsolete Optimizer internals on modern PyTorch.
    This is an engineering adapter, not a new optimizer.
    """
    def __init__(self,parameters,lr=.001,weight_decay=.0001,alpha=.5,k=6):
        self.params=list(parameters)
        self.fast=torch.optim.AdamW(self.params,lr=lr,weight_decay=weight_decay)
        self.alpha=alpha;self.k=k;self.count=0;self.slow={}
    def zero_grad(self):self.fast.zero_grad(set_to_none=True)
    @torch.no_grad()
    def step(self):
        self.fast.step();self.count+=1
        if self.count%self.k==0:
            for i,p in enumerate(self.params):
                if p.grad is None:continue
                if i not in self.slow:self.slow[i]=p.detach().clone()
                self.slow[i].add_(p-self.slow[i],alpha=self.alpha);p.copy_(self.slow[i])
    def state_dict(self):
        return dict(fast=self.fast.state_dict(),slow=self.slow,count=self.count,alpha=self.alpha,k=self.k)
    def load_state_dict(self,state):
        assert state['alpha']==self.alpha and state['k']==self.k
        self.fast.load_state_dict(state['fast']);self.count=state['count']
        self.slow={int(i):v.to(self.params[int(i)].device).clone() for i,v in state['slow'].items()}
