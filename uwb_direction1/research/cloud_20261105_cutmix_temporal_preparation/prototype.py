"""Conditional transfer of released SignEval2026 TEMPORAL CutMix, not 2D boxes."""
import torch
from torch.nn import functional as F
def splice(x,index,start,length):
    """Shared temporal span over all567features (3radars x189range bins)."""
    assert x.ndim==3 and 0<=start<=x.shape[1] and 0<=length<=x.shape[1]-start
    out=x.clone();out[:,start:start+length,:]=x[index,start:start+length,:]
    return out
def batch(x,y,alpha=1.,probability=.8):
    assert x.ndim==3 and len(x)==len(y) and alpha>0 and 0<=probability<=1
    draw=float(torch.rand(1).item())
    if draw>=probability:
        return x,y,y,1.,dict(mixed=False,decision=draw,raw_lambda=None,start=None,length=0,permutation=None)
    raw_lambda=float(torch.distributions.Beta(alpha,alpha).sample().item())
    index=torch.randperm(len(x),device=x.device)
    length=int(x.shape[1]*(1-raw_lambda))
    start=int(torch.randint(0,x.shape[1]-length+1,(1,)).item()) if length<x.shape[1] else 0
    return splice(x,index,start,length),y,y[index],1-length/x.shape[1],dict(mixed=True,decision=draw,raw_lambda=raw_lambda,start=start,length=length,permutation=index)
def native_bce(logits,a,b,lam):
    target=lam*F.one_hot(a,logits.shape[-1]).to(logits.dtype)+(1-lam)*F.one_hot(b,logits.shape[-1]).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits,target)
def to_author_layout(x):
    assert x.shape[1:]==(90,567)
    return x.reshape(len(x),90,3,189).permute(0,2,1,3)
def from_author_layout(x):
    assert x.shape[1:]==(3,90,189)
    return x.permute(0,2,1,3).reshape(len(x),90,567)
