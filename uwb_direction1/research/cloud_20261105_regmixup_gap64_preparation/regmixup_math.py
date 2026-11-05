"""Published RegMixup batching/loss mathematics with an explicit native OvR-BCE transfer."""
import numpy as np
import torch
from torch.nn import functional as F
def batch(x,y,alpha=20.,beta=20.):
    """Author class default/CIFAR10 recipe, one concatenated real+mixed batch."""
    lam=float(np.random.beta(alpha,beta)) if alpha>0 and beta>0 else 1.
    p=torch.randperm(len(x),device=x.device)
    mixed=lam*x+(1-lam)*x[p]
    return torch.cat([x,mixed]),torch.cat([y,y]),torch.cat([y,y[p]]),lam,p
def weighted_ce(logits,a,b,lam):
    return lam*F.cross_entropy(logits,a)+(1-lam)*F.cross_entropy(logits,b)
def native_ovr_bce_transfer(logits,a,b,lam):
    """Proposed matched-GAP loss adaptation, not author's softmax-CE benchmark."""
    t=lam*F.one_hot(a,logits.shape[1]).to(logits.dtype)+(1-lam)*F.one_hot(b,logits.shape[1]).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits,t)
