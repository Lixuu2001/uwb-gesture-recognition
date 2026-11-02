"""Existing group-DRO objective with log-space adversarial probabilities.
Sagawa et al., ICLR2020; not a novel algorithm. User groups are source-only.
"""
import math
import numpy as np
import torch


def balanced_order(ids,users,seed,held,epoch,per_group=16):
    ids=np.asarray(ids,dtype=np.int64);groups=np.unique(users[ids])
    if len(groups)!=4:raise ValueError('Exactly four source users required')
    pools=[ids[users[ids]==g] for g in groups]
    if any(len(p)<per_group for p in pools):raise ValueError('Insufficient source group')
    steps=math.ceil(max(map(len,pools))/per_group);columns=[]
    for g,p in zip(groups,pools):
        p=np.random.default_rng(seed+10000*held+100000*epoch+int(g)).permutation(p)
        columns.append(p[np.arange(steps*per_group)%len(p)].reshape(steps,per_group))
    return np.concatenate(columns,axis=1),groups


class GroupRisk:
    def __init__(self,n=4,eta=.01,robust=False):
        self.logq=torch.full((n,),-math.log(n),dtype=torch.float64)
        self.eta=eta;self.robust=robust
    def loss(self,losses,group):
        if losses.ndim!=1 or group.shape!=losses.shape or not torch.isfinite(losses).all():raise ValueError('Invalid losses/groups')
        if set(group.tolist())!=set(range(len(self.logq))):raise ValueError('Every source group must appear')
        means=torch.stack([losses[group==g].mean() for g in range(len(self.logq))])
        if self.robust:
            updated=self.logq+self.eta*means.detach().double().cpu()
            if not torch.isfinite(updated).all():raise ValueError('Nonfinite adversarial update')
            self.logq=updated-torch.logsumexp(updated,0)
        q=self.logq.exp().to(means)
        return (means*q).sum(),means.detach(),q.detach()
    def state(self):return dict(logq=self.logq.clone(),eta=self.eta,robust=self.robust)
    def load(self,s):
        assert s['eta']==self.eta and s['robust']==self.robust
        assert torch.isfinite(s['logq']).all() and abs(float(torch.logsumexp(s['logq'],0)))<1e-10
        self.logq=s['logq'].clone()
