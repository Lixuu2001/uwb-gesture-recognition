"""Fixed diagnostic transforms; no fitting or trainable components."""
import numpy as np
import torch
from scipy.spatial.distance import cdist

def time_map(n, amplitude):
    t=np.arange(n,dtype=np.float64)
    g=t+amplitude*np.sin(2*np.pi*t/(n-1))
    g[0]=0; g[-1]=n-1
    if not np.all(np.diff(g)>0): raise ValueError('Nonmonotone time map')
    return g

def warp(x, g):
    g=torch.as_tensor(g,device=x.device,dtype=torch.float32).clamp(0,x.shape[-2]-1)
    lo=g.floor().long(); hi=(lo+1).clamp(max=x.shape[-2]-1)
    f=(g-lo).reshape(*([1]*(x.ndim-2)),-1,1)
    return (x[...,lo,:].float()*(1-f)+x[...,hi,:].float()*f).to(x.dtype)

def renorm(x):
    a,b=x.float().split(189,dim=-1)
    def norm(z): return (z-z.mean((-2,-1),keepdim=True))/(z.std((-2,-1),unbiased=False,keepdim=True)+1e-6)
    return torch.cat([norm(a),norm(b)],-1).to(x.dtype)

def dtw_map(cost, band=12):
    c=np.asarray(cost,dtype=np.float64)
    if c.ndim!=2 or c.shape[0]!=c.shape[1] or not np.isfinite(c).all():
        raise ValueError('Require finite square cost')
    n=len(c); d=np.full((n+1,n+1),np.inf); d[0,0]=0
    prev=np.zeros((n,n),dtype=np.uint8)
    for i in range(1,n+1):
        for j in range(max(1,i-band),min(n,i+band)+1):
            a,b,e=d[i-1,j-1],d[i-1,j],d[i,j-1]
            if a<=b and a<=e: v,k=a,0
            elif b<=e: v,k=b,1
            else: v,k=e,2
            d[i,j]=c[i-1,j-1]+v; prev[i-1,j-1]=k
    sums=np.zeros(n); counts=np.zeros(n); i=j=n-1
    while True:
        sums[i]+=j; counts[i]+=1
        if i==0 and j==0: break
        k=prev[i,j]
        if k==0: i-=1; j-=1
        elif k==1: i-=1
        else: j-=1
        if i<0 or j<0: raise RuntimeError('Invalid DTW path')
    return sums/counts

def distance(a,b):
    c=cdist(np.asarray(a,dtype=float),np.asarray(b,dtype=float),'sqeuclidean')
    positive=c[c>1e-12]
    return c/(np.median(positive) if len(positive) else 1)

def descriptors(x):
    # (...,90,378): two modalities, each 21 adjacent range bins.
    a=np.asarray(x,dtype=np.float32)
    return a.reshape(*a.shape[:-1],42,9).mean(-1)

def style_mix(x,donor,share):
    if share==0: return x
    if not 0<=share<=1 or x.shape!=donor.shape: raise ValueError('Bad style inputs')
    z=x.float(); b=donor.float()
    mu=z.mean(1,keepdim=True); sd=z.std(1,unbiased=False,keepdim=True).clamp_min(1e-6)
    mb=b.mean(1,keepdim=True); sb=b.std(1,unbiased=False,keepdim=True).clamp_min(1e-6)
    return (((z-mu)/sd)*(sd*(1-share)+sb*share)+mu*(1-share)+mb*share).to(x.dtype)

def partners(ids,y,subjects,seed,cross_user):
    rng=np.random.default_rng(seed); out=[]
    for idx in ids:
        keep=(y[ids]==y[idx]) & (ids!=idx)
        keep &= ((subjects[ids]!=subjects[idx]) if cross_user else (subjects[ids]==subjects[idx]))
        choices=ids[keep]
        if not len(choices): raise ValueError('No eligible partner')
        out.append(rng.choice(choices))
    return np.array(out,dtype=np.int64)
