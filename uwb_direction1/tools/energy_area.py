"""Within-radar range-energy path areas (known level-two log signatures).

Independent mathematical implementation; no physical trajectory claim.
"""
import numpy as np


def energy_path(raw):
    x=np.asarray(raw,dtype=np.float64)
    if x.shape[-3:]!=(3,90,189) or not np.isfinite(x).all():
        raise ValueError('Expected finite [...,3,90,189] RF')
    p=np.sqrt(np.mean(x.reshape(*x.shape[:-1],7,27)**2,axis=-1))
    scale=p.max(axis=(-2,-1),keepdims=True)
    return np.divide(p,scale,out=np.zeros_like(p),where=scale>0)


def areas(path):
    p=np.asarray(path,dtype=np.float64);q=p-p[...,:1,:];d=np.diff(q,axis=-2)
    integral=np.einsum('...ti,...tj->...ij',q[...,:-1,:],d)
    i,j=np.triu_indices(p.shape[-1],1)
    return .5*(integral[...,i,j]-integral[...,j,i])


def describe(path,ids,seed):
    p=np.asarray(path,dtype=np.float64);n=len(p)
    base=np.concatenate([p.mean(-2),p.std(-2),p[...,-1,:]-p[...,0,:]],-1).reshape(n,-1)
    signed=areas(p).reshape(n,-1);shuffled=[]
    for row,identity in zip(p,ids):
        rng=np.random.default_rng(np.random.SeedSequence([seed,int(identity)]))
        # Same deterministic order in all views, endpoints fixed; no class access.
        order=np.r_[0,rng.permutation(np.arange(1,89)),89]
        shuffled.append(areas(row[:,order,:]).ravel())
    return dict(base=base,signed=signed,unsigned=abs(signed),shuffled=np.asarray(shuffled))


def source_roles(y,u,held):
    valid=np.isin(u,np.arange(3,9))&~((u==6)&(y==3))
    val=held+1 if held<8 else 3
    return {name:np.flatnonzero(valid&mask) for name,mask in
            [('train',(u!=held)&(u!=val)),('val',u==val),('diagnostic',u==held)]}


def arm_features(hidden,desc):
    base=np.concatenate([hidden,desc['base']],1)
    return {'cnn_readout':hidden,'base':base,**{k:np.concatenate([base,desc[k]],1) for k in ['signed','unsigned','shuffled']}}


def decision(scores,c):
    out={};candidate=np.asarray(scores['signed'])
    for arm,gain in c['gate']['minimum_mean_gain'].items():
        d=candidate-np.asarray(scores[arm])
        out[arm]={'mean_gain':float(d.mean()),'wins':int((d>1e-12).sum()),'minimum_paired_gain':float(d.min()),
                  'passed':bool(d.mean()>=gain and (d>1e-12).sum()>=c['gate']['minimum_wins'] and d.min()>=-c['gate']['maximum_paired_decline'])}
    return out
