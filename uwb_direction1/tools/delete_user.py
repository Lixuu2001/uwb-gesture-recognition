"""Prespecified source-user deletion and matched random-row stress tests."""
import numpy as np


def subsets(ids,y,u,held,seed):
    result={}
    for deleted in sorted(np.unique(u)):
        domain=ids[u!=deleted];drop=[]
        for k in range(12):
            pool=ids[y==k];n=int(((u==deleted)&(y==k)).sum());rng=np.random.default_rng(np.random.SeedSequence([seed,int(held),int(deleted),k]));drop.extend(rng.choice(pool,n,replace=False))
        random=np.setdiff1d(ids,drop);result[f'domain_{deleted}']=domain;result[f'row_{deleted}']=random
    return result


def aggregate(scores):
    out={}
    for group in ['domain','row']:
        a=np.array([v for k,v in sorted(scores.items()) if k.startswith(group+'_')]);assert len(a)==4
        out[group+'_min']=a.min(0);out[group+'_mean']=a.mean(0)
    return out
