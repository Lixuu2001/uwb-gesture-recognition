"""Train-side empty-echo subspace selectivity; no classification or novelty claim."""
import numpy as np


def normalize(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.sqrt(np.mean(x*x,axis=(-2,-1),keepdims=True)),1e-12)


def moments(raw,ids,batch):
    result=np.zeros((3,189,189))
    for st in range(0,len(ids),batch):
        x=normalize(raw[ids[st:st+batch]])
        for v in range(3):
            z=x[:,v].reshape(-1,189);result[v]+=z.T@z
    return result/(len(ids)*90)


def projectors(cov,held,c):
    output={}
    for v in range(3):
        e,u=np.linalg.eigh(cov[v]);e=e[::-1];u=u[:,::-1]
        assert e.min()>-1e-9
        rank=int(np.searchsorted(np.cumsum(np.maximum(e,0)),c['energy_fraction']*np.maximum(e,0).sum())+1)
        basis=u[:,:rank];random,_=np.linalg.qr(np.random.default_rng(np.random.SeedSequence([c['seed'],held,v])).normal(size=(189,rank)))
        output[f'v{v}_eigenvalues']=e;output[f'v{v}_basis']=basis;output[f'v{v}_random_basis']=random
    return output


def fractions(raw,ids,model,batch):
    actual=[];random=[]
    for st in range(0,len(ids),batch):
        x=normalize(raw[ids[st:st+batch]]);den=np.sum(x*x,axis=(-2,-1));aa=[];bb=[]
        assert (den>0).all()
        for v in range(3):
            for key,out in [('basis',aa),('random_basis',bb)]:
                p=x[:,v]@model[f'v{v}_{key}'];out.append(1-np.sum(p*p,axis=(-2,-1))/den[:,v])
        actual.append(np.stack(aa,1));random.append(np.stack(bb,1))
    return np.concatenate(actual),np.concatenate(random)


def evaluate(labels,actual,random,model,c):
    rows=[]
    for v in range(3):
        class_means={str(int(k)):float(actual[labels==k,v].mean()) for k in np.unique(labels)}
        active=float(np.mean([value for k,value in class_means.items() if k!='11']));empty=class_means['11']
        rand_active=float(np.mean([random[labels==k,v].mean() for k in np.unique(labels) if k!=11]));rand_empty=float(random[labels==11,v].mean());rank=model[f'v{v}_basis'].shape[1]
        checks=dict(low_rank=rank<=c['maximum_rank'],empty_suppression=empty<=c['maximum_empty_retention'],active_preservation=active>=c['minimum_active_retention'],selectivity=active-empty>=c['minimum_selectivity'],over_random=active-empty-(rand_active-rand_empty)>=c['minimum_selectivity_over_random'])
        rows.append(dict(view=v,rank=rank,empty_retention=empty,active_retention=active,random_empty_retention=rand_empty,random_active_retention=rand_active,class_means=class_means,checks=checks,passed=all(checks.values())))
    return rows
