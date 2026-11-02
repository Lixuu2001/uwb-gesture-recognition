"""Small, independently implemented temporal response pooling baseline.
Inspired by MiniRocket/MultiRocket; NOT an exact reproduction or novel method.
"""
import itertools
import numpy as np


def bank(seed=20261004):
    rng=np.random.default_rng(seed)
    triples=np.array(list(itertools.combinations(range(9),3)))
    triples=triples[rng.choice(84,14,replace=False)]
    kernels=-np.ones((14,9),dtype=np.float64)
    for i,tri in enumerate(triples):kernels[i,tri]=2
    channels=np.stack([rng.choice(63,8,replace=False) for _ in range(14)])
    return kernels,channels


def responses(x,kernels,channels):
    # Block order: view, signal, difference, dilation; kernel is within block.
    for view in range(3):
        for signal in [0,2]:
            series=np.take(np.asarray(x[:,view,signal],dtype=np.float64),channels,axis=-1).mean(-1).transpose(0,2,1)
            for difference in [False,True]:
                z=np.diff(series,axis=-1) if difference else series
                for dilation in [1,3,9]:
                    length=z.shape[-1]-8*dilation
                    c=np.zeros((len(z),14,length),dtype=np.float64)
                    for k in range(9):c+=kernels[None,:,k,None]*z[:,:,k*dilation:k*dilation+length]
                    yield c


def pool(c,bias):
    excess=c-bias[None,:,None];positive=excess>0;n=positive.sum(-1);length=c.shape[-1]
    ppv=n/length
    mean=np.maximum(excess,0).sum(-1)/np.maximum(n,1)
    index=(positive*np.arange(length)[None,None,:]).sum(-1)/np.maximum(n,1)/max(length-1,1)
    index=np.where(n>0,index,-1.)
    run=np.zeros(n.shape,dtype=np.int64);longest=run.copy()
    for j in range(length):
        run=np.where(positive[:,:,j],run+1,0);longest=np.maximum(longest,run)
    return np.stack([ppv,mean,index,longest/length],-1)


def permutation(ids,length,seed=20261004):
    return np.stack([np.random.default_rng(np.random.SeedSequence([seed,int(i),length])).permutation(length) for i in ids])


def extract(x,ids,kernels,channels,biases,seed=20261004):
    ppv4=[];multi=[];null=[]
    for block,c in enumerate(responses(x,kernels,channels)):
        b=biases[block] # 14 kernels x 5 quantiles .2,.4,.5,.6,.8
        stats=pool(c,b[:,2]);multi.append(stats.reshape(len(c),-1))
        ppv4.append((c[:,:,:,None]>b[:,[0,1,3,4]][None,:,None,:]).mean(2).reshape(len(c),-1))
        perm=permutation(ids,c.shape[-1],seed)
        shuffled=np.take_along_axis(c,perm[:,None,:],axis=-1)
        control=pool(shuffled,b[:,2]);np.testing.assert_allclose(control[:,:,:2],stats[:,:,:2],rtol=1e-12,atol=1e-12)
        null.append(control.reshape(len(c),-1))
    z=np.concatenate(multi,1)
    return {'ppv_single':z.reshape(len(x),-1,4)[:,:,0].copy(), 'ppv_four':np.concatenate(ppv4,1), 'multi_pool':z, 'temporal_null':np.concatenate(null,1)}


def calibrate(cache,train,kernels,channels,seed=20261004):
    chosen=np.random.default_rng(seed).choice(np.asarray(train),size=(36,14),replace=True)
    unique=np.unique(chosen);lookup={int(i):j for j,i in enumerate(unique)}
    biases=[]
    for block,c in enumerate(responses(cache[unique],kernels,channels)):
        curves=c[[lookup[int(i)] for i in chosen[block]],np.arange(14)]
        biases.append(np.quantile(curves,[.2,.4,.5,.6,.8],axis=1).T)
    return np.stack(biases),chosen


def decision(scores,c):
    out={}
    for control,threshold in c['gate']['minimum_mean_gain'].items():
        delta=np.array(scores['multi_pool'])-scores[control]
        out[control]={'mean_gain':float(delta.mean()),'wins':int((delta>1e-12).sum()),'minimum_gain':float(delta.min()),'passed':bool(delta.mean()>=threshold and (delta>1e-12).sum()>=c['gate']['minimum_wins'] and delta.min()>=-c['gate']['maximum_paired_decline'])}
    return out
