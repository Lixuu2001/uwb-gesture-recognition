"""Classical OAS-QDA, with prespecified covariance controls."""
import numpy as np
from sklearn.covariance import oas


def fit(x,y,seed,jitter):
    means=[];cov=[];shrink=[];counts=[]
    for k in range(12):
        z=x[y==k];assert len(z)>1
        means.append(z.mean(0));v,s=oas(z);cov.append(v+jitter*np.eye(x.shape[1]));shrink.append(s);counts.append(len(z))
    cov=np.array(cov);perm=np.random.default_rng(seed).permutation(12)
    return np.array(means),dict(qda=cov,pooled=np.repeat(cov.mean(0)[None],12,axis=0),diagonal=np.array([np.diag(np.diag(v)) for v in cov]),permuted=cov[perm]),dict(shrinkage=shrink,counts=counts,permutation=perm.tolist())


def predict(x,means,covs):
    out=[]
    for mu,cov in zip(means,covs):
        val,vec=np.linalg.eigh(cov);assert val.min()>0
        z=(x-mu)@vec;out.append(-.5*((z*z/val).sum(1)+np.log(val).sum())-np.log(12))
    return np.array(out).T
