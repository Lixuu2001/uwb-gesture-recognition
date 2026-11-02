"""Classical convex query-graph smoothing; batch-transductive, not novel."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse
from scipy.sparse.linalg import spsolve


def probabilities(scores):
    p=np.maximum(np.asarray(scores,dtype=np.float64),1e-12);return p/p.sum(1,keepdims=True)


def graphs(x,ids,k=10,seed=20261004):
    if len(np.unique(ids))!=len(ids):raise ValueError('Duplicate sample IDs')
    order=np.argsort(ids);inverse=np.argsort(order);z=np.asarray(x,dtype=np.float64)[order]
    norm=np.linalg.norm(z,axis=1,keepdims=True);z=np.divide(z,norm,out=np.zeros_like(z),where=norm>0)
    n=len(z);dist=cdist(z,z,'sqeuclidean');np.fill_diagonal(dist,np.inf)
    nn=np.argsort(dist,axis=1,kind='stable')[:,:min(k,n-1)]
    a=sparse.csr_matrix((np.ones(nn.size),(np.repeat(np.arange(n),nn.shape[1]),nn.ravel())),shape=(n,n))
    mutual=a.multiply(a.T).tocsr();union=((a+a.T)*.5).tocsr();perm=np.random.default_rng(seed).permutation(n);null=mutual[perm][:,perm]
    return {name:w[inverse][:,inverse].tocsr() for name,w in [('mutual',mutual),('union',union),('permuted',null)]}


def smooth(p,w,mu=1.):
    n=len(p);lap=sparse.diags(np.asarray(w.sum(1)).ravel())-w;system=sparse.eye(n,format='csr')+mu*lap
    f=spsolve(system.tocsc(),p);residual=float(np.max(abs(system@f-p)))
    if residual>1e-10 or f.min()<-1e-12 or np.max(abs(f.sum(1)-1))>1e-10:raise ValueError('Invalid solve')
    return f,dict(residual=residual,minimum=float(f.min()),row_sum_error=float(np.max(abs(f.sum(1)-1))),edges=int(w.nnz//2),mean_degree=float(np.mean(np.asarray(w.sum(1)))))
