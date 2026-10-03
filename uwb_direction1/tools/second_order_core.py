"""Independently implemented covariance descriptors and weighted ridge.

Mathematical covariance/spectral square root are established methods, not novel.
No copied fast-MPN-COV implementation; no test-time reference adaptation.
"""
import numpy as np
from scipy.linalg import solve


def projection(seed=20261003):
    q,_=np.linalg.qr(np.random.default_rng(seed).normal(size=(48,16)))
    return q


def describe(z,p):
    # z: N,3,48,spatial_positions. Covariance is within each radar only.
    z=np.asarray(z,dtype=np.float64)
    if z.ndim!=4 or z.shape[1:3]!=(3,48):raise ValueError('Wrong feature-map shape')
    a=np.einsum('nvcq,cd->nvdq',z,p);mu=a.mean(-1);a=a-mu[...,None]
    cov=(a@a.swapaxes(-1,-2))/a.shape[-1]
    eig,vec=np.linalg.eigh(cov)
    if eig.min() < -1e-10*max(1.,abs(eig).max()):raise ValueError('Non-PSD covariance')
    root=(vec*np.sqrt(np.maximum(eig,0))[...,None,:])@vec.swapaxes(-1,-2)
    i,j=np.triu_indices(16);w=np.where(i==j,1.,np.sqrt(2.))
    return {'diag':np.concatenate([mu,np.sqrt(np.maximum(np.diagonal(cov,axis1=-2,axis2=-1),0))],-1).reshape(len(z),-1),
            'cov':np.concatenate([mu,cov[...,i,j]*w],-1).reshape(len(z),-1),
            'sqrt':np.concatenate([mu,root[...,i,j]*w],-1).reshape(len(z),-1)}


def weights(y,u):
    w=np.zeros(len(y),dtype=np.float64)
    for user in np.unique(u):
        for label in np.unique(y[u==user]):
            m=(u==user)&(y==label);w[m]=1/(len(np.unique(u))*len(np.unique(y[u==user]))*m.sum())
    return w


def fit_family(x,y,u,lambdas):
    w=weights(y,u);mean=w@x;scale=np.sqrt(w@((x-mean)**2));scale=np.where(scale>1e-12,scale,1.)
    a=(x-mean)/scale;target=np.eye(12)[y];bias=w@target
    gram=a.T@(w[:,None]*a);rhs=a.T@(w[:,None]*(target-bias));eye=np.eye(a.shape[1])
    coefs=[solve(gram+lam*eye,rhs,assume_a='pos') for lam in lambdas]
    return mean,scale,bias,coefs


def score(x,mean,scale,bias,coef):return ((x-mean)/scale)@coef+bias


def macro(y,pred):return float(np.mean([np.mean(pred[y==c]==c) for c in np.unique(y)]))
