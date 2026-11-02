"""Unedited aeon 1.6.0 default-size multivariate ROCKET, explicit seed."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[k]='1'
import sys
sys.path.insert(0,'/workspace/uwb-multirocket-20261029/vendor/site-packages')
import numpy as np
from aeon.classification.convolution_based import RocketClassifier

def make():
    return RocketClassifier(n_kernels=10000, estimator=None, class_weight=None,
                            n_jobs=1, random_state=20261029)

def assert_native(m, fitted=False):
    assert m.n_kernels==10000 and m.estimator is None and m.class_weight is None
    assert m.random_state==20261029 and m.n_jobs==1
    if fitted:
        t,s,r=m.pipeline_.steps[0][1],m.pipeline_.steps[1][1],m.pipeline_.steps[2][1]
        assert t.normalise is True and len(t.kernels[1])==10000
        assert not s.with_mean and s.with_std
        assert r.cv is None and r.scoring is None and r.class_weight is None
        assert r.fit_intercept and np.array_equal(r.alphas,np.logspace(-3,3,10))
        assert r.coef_.shape==(12,20000)

def numpy_features(x,kernels,which):
    """Independent scalar convolution, no aeon/Numba convolution calls."""
    w,l,b,d,p,n,c=kernels
    means=x.mean(-1,keepdims=True);std=x.std(-1,keepdims=True)
    z=(x-means)/np.where(std==0,1,std)
    out=[];a1=a2=0
    for j in range(len(l)):
        L,D,P,N=map(int,(l[j],d[j],p[j],n[j]))
        if j in which:
            W=w[a1:a1+L*N].reshape(N,L);C=c[a2:a2+N]
            values=[]
            for start in range(-P,90+P-(L-1)*D):
                v=np.float32(b[j])
                for t in range(L):
                    ix=start+t*D
                    if 0<=ix<90:
                        for ch in range(N):v=np.float32(v+np.float32(W[ch,t]*z[C[ch],ix]))
                values.append(float(v))
            a=np.array(values);out.extend([np.float32((a>0).mean()),np.float32(a.max())])
        a1+=L*N;a2+=N
    return np.array(out,np.float32)
