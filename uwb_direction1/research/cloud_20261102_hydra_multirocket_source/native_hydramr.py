import os,sys
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,'/workspace/uwb-multirocket-20261029/vendor/site-packages')
import numpy as np
import torch
torch.set_num_threads(1)
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.transformations.collection.convolution_based._multirocket import _transform_multi
def make():return MultiRocketHydraClassifier(n_kernels=8,n_groups=64,class_weight=None,n_jobs=1,random_state=20261029)
def assert_native(m,fitted=False):
    assert m.n_kernels==8 and m.n_groups==64 and m.class_weight is None and m.n_jobs==1 and m.random_state==20261029
    if fitted:
        a=m._transform_multirocket;r=m.classifier
        assert a.n_kernels==6250 and a.max_dilations_per_kernel==32 and a.n_features_per_kernel==4 and a.normalise is False
        assert m._transform_hydra.max_num_channels==8
        assert r.coef_.shape==(12,53824) and r.cv is None and r.scoring is None and r.class_weight is None and r.fit_intercept
        np.testing.assert_array_equal(r.alphas,np.logspace(-3,3,10));assert m._scale_multirocket.with_mean and m._scale_multirocket.with_std
        assert m._scale_hydra.mask and m._scale_hydra.exponent==4
def components(m,x):
    h=m._transform_hydra.transform(x);mr=m._transform_multirocket
    raw=_transform_multi(np.ascontiguousarray(x),np.diff(x,axis=2),mr.parameter,mr.parameter1,4,mr._indices,mr.random_state_)
    assert raw.shape==(len(x),49728) and np.isfinite(raw).all(),'nonfinite raw MultiRocket core before native fallback'
    np.testing.assert_array_equal(raw,mr.transform(x))
    a=m._scale_hydra.transform(h).numpy();b=m._scale_multirocket.transform(raw)
    return h,raw,np.concatenate([a,b],axis=1)
def independent_first_group(x,h):
    result=[]
    for di,d in enumerate(h.dilations.tolist()):
        for side in range(2):
            source=x if side==0 else np.diff(x,axis=1)
            idx=h.idx[di][side,0].numpy();series=source[idx].sum(axis=0,dtype=np.float32)
            weights=h.W[di,side,:8,0].numpy().astype(np.float64)
            response=np.zeros((8,series.shape[-1]),np.float64)
            for center in range(len(series)):
                for tap in range(9):
                    ix=center+(tap-4)*d
                    if 0<=ix<len(series):response[:,center]+=weights[:,tap]*float(series[ix])
            winners=response.argmax(0);losers=response.argmin(0)
            soft=np.bincount(winners,weights=response.max(0),minlength=8)
            hard=np.bincount(losers,minlength=8).astype(float);result.extend([soft,hard])
    return np.concatenate(result).astype(np.float32)
