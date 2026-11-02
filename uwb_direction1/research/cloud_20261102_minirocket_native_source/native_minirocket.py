import os,sys
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,'/workspace/uwb-multirocket-20261029/vendor/site-packages')
import numpy as np
from aeon.classification.convolution_based import MiniRocketClassifier
def make():return MiniRocketClassifier(n_kernels=10000,max_dilations_per_kernel=32,estimator=None,class_weight=None,n_jobs=1,random_state=20261029)
def assert_native(m,fitted=False):
    assert m.n_kernels==10000 and m.max_dilations_per_kernel==32 and m.estimator is None and m.class_weight is None and m.n_jobs==1 and m.random_state==20261029
    if fitted:
        r=m.estimator_;assert r.coef_.shape==(12,9996) and r.cv is None and r.scoring is None and r.class_weight is None and r.fit_intercept
        np.testing.assert_array_equal(r.alphas,np.logspace(-3,3,10));assert not m._scaler.with_mean and m._scaler.with_std
def independent_ppv(x,trf,which):
    counts,channels,dilations,nfeat,bias=trf.parameters
    ci=fi=comb=0;result=[]
    for j,dilation in enumerate(dilations):
        D=int(dilation);P=4*D
        for k,index in enumerate(trf._indices):
            n=int(counts[comb]);ch=channels[ci:ci+n];ci+=n;comb+=1
            selected=[v for v in which if fi<=v<fi+nfeat[j]]
            if selected:
                values=[];w=-np.ones(9,np.float32);w[index]=2
                points=range(90) if (j+k)%2==0 else range(P,90-P)
                for center in points:
                    v=0.
                    for channel in ch:
                        for tap in range(9):
                            ix=center+(tap-4)*D
                            if 0<=ix<90:v+=float(w[tap])*float(x[channel,ix])
                    values.append(v)
                for v in selected:result.append(np.float32(np.mean(np.array(values)>bias[v])))
            fi+=int(nfeat[j])
    return np.array(result,np.float32)
