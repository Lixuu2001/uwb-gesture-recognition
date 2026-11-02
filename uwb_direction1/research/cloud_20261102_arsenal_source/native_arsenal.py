"""Unchanged aeon1.6 full Arsenal25x2000/defaultCV-accuracy weighting.

Published ensemble transfer,not novelty or original Java execution parity.
Prior single ROCKET10000/RidgeCV-MSE negative remains closed.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys
sys.path.insert(0,'/workspace/uwb-multirocket-20261029/vendor/site-packages')
import numpy as np
from aeon.classification.convolution_based import Arsenal
def make():
    return Arsenal(n_kernels=2000,n_estimators=25,rocket_transform='rocket',time_limit_in_minutes=0,class_weight=None,n_jobs=1,random_state=20261029,verbose=0)
def assert_native(m,fitted=False):
    assert m.n_kernels==2000 and m.n_estimators==25 and m.rocket_transform=='rocket'
    assert m.time_limit_in_minutes==0 and m.class_weight is None and m.n_jobs==1 and m.random_state==20261029
    if fitted:
        assert m.n_estimators_==25 and len(m.estimators_)==25 and len(m.weights_)==25
        for pipe,weight in zip(m.estimators_,m.weights_):
            t,s,r=[v for _,v in pipe.steps]
            assert len(t.kernels[1])==2000 and t.normalise is True
            assert not s.with_mean and s.with_std and r.scoring=='accuracy' and r.cv is None
            assert r.class_weight is None and r.fit_intercept and np.array_equal(r.alphas,np.logspace(-3,3,10))
            assert float(weight)==float(r.best_score_) and r.coef_.shape==(12,4000)
        assert m._weight_sum==float(np.sum(m.weights_)) and m._weight_sum>0
def canonical_predictions(m,probabilities):
    # Exactly native _predict tie handling ONCE for the full canonical ID order.
    # Calling native predict independently in batches would reset tie RNG eachtime.
    rng=np.random.RandomState(m.random_state)
    return np.asarray([m.classes_[int(rng.choice(np.flatnonzero(p==p.max())))] for p in probabilities])
