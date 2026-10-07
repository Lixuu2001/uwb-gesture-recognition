import numpy as np
from scipy.special import huber
from scipy.optimize import check_grad
from huber_readout import objective,weighted_fit
from threadpoolctl import threadpool_limits
with threadpool_limits(limits=1):
 rng=np.random.default_rng(7);x=np.c_[rng.normal(size=(90,5)),np.ones(90)];y=rng.normal(size=90);w=rng.random(90);w/=w.sum();t=rng.normal(size=6);lam=.03;delta=.4
 loss,g=objective(t,x,y,w,delta,lam);np.testing.assert_allclose(loss,w@huber(delta,x@t-y)+lam/2*(t[:-1]@t[:-1]))
 err=check_grad(lambda v:objective(v,x,y,w,delta,lam)[0],lambda v:objective(v,x,y,w,delta,lam)[1],t);assert err<1e-6
 theta=weighted_fit(x,y,w,lam);assert abs(objective(theta,x,y,w,1e6,lam)[1]).max()<1e-12
 print('SciPy loss, finite-difference gradient and quadratic-limit normal equations passed',err)
