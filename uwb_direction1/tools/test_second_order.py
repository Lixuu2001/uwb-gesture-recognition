"""Numerical and held-label isolation checks before UWB scoring."""
import json
import numpy as np
from scipy.linalg import sqrtm
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from second_order_core import projection,describe,weights,fit_family,score
with threadpool_limits(limits=1):
 rng=np.random.default_rng(123);p=projection();np.testing.assert_allclose(p.T@p,np.eye(16),atol=1e-14)
 z=rng.normal(size=(5,3,48,24));z[0]=0;z[1,:,:,1:]=z[1,:,:,:1]
 d=describe(z,p);i,j=np.triu_indices(16);w=np.where(i==j,1,np.sqrt(2))
 for n in range(5):
  for v in range(3):
   a=p.T@z[n,v];cov=np.cov(a,bias=True);root=sqrtm(cov)
   if np.iscomplexobj(root):assert abs(root.imag).max()<1e-7;root=root.real
   np.testing.assert_allclose(d['cov'][n].reshape(3,-1)[v,16:],cov[i,j]*w,atol=1e-12)
   np.testing.assert_allclose(d['sqrt'][n].reshape(3,-1)[v,16:],root[i,j]*w,atol=1e-7)
 for name,x in describe(z[:,:,:,::-1],p).items():np.testing.assert_allclose(x,d[name],atol=1e-7)
 x=rng.normal(size=(96,20));y=np.arange(96)%12;u=np.arange(96)%4
 pars=fit_family(x,y,u,[.001,.01,.1]);m,s,b,cs=pars
 for lam,coef in zip([.001,.01,.1],cs):
  ref=Ridge(alpha=lam*len(y),solver='cholesky').fit((x-m)/s,np.eye(12)[y],sample_weight=weights(y,u)*len(y))
  np.testing.assert_allclose(score(x,m,s,b,coef),ref.predict((x-m)/s),atol=1e-11)
 # Training API accepts only source rows. Appending/changing held rows cannot enter it.
 fullx=np.concatenate([x,rng.normal(size=(20,20))]);fully=np.r_[y,np.zeros(20,dtype=int)];fullu=np.r_[u,np.full(20,9)]
 before=fit_family(fullx[:96],fully[:96],fullu[:96],[.01]);fullx[96:]=1e8;fully[96:]=11
 after=fit_family(fullx[:96],fully[:96],fullu[:96],[.01]);np.testing.assert_array_equal(before[-1][0],after[-1][0])
 print(json.dumps(dict(state='passed',orthogonal_projection=True,independent_covariance_and_scipy_sqrtm_cases=15,zero_and_rank_deficient_inputs=True,spatial_permutation_invariance=True,sklearn_ridge_differential_fits=3,held_rows_isolated=True,real_uwb_scored=False)))
