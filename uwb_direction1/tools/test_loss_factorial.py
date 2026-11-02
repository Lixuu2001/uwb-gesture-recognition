import json,warnings
import numpy as np
from scipy.optimize import check_grad
from sklearn.linear_model import LogisticRegression,Ridge
from loss_factorial import objective,fit,output
from threadpoolctl import threadpool_limits

def main():
 rng=np.random.default_rng(17);z=rng.normal(size=(144,7));y=np.arange(144)%12;w=rng.uniform(.1,2,len(y));w/=w.sum();theta=rng.normal(scale=.1,size=8*12);lam=.01
 error=check_grad(lambda t:objective(t,z,y,w,lam)[0],lambda t:objective(t,z,y,w,lam)[1],theta);assert error<1e-6
 b,info=fit(z,y,w,lam,'logit')
 with warnings.catch_warnings():
  warnings.simplefilter('error');ref=LogisticRegression(C=1/lam,solver='newton-cg',tol=1e-10,max_iter=3000).fit(z,y,sample_weight=w)
 err=float(abs(output(z,b,'logit')-ref.predict_proba(z)).max());assert err<2e-6
 rb,_=fit(z,y,w,lam,'ridge');rr=Ridge(alpha=lam,fit_intercept=True).fit(z,np.eye(12)[y],sample_weight=w);np.testing.assert_allclose(output(z,rb,'ridge'),rr.predict(z),atol=1e-12)
 print(json.dumps(dict(passed=True,finite_difference_error=error,independent_probability_error=err,optimizer=info,real_data_read=False)))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
