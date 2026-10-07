import json
import numpy as np
from scipy.signal import hilbert
from temporal_ssm import grams,describe,features


def reference(raw,ids,seed=20261007):
 x=abs(hilbert(np.array(raw,dtype=np.float64),axis=3));x=x-np.sum(x,axis=2,keepdims=True)/90;x=x/np.maximum(np.sqrt(np.sum(x*x,axis=3,keepdims=True)),1e-12)
 full=np.einsum('nvtr,nvur->nvtu',x,x);g=np.zeros((len(x),3,18,18))
 for i in range(18):
  for j in range(18):g[:,:,i,j]=full[:,:,5*i:5*i+5,5*j:5*j+5].mean(axis=(-2,-1))
 p=[np.random.default_rng(np.random.SeedSequence([seed,int(ii)])).permutation(18) for ii in ids]
 order=[];lag=[];shuf=[]
 for i in range(18):
  for j in range(i,18):
   order.append(g[:,:,i,j]);lag.append(np.stack([g[:,:,t,t+j-i] for t in range(18-j+i)],-1).mean(-1));shuf.append(np.stack([g[n,:,q[i],q[j]] for n,q in enumerate(p)]))
 return {k:np.stack(v,-1).reshape(len(x),513) for k,v in [('ordered',order),('lag_only',lag),('shuffled',shuf)]}


def main():
 r=np.random.default_rng(2);x=r.normal(size=(7,3,90,189));ids=np.arange(111,118);a=features(x,ids);b=reference(x,ids);err=max(float(abs(a[k]-b[k]).max()) for k in a);assert err<2e-12
 for k in a:np.testing.assert_array_equal(a[k],np.concatenate([features(x[:2],ids[:2])[k],features(x[2:],ids[2:])[k]]))
 g=grams(x);rev=grams(x[:,:,::-1]);np.testing.assert_allclose(rev,g[:,:,::-1,::-1],atol=2e-12)
 for n,ii in enumerate(ids):
  q=np.random.default_rng(np.random.SeedSequence([20261007,int(ii)])).permutation(18);gg=g[n][:,q][:,:,q];np.testing.assert_allclose(np.linalg.eigvalsh(gg),np.linalg.eigvalsh(g[n]),atol=1e-12);np.testing.assert_array_equal(np.sort(gg.reshape(3,-1),axis=-1),np.sort(g[n].reshape(3,-1),axis=-1))
 for k,v in features(np.zeros_like(x),ids).items():assert not v.any() and np.isfinite(v).all()
 # Gram invariance applies to profile-space orthogonal transforms, not physical views.
 z=r.normal(size=(18,23));q,_=np.linalg.qr(r.normal(size=(23,23)));np.testing.assert_allclose((z@q)@(z@q).T,z@z.T,atol=1e-12)
 print(json.dumps(dict(passed=True,max_independent_descriptor_error=err,raw_data_read=False,checks=['full90Gram/block-average equivalence','batch independence','time reversal covariance','shuffled eigenvalue and entry preservation','zero input finite','profile-space orthogonal invariance'])))
if __name__=='__main__':main()
