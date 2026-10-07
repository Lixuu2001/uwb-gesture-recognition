"""Training-empty spectral coloring; no query labels or learned CNN modifications."""
import numpy as np


def normalized(raw):
 x=np.asarray(raw,dtype=np.float64);rms=np.sqrt(np.mean(x*x,axis=(-2,-1),keepdims=True));return x/np.maximum(rms,1e-12)


def full_mean(a):return (a[...,0]+a[...,-1]+2*a[...,1:-1].sum(-1))/512


def template(raw,ids,users,held,c):
 per_user=[]
 for u in sorted(set(users[ids])):
  ii=ids[users[ids]==u];total=np.zeros((3,257))
  for st in range(0,len(ii),c['batch_size']):
   z=np.fft.rfft(normalized(raw[ii[st:st+c['batch_size']]]),n=c['nfft'],axis=-1);total+=(abs(z)**2/189).mean(-2).sum(0)
  per_user.append(total/len(ii))
 psd=np.mean(per_user,axis=0);gain=1/np.sqrt(np.maximum(psd+c['shrinkage']*full_mean(psd)[:,None],1e-30));gain/=np.sqrt(full_mean(gain**2))[:,None];shuffled=gain.copy()
 for v in range(3):
  ix=np.random.default_rng(np.random.SeedSequence([c['seed'],held,v])).permutation(np.arange(1,256));shuffled[v,1:-1]=gain[v,ix]
 return dict(psd=psd,gain=gain,shuffled_gain=shuffled,user_psd=np.stack(per_user),empty_ids=ids)


def features(raw,gain,c):
 x=normalized(raw)
 if gain is not None:x=np.fft.irfft(np.fft.rfft(x,n=c['nfft'],axis=-1)*gain[None,:,None,:],n=c['nfft'],axis=-1)[...,:189]
 e=np.sqrt(np.mean(x.reshape(len(x),3,9,10,21,9)**2,axis=(3,5)))
 return np.log1p(e).reshape(len(x),567)
