"""Independent temporal circular-convolution/decimation check using pinned filters."""
import json
import numpy as np
from scipy.signal import hilbert
from temporal_scattering import transform,features,profiles


def conv(x,f):
    # Time-domain kernel convolution rather than production Fourier periodization.
    kernel=np.fft.ifft(f)
    out=np.zeros(x.shape,dtype=np.complex128)
    for shift,w in enumerate(kernel):out+=w*np.roll(x,shift,axis=-1)
    return out


def scattering_reference(x):
    s=transform();a=np.pad(x,[(0,0)]*(x.ndim-1)+[(s.pad_left,s.pad_right)],mode='reflect');T=s.phi_f['j'];sl=slice(s.ind_start[T],s.ind_end[T]);out=[conv(a,s.phi_f['levels'][0])[...,::2**T].real[...,sl]];second=[]
    for p1 in s.psi1_f:
        k1=min(p1['j'],T);u=abs(conv(a,p1['levels'][0])[...,::2**k1]);out.append(conv(u,s.phi_f['levels'][k1])[...,::2**(T-k1)].real[...,sl])
        for p2 in s.psi2_f:
            if p2['j']>p1['j']:
                k2=max(min(p2['j'],T)-k1,0);v=abs(conv(u,p2['levels'][k1])[...,::2**k2]);second.append(conv(v,s.phi_f['levels'][k1+k2])[...,::2**(T-k1-k2)].real[...,sl])
    return np.stack(out+second,-2)


def profile_reference(raw):
    x=np.asarray(raw,dtype=np.float64);x=x/np.maximum(np.sqrt(np.sum(x*x,axis=(2,3),keepdims=True)/(90*189)),1e-12);a=abs(hilbert(x,axis=3));return np.stack([np.log1p(a[:,v,:,r*21:r*21+21].sum(-1)/21) for v in range(3) for r in range(9)],axis=1)


def reference(raw,ids,seed=20261007):
    s=scattering_reference(profile_reference(raw));t=s.copy()
    for n,ii in enumerate(ids):
        p=np.random.default_rng(np.random.SeedSequence([seed,int(ii)])).permutation(11)
        for channel in range(27):
            for path in range(5,8):t[n,channel,path]=s[n,channel,path,p]
    def pool(a):return np.stack([a[...,:4].mean(-1),a[...,4:8].mean(-1),a[...,8:11].mean(-1)],-1).reshape(len(a),-1)
    return dict(first=pool(s[:,:,:5]),second=pool(s),shuffled=pool(t),standalone=pool(s))


def main():
    rng=np.random.default_rng(17);x=rng.normal(size=(2,3,90,189));ii=np.array([17,52]);a=features(x,ii);b=reference(x,ii);err=max(float(abs(a[k]-b[k]).max()) for k in a);assert err<1e-11
    for k in a:np.testing.assert_array_equal(a[k],np.concatenate([features(x[:1],ii[:1])[k],features(x[1:],ii[1:])[k]]))
    s=transform()(profiles(x));assert s.shape==(2,27,8,11)
    for n,i in enumerate(ii):
        p=np.random.default_rng(np.random.SeedSequence([20261007,int(i)])).permutation(11);np.testing.assert_array_equal(np.sort(s[n,:,5:,:][:,:,p],axis=-1),np.sort(s[n,:,5:,:],axis=-1))
    for k,v in features(np.zeros_like(x),ii).items():assert np.isfinite(v).all() and not v.any()
    np.testing.assert_allclose(profiles(2*x),profiles(x),atol=1e-14)
    print(json.dumps(dict(passed=True,max_time_domain_reference_error=err,real_data_read=False,dimensions={k:v.shape[1] for k,v in a.items()},checks=['direct circular convolution plus time decimation','independent Hilbert/range pooling','batch independence','second-order time-shuffle multiset','zero input','clip gain normalization'])))
if __name__=='__main__':main()
