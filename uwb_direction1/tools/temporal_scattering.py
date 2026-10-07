"""Fixed Kymatio temporal scattering baseline; no learned filters or novelty claim."""
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np
import kymatio
# Import only the1D frontend: scipy1.17 removed sph_harm used by unrelated3D imports.
from kymatio.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D


@lru_cache(None)
def transform():
    assert kymatio.__version__ == '0.3.0'
    s=ScatteringNumPy1D(J=3,shape=90,Q=1,T=8,max_order=2,oversampling=0,out_type='array')
    assert s.meta()['order'].tolist()==[0,1,1,1,1,2,2,2]
    return s


def package_hashes():
    root=Path(kymatio.__path__[0])
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*.py'))}


def profiles(raw):
    x=np.asarray(raw,dtype=np.float64)
    if x.ndim!=4 or x.shape[1:]!=(3,90,189):raise ValueError('Expected N,3,90,189')
    x=x/np.maximum(np.sqrt(np.mean(x*x,axis=(-2,-1),keepdims=True)),1e-12)
    h=np.zeros(189);h[0]=1;h[1:95]=2
    env=abs(np.fft.ifft(np.fft.fft(x,axis=-1)*h,axis=-1))
    band=env.reshape(len(x),3,90,9,21).mean(-1)
    return np.log1p(band).transpose(0,1,3,2).reshape(len(x),27,90)


def pool(s):
    assert s.shape[-1]==11
    return np.stack([v.mean(-1) for v in np.array_split(s,3,axis=-1)],-1)


def features(raw,ids,seed=20261007):
    s=transform()(profiles(raw));assert s.shape[1:]==(27,8,11)
    shuffled=s.copy()
    for n,ii in enumerate(ids):
        p=np.random.default_rng(np.random.SeedSequence([seed,int(ii)])).permutation(11)
        shuffled[n,:,5:,:]=s[n,:,5:,:][:,:,p]
    full=pool(s).reshape(len(s),648)
    return dict(first=pool(s[:,:,:5]).reshape(len(s),405),second=full,
                shuffled=pool(shuffled).reshape(len(s),648),standalone=full)
