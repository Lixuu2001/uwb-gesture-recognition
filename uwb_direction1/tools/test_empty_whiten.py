import numpy as np
from scipy.signal import periodogram
from empty_whiten import normalized,full_mean,features
rng=np.random.default_rng(7);raw=rng.normal(size=(2,3,90,189));c=dict(nfft=512)
np.testing.assert_allclose(features(raw,np.ones((3,257)),c),features(raw,None,c),atol=1e-14,rtol=1e-14)
x=normalized(raw);_,psd=periodogram(x,fs=1,window='boxcar',nfft=512,detrend=False,return_onesided=False,scaling='density',axis=-1);np.testing.assert_allclose(psd[...,:257],abs(np.fft.rfft(x,n=512))**2/189,atol=1e-12,rtol=1e-12)
g=rng.random((3,257));z=np.fft.irfft(g,n=512);np.testing.assert_allclose(np.sum(z*z,axis=-1),full_mean(g*g),atol=1e-14)
print('Identity-filter descriptor, two-sided periodogram and Parseval gain checks passed')
