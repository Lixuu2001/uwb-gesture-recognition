import json
import numpy as np
from threadpoolctl import threadpool_limits
from empty_subspace import moments,projectors,fractions


def main():
    rng=np.random.default_rng(17);basis=np.linalg.qr(rng.normal(size=(189,3)))[0];x=np.einsum('nvtr,fr->nvtf',rng.normal(size=(12,3,90,3)),basis)
    cov=moments(x,np.arange(12),4);m=projectors(cov,3,dict(seed=20261007,energy_fraction=.9));assert all(m[f'v{v}_basis'].shape[1]==3 for v in range(3))
    actual,_=fractions(x,np.arange(12),m,4);assert abs(actual).max()<1e-12
    scaled=x*np.arange(1,13)[:,None,None,None];np.testing.assert_allclose(moments(scaled,np.arange(12),3),cov,atol=1e-12,rtol=1e-12)
    noise=rng.normal(size=x.shape);orthogonal=noise-(noise@basis)@basis.T;remaining,_=fractions(orthogonal,np.arange(12),m,4);np.testing.assert_allclose(remaining,1,atol=1e-12,rtol=1e-12)
    print(json.dumps(dict(state='passed',in_subspace_max_retention=float(abs(actual).max()),orthogonal_min_retention=float(remaining.min()),per_clip_scale_invariance=True)))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
