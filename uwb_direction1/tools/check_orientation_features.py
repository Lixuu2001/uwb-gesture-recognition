import json
import numpy as np
from orientation_features import describe,histogram


def scalar_hist(angle,w,den):
    out=np.zeros((6,9));mass=np.zeros(6)
    for t in range(14):
        for r in range(9):
            i=t*9+r;cell=(t//7)*3+r//3;v=(angle[i]+np.pi/2)/np.pi*9;lo=int(np.floor(v));f=v-lo
            out[cell,lo%9]+=w[i]*(1-f);out[cell,(lo+1)%9]+=w[i]*f;mass[cell]+=den[i]
    for i,m in enumerate(mass):
        if m>1e-12:out[i]/=m
        else:out[i]=0
    return out


def main():
    rng=np.random.default_rng(504);a=rng.uniform(-np.pi/2,np.pi/2,126);w=rng.random(126);d=rng.random(126);np.testing.assert_allclose(histogram(a,w,d),scalar_hist(a,w,d),atol=1e-12)
    g=rng.normal(size=(3,126,2));z=np.stack([g[:,:,0]**2,g[:,:,0]*g[:,:,1],g[:,:,1]**2],-1);x=describe(z,24);y=describe(z,24)
    for k in x:assert x[k].shape==(162,);np.testing.assert_array_equal(x[k],y[k]);assert np.isfinite(x[k]).all()
    for arm in ['sign_null','cell_null']:np.testing.assert_allclose(x[arm].reshape(3,6,9).sum((1,2)),x['coherent'].reshape(3,6,9).sum((1,2)),atol=1e-12)
    np.testing.assert_allclose(np.sort(x['cell_null'].reshape(3,6,9),axis=1),np.sort(x['coherent'].reshape(3,6,9),axis=1),atol=1e-12)
    for v in describe(np.zeros((3,126,3)),0).values():assert not v.any()
    print(json.dumps({'passed':True,'checks':['scalar_cyclic_histogram','fixed162dimensions','seed_determinism','null_mass_preservation','cell_permutation','zero_input']}))

if __name__=='__main__':main()
