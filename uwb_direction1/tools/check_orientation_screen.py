import json
import numpy as np
from scipy.signal import correlate2d
from orientation_screen import image,tensor,measures,extract


def reference_tensor(im):
    gt=correlate2d(im,np.outer([-1,0,1],[1,2,1])/8,mode='same',boundary='symm')
    gr=correlate2d(im,np.outer([1,2,1],[-1,0,1])/8,mode='same',boundary='symm')
    g=np.exp(-np.arange(-8,9,dtype=float)**2/8);g/=g.sum()
    return np.stack([correlate2d(correlate2d(z,g[None,:],mode='same',boundary='symm'),g[:,None],mode='same',boundary='symm') for z in [gt*gt,gt*gr,gr*gr]],-1)


def main():
    rng=np.random.default_rng(48);im=rng.normal(size=(90,63));err=float(abs(tensor(im)-reference_tensor(im)).max());assert err<1e-12
    zero=measures(tensor(np.zeros((90,63))).reshape(-1,3),0);assert zero['eligible']==0 and zero['fraction']==0
    t,r=np.indices((90,63));z=tensor(r-t)[10:-10,10:-10].reshape(-1,3);m=measures(z,0);assert m['fraction']==1
    static=measures(tensor(r.astype(float))[10:-10,10:-10].reshape(-1,3),0);assert static['fraction']==0
    reverse=tensor(im[::-1])[::-1];original=tensor(im);np.testing.assert_allclose(reverse[...,[0,2]],original[...,[0,2]],atol=1e-12);np.testing.assert_allclose(reverse[...,1],-original[...,1],atol=1e-12)
    raw=rng.normal(size=(3,90,189));np.testing.assert_allclose(image(raw),image(raw*17),atol=1e-12)
    a,p=extract(raw,14);b,q=extract(raw,14);np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(p,q);assert a.shape==(3,2,126,3)
    print(json.dumps({'passed':True,'independent_tensor_max_error':err,'checks':['direct_separable_reference','zero_texture','diagonal_ramp','static_axis_rejection','reversal_tensor_sign','positive_gain_invariance','sample_seed_determinism','grid_shape']}))

if __name__=='__main__':main()
