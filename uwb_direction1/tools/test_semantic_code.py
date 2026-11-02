import numpy as np
from sklearn.linear_model import Ridge
from semantic_code import codes,decode,choice
from second_order_core import fit_family,score,weights
from threadpoolctl import threadpool_limits

def main():
    rng=np.random.default_rng(45);x=rng.normal(size=(240,9));v=rng.normal(size=(35,9));y=np.arange(240)%12;u=np.arange(240)%5
    cc=codes(20261004);mean,scale,bias,bs=fit_family(x,y,u,[.01]);p=score(v,mean,scale,bias,bs[0]);z=(x-mean)/scale;w=weights(y,u)
    for name,c in cc.items():
        np.testing.assert_allclose(np.square(c).sum(1),1,atol=1e-14)
        reg=Ridge(alpha=.01*len(y),fit_intercept=True,solver='cholesky').fit(z,c[y],sample_weight=w*len(y));vp=reg.predict((v-mean)/scale)
        distance=np.square(vp[:,None,:]-c[None,:,:]).sum(-1)
        np.testing.assert_array_equal(decode(p,c).argmax(1),distance.argmin(1));np.testing.assert_allclose(vp,p@c,atol=1e-12)
    np.testing.assert_allclose(cc['orthogonal']@cc['orthogonal'].T,np.eye(12),atol=1e-14)
    assert choice([{'macro':.8,'mse':.2,'lam':.01},{'macro':.8,'mse':.2,'lam':.1}])==1
    print('PASS independent direct code-target Ridge / explicit distances / unit norm / orthogonal equivalence / tie policy')
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
