"""Scalar independent convolution/pooling and role/seed sanity checks."""
import json,itertools
import numpy as np
from response_pool import bank,responses,pool,extract,calibrate
from energy_area import source_roles


def scalar(curve,b):
    values=[(j,float(x-b)) for j,x in enumerate(curve) if x>b]
    run=longest=0
    for x in curve:
        run=run+1 if x>b else 0;longest=max(run,longest)
    return [len(values)/len(curve),sum(v for _,v in values)/len(values) if values else 0,sum(j for j,_ in values)/len(values)/max(len(curve)-1,1) if values else -1,longest/len(curve)]


def main():
    rng=np.random.default_rng(919);k,ch=bank();assert k.shape==(14,9) and np.all(k.sum(1)==0)
    assert all(len(set(x))==8 for x in ch)
    x=rng.normal(size=(3,3,6,90,63));count=0;err=0.
    for block,c in enumerate(responses(x,k,ch)):
        v=block//12;s=[0,2][(block//6)%2];diff=(block//3)%2;d=[1,3,9][block%3]
        for n,j in [(0,0),(2,13)]:
            z=np.array([sum(float(x[n,v,s,t,r]) for r in ch[j])/8 for t in range(90)])
            if diff:z=np.array([z[t+1]-z[t] for t in range(len(z)-1)])
            ref=np.array([sum(float(k[j,q])*z[t+q*d] for q in range(9)) for t in range(len(z)-8*d)])
            e=float(abs(ref-c[n,j]).max());err=max(err,e);assert e<1e-12;count+=len(ref)
    for curve,b in [([0,0,0],0),([1,1,1],0),([-1,2,2,0,1],0),([0,1,0,1,1],1),([1],0)]:
        np.testing.assert_allclose(pool(np.array(curve,dtype=float)[None,None,:],np.array([b]))[0,0],scalar(curve,b),atol=1e-14)
    biases=rng.normal(size=(36,14,5));ids=np.array([41,84,107]);a=extract(x,ids,k,ch,biases)
    for name in a:
        b=np.concatenate([extract(x[i:i+1],ids[i:i+1],k,ch,biases)[name] for i in range(3)])
        np.testing.assert_allclose(a[name],b,atol=1e-13)
    np.testing.assert_allclose(a['multi_pool'].reshape(3,-1,4)[:,:,:2],a['temporal_null'].reshape(3,-1,4)[:,:,:2],atol=1e-13)
    b,chosen=calibrate(x,np.array([0,2]),k,ch);assert set(chosen.flatten())<={0,2}
    altered=x.copy();altered[1]=1e9;b2,chosen2=calibrate(altered,np.array([0,2]),k,ch)
    np.testing.assert_array_equal(chosen,chosen2);np.testing.assert_array_equal(b,b2)
    y=np.tile(np.arange(12),8);u=np.repeat(np.arange(1,9),12)
    for h in range(3,9):
        p=source_roles(y,u,h);sets=[set(z) for z in p.values()]
        assert all(not(a&b) for a,b in itertools.combinations(sets,2));assert set(u[p['train']])<=set(range(3,9))-{h,3 if h==8 else h+1}
    print(json.dumps({'passed':True,'scalar_convolution_points':count,'max_error':err,'checks':['zero_sum_bank','distinct_channels','scalar_valid_dilated_difference_convolution','empty_all_equal_endpoint_run','batch_invariance','null_marginal_preservation','bias_training_only','six_source_roles']}))

if __name__=='__main__':main()
