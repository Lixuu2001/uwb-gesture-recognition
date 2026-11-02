"""Verify label involution, incomplete pairs and correct phase reversal."""
import json
import numpy as np
from circular_signal import encode,get_batch
from temporal_reversal import Q,eligible,augment
from audit_temporal_reversal import independent_input
from second_order_core import fit_family,score
from threadpoolctl import threadpool_limits


def main():
    np.testing.assert_array_equal(Q[Q[:10]],np.arange(10));assert (Q[10:] == -1).all()
    y=np.r_[np.arange(12),[0,1,2,10,11]];u=np.r_[np.zeros(12,int),np.ones(5,int)]
    mask=eligible(y,u);np.testing.assert_array_equal(np.flatnonzero(mask),np.r_[np.arange(10),12,13])
    rng=np.random.default_rng(43);raw=rng.normal(size=(4,3,90,189)).astype(np.float32)
    a=get_batch(encode(raw[:,:,::-1]),np.arange(4),'legacy').numpy();b=independent_input(raw).numpy();np.testing.assert_allclose(a,b,atol=2e-5,rtol=2e-5)
    naive=get_batch(encode(raw),np.arange(4),'legacy').flip(-2).numpy();assert np.max(abs(naive-a))>.1
    np.testing.assert_array_equal(raw[:,:,::-1][:,:,::-1],raw)
    x=rng.normal(size=(len(y),5));xx,yy,uu=augment(x,y,u,x[mask],x[mask],'unmapped_mix')
    a0=fit_family(x,y,u,[.01]);a1=fit_family(xx,yy,uu,[.01]);query=rng.normal(size=(20,5))
    np.testing.assert_allclose(score(query,*a0[:3],a0[3][0]),score(query,*a1[:3],a1[3][0]),atol=1e-12,rtol=1e-12)
    print(json.dumps(dict(state='passed',raw_reversal_involution=True,incomplete_pair_exclusion=True,independent_phase_error=float(abs(a-b).max()),naive_phase_error=float(abs(naive-a).max()),duplicate_control_equals_ridge=True)))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
