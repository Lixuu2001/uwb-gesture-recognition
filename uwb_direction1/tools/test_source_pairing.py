"""Combinatorial invariants, outside-pool isolation and explicit infeasibility."""
import itertools,json
import numpy as np
from source_pairing import same_user_derangement,different_user_permutation


def main():
    cases=0
    for n in range(2,41):
        for seed in range(20):
            y=np.zeros(n,int);u=np.ones(n,int);p=np.arange(n)
            m,info=same_user_derangement(y,u,p,seed)
            assert len(np.unique(m))==n and np.all(m!=p) and info['changed']==n
            cases+=1
    cross_cases=0;infeasible=0
    for counts in itertools.chain.from_iterable(itertools.product(range(1,5),repeat=k) for k in [2,3,4]):
        u=np.repeat(np.arange(len(counts)),counts);y=np.zeros(len(u),int);p=np.arange(len(u))
        for seed in [0,9,2026]:
            if max(counts)*2>sum(counts):
                try:different_user_permutation(y,u,p,seed)
                except ValueError:infeasible+=1
                else:raise AssertionError('Infeasibility ignored')
            else:
                m=different_user_permutation(y,u,p,seed)
                assert np.array_equal(np.sort(m),p) and np.all(u[m]!=u)
                cross_cases+=1
    y=np.repeat([0,1],9);u=np.tile(np.repeat([3,4,5],3),2);pool=np.array([0,1,3,4,6,7,9,10,12,13,15,16]);outside=np.setdiff1d(np.arange(18),pool)
    for fn in [same_user_derangement,different_user_permutation]:
        a=fn(y,u,pool,9);a=a[0] if isinstance(a,tuple) else a
        yy=y.copy();uu=u.copy();yy[outside]=99;uu[outside]=99
        b=fn(yy,uu,pool,9);b=b[0] if isinstance(b,tuple) else b
        assert np.array_equal(a,b) and np.array_equal(a[outside],outside)
    m,info=same_user_derangement(np.array([0,1,1]),np.ones(3),np.arange(3),0)
    assert m.tolist()==[0,2,1] and info['unchanged_singleton_ids']==[0]
    for pool in [np.array([0,0]),np.array([0.,1.]),np.array([-1,0]),np.array([0,18])]:
        try:same_user_derangement(y,u,pool,0)
        except ValueError:pass
        else:raise AssertionError('Invalid pool accepted')
    print(json.dumps({'state':'passed','non_singleton_same_user_cases':cases,'cross_user_feasible_cases':cross_cases,'cross_user_infeasible_cases':infeasible,'outside_pool_labels_users_isolated':True,'singletons_explicit':True,'invalid_pools_rejected':True,'real_model_evaluated':False}))


if __name__=='__main__':main()
