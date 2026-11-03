"""Source-only roles and gate implementation; contains no data loading/fitting."""
import numpy as np
COUNTS=[1200,1000,1200,1200,1200,1100]
def roles(ids,users,held):
    ids=np.asarray(ids);users=np.asarray(users)
    assert held in range(1,7) and ids.shape==users.shape==(6900,)
    assert len(np.unique(ids))==6900 and set(np.unique(users))==set(range(1,7))
    assert [(users==u).sum() for u in range(1,7)]==COUNTS
    val=held%6+1
    result=dict(train=np.flatnonzero((users!=held)&(users!=val)),validation=np.flatnonzero(users==val),diagnostic=np.flatnonzero(users==held))
    assert len(np.unique(np.concatenate(list(result.values()))))==6900
    return result
def improvement_gate(candidate,reference,mean_gain,wins,worst,tol=1e-12):
    c=np.asarray(candidate,dtype=np.float64);r=np.asarray(reference,dtype=np.float64)
    assert c.shape==r.shape==(6,) and np.isfinite(c).all() and np.isfinite(r).all()
    d=c-r
    return bool(d.mean()+tol>=mean_gain and (d>tol).sum()>=wins and d.min()+tol>=worst)
